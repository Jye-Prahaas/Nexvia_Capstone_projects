import asyncio
import io
import json
import shutil
import tempfile
import uuid
import zipfile

from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import torch

from PIL import Image

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from pydantic import BaseModel, Field

from models.classifier import load_classifier
from models.segmentation3d import load_3d_segmentation_model

from src.services.classification_service import predict_classification
from src.services.Grad_cam import generate_gradcam
from src.services.segmentation3d_service import predict_3d_segmentation
from src.services.dicom_service import read_dicom_series

from src.services.orthanc_worker import orthanc_worker
from src.services import audit


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CLASSIFIER_PATH = BASE_DIR / "models" / "densenet121.pth"

SEGMENTATION_PATH = BASE_DIR / "models" / "best_3d_unet_v3.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SPACING = (1.5, 1.5, 2.0)

MAX_CACHED_SCANS = 3

MAX_DICOM_FILES = 2000

MAX_DICOM_SIZE = 500 * 1024 * 1024

classifier = None

segmentation_model = None

scans = OrderedDict()


# ============================================================
# APPLICATION LIFECYCLE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global classifier
    global segmentation_model

    print("Loading DenseNet121...")

    classifier = load_classifier(
        CLASSIFIER_PATH,
        DEVICE
    )

    print("Loading 3D U-Net v3...")

    segmentation_model = load_3d_segmentation_model(
        SEGMENTATION_PATH,
        DEVICE
    )

    audit.init_db()

    print("Both models loaded successfully!")
    print("Device:", DEVICE)

    orthanc_task = asyncio.create_task(
        orthanc_worker()
    )

    print("Orthanc automatic monitoring enabled!")

    yield

    orthanc_task.cancel()

    try:
        await orthanc_task

    except asyncio.CancelledError:
        print("Orthanc worker stopped.")

    classifier = None
    segmentation_model = None

    scans.clear()


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Medical Imaging API",
    description=(
        "Research prototype supporting "
        "chest X-ray classification, Grad-CAM "
        "and 3D CT segmentation."
    ),
    version="2.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "message": "AI Medical Imaging API",
        "status": "running",
        "version": "2.2.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "classifier_loaded": classifier is not None,
        "segmentation_loaded": segmentation_model is not None
    }


# ============================================================
# IMAGE UPLOAD HELPER
# ============================================================

def save_uploaded_image(contents):

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False
    )

    try:
        temp_file.write(contents)

    finally:
        temp_file.close()

    return Path(temp_file.name)


# ============================================================
# CHEST X-RAY CLASSIFICATION
# ============================================================

@app.post("/predict/classification")
async def classification_endpoint(
    file: UploadFile = File(...)
):

    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier not loaded"
        )

    image_path = None

    try:

        contents = await file.read()

        if not contents:
            raise ValueError("Uploaded image is empty.")

        image_path = save_uploaded_image(contents)

        result = predict_classification(
            classifier,
            image_path,
            DEVICE
        )

        return {
            "task": "chest_xray_classification",
            "result": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    finally:

        if image_path and image_path.exists():
            image_path.unlink()

        await file.close()


# ============================================================
# CHEST X-RAY GRAD-CAM
# ============================================================

@app.post("/predict/gradcam")
async def gradcam_endpoint(
    file: UploadFile = File(...)
):

    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier not loaded"
        )

    image_path = None

    try:

        contents = await file.read()

        if not contents:
            raise ValueError("Uploaded image is empty.")

        image_path = save_uploaded_image(contents)

        result = generate_gradcam(
            classifier,
            image_path,
            DEVICE
        )

        import cv2

        original = result["original_image"].astype(
            np.uint8
        )

        heatmap = (
            result["heatmap"] * 255
        ).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        overlay = (
            0.55 * original +
            0.45 * heatmap_color
        ).clip(0, 255).astype(np.uint8)

        buffer = io.BytesIO()

        Image.fromarray(overlay).save(
            buffer,
            format="PNG"
        )

        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "X-Prediction": str(result["prediction"]),
                "X-Probability": str(result["probability"])
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    finally:

        if image_path and image_path.exists():
            image_path.unlink()

        await file.close()


# ============================================================
# STORE 3D SEGMENTATION RESULT
# ============================================================

def store_segmentation_result(result, filename):

    image = np.asarray(result["image"])

    mask = np.asarray(result["mask"])

    gradcam = np.asarray(result["gradcam"])

    # Validate volume dimensions.

    if image.ndim != 3:
        raise ValueError(
            "CT image must be a 3D volume."
        )

    if mask.shape != image.shape:
        raise ValueError(
            "Segmentation mask dimensions do not match CT image."
        )

    if gradcam.shape != image.shape:
        raise ValueError(
            "Grad-CAM dimensions do not match CT image."
        )

    scan_id = uuid.uuid4().hex[:12]

    tumour_voxels = int(np.count_nonzero(mask))

    voxel_volume = (
        SPACING[0] *
        SPACING[1] *
        SPACING[2]
    )

    volume_ml = float(
        round(
            tumour_voxels * voxel_volume / 1000,
            2
        )
    )

    confidence = result.get("confidence")

    if confidence is not None:

        confidence = float(confidence)

        if not np.isfinite(confidence):
            confidence = None

    # Store arrays internally.
    # NEVER return these NumPy arrays directly through FastAPI JSON.

    scans[scan_id] = {
        "image": image,
        "mask": mask,
        "gradcam": gradcam,
        "filename": str(filename),
        "confidence": confidence,
        "volume_ml": volume_ml
    }

    # Keep only the latest scans.

    while len(scans) > MAX_CACHED_SCANS:

        scans.popitem(last=False)

    slices_with_tumour = np.where(
        mask.sum(axis=(0, 1)) > 0
    )[0]

    # Return JSON-compatible metadata only.

    return {
        "scan_id": scan_id,

        "filename": str(filename),

        "shape": [
            int(dim)
            for dim in image.shape
        ],

        "num_slices": int(image.shape[2]),

        "tumour_detected": bool(
            tumour_voxels > 0
        ),

        "tumour_voxels": tumour_voxels,

        "tumour_volume_ml": volume_ml,

        "confidence": confidence,

        "first_tumour_slice": (
            int(slices_with_tumour[0])
            if tumour_voxels > 0
            else None
        ),

        "last_tumour_slice": (
            int(slices_with_tumour[-1])
            if tumour_voxels > 0
            else None
        ),

        "best_slice": (
            int(
                np.argmax(
                    mask.sum(axis=(0, 1))
                )
            )
            if tumour_voxels > 0
            else int(image.shape[2] // 2)
        ),

        "device": str(DEVICE)
    }


# ============================================================
# 3D CT SEGMENTATION - NIFTI
# ============================================================

@app.post("/predict")
async def segmentation_endpoint(
    file: UploadFile = File(...)
):

    if segmentation_model is None:

        raise HTTPException(
            status_code=503,
            detail="3D U-Net not loaded"
        )

    filename = Path(
        file.filename or "scan.nii"
    ).name

    if not filename.lower().endswith(
        (".nii", ".nii.gz")
    ):

        raise HTTPException(
            status_code=400,
            detail="Upload a .nii or .nii.gz CT volume"
        )

    temp_dir = tempfile.mkdtemp()

    try:

        scan_path = Path(temp_dir) / filename

        with open(scan_path, "wb") as f:

            shutil.copyfileobj(
                file.file,
                f
            )

        if scan_path.stat().st_size == 0:

            raise ValueError(
                "Uploaded CT file is empty."
            )

        result = predict_3d_segmentation(
            segmentation_model,
            scan_path,
            DEVICE
        )

        response = store_segmentation_result(
            result,
            filename
        )

        return response

    except Exception as e:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=(
                f"CT segmentation failed: "
                f"{type(e).__name__}: {str(e)}"
            )
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        await file.close()


# ============================================================
# DICOM SERIES UPLOAD AND SEGMENTATION
# ============================================================

@app.post("/predict/dicom-series")
async def dicom_series_endpoint(
    file: UploadFile = File(...)
):

    if segmentation_model is None:

        raise HTTPException(
            status_code=503,
            detail="3D U-Net not loaded"
        )

    filename = Path(
        file.filename or ""
    ).name

    if not filename.lower().endswith(".zip"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a ZIP file containing DICOM slices."
        )

    temp_dir = tempfile.mkdtemp()

    try:

        zip_path = Path(temp_dir) / "dicom_series.zip"

        with open(zip_path, "wb") as f:

            shutil.copyfileobj(
                file.file,
                f
            )

        dicom_dir = Path(temp_dir) / "dicom_files"

        dicom_dir.mkdir()

        # Read and validate ZIP archive.

        with zipfile.ZipFile(zip_path, "r") as archive:

            entries = [
                item
                for item in archive.infolist()
                if not item.is_dir()
            ]

            if len(entries) > MAX_DICOM_FILES:

                raise ValueError(
                    "Too many files in ZIP archive."
                )

            total_size = sum(
                item.file_size
                for item in entries
            )

            if total_size > MAX_DICOM_SIZE:

                raise ValueError(
                    "Extracted archive exceeds 500 MB."
                )

            saved_files = 0

            for index, item in enumerate(entries):

                # Reject symbolic links.

                if (
                    item.external_attr >> 16
                ) & 0o170000 == 0o120000:

                    raise ValueError(
                        "ZIP archive contains an invalid symbolic link."
                    )

                output_path = (
                    dicom_dir /
                    f"slice_{index:06d}.dcm"
                )

                with archive.open(item) as source:

                    with open(output_path, "wb") as target:

                        shutil.copyfileobj(
                            source,
                            target
                        )

                saved_files += 1

        if saved_files < 2:

            raise ValueError(
                "ZIP must contain at least two DICOM files."
            )

        # Read DICOM series.

        series = read_dicom_series(
            dicom_dir
        )

        # Convert DICOM volume into NIfTI.

        nifti_path = (
            Path(temp_dir) /
            "converted_ct.nii.gz"
        )

        nifti_image = nib.Nifti1Image(
            series["volume"],
            series["affine"]
        )

        nib.save(
            nifti_image,
            str(nifti_path)
        )

        # Run 3D U-Net.

        result = predict_3d_segmentation(
            segmentation_model,
            nifti_path,
            DEVICE
        )

        response = store_segmentation_result(
            result,
            f"DICOM CT Series ({series['slices']} slices)"
        )

        return response

    except Exception as e:

        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=(
                f"DICOM processing failed: "
                f"{type(e).__name__}: {str(e)}"
            )
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        await file.close()


# ============================================================
# SCAN RETRIEVAL HELPER
# ============================================================

def get_scan(scan_id, z):

    scan = scans.get(scan_id)

    if scan is None:

        raise HTTPException(
            status_code=404,
            detail="Unknown or expired scan ID"
        )

    if not isinstance(z, int):

        raise HTTPException(
            status_code=400,
            detail="Slice index must be an integer."
        )

    if not 0 <= z < scan["image"].shape[2]:

        raise HTTPException(
            status_code=404,
            detail="Slice out of range"
        )

    return scan


# ============================================================
# PNG RESPONSE HELPER
# ============================================================

def png_response(array):

    buffer = io.BytesIO()

    Image.fromarray(array).save(
        buffer,
        format="PNG"
    )

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )


# ============================================================
# CT SLICE ENDPOINT
# ============================================================

@app.get("/slice/{scan_id}/{z}")
def get_slice(
    scan_id: str,
    z: int
):

    scan = get_scan(
        scan_id,
        z
    )

    image = scan["image"]

    slice_image = np.flipud(
        image[:, :, z].T
    ).copy()

    slice_image = (
        slice_image * 255
    ).clip(0, 255).astype(np.uint8)

    return png_response(
        slice_image
    )


# ============================================================
# SEGMENTATION MASK OVERLAY
# ============================================================

@app.get("/mask/{scan_id}/{z}")
def get_mask(
    scan_id: str,
    z: int
):

    scan = get_scan(
        scan_id,
        z
    )

    mask = np.flipud(
        scan["mask"][:, :, z].T
    ).copy()

    rgba = np.zeros(
        (*mask.shape, 4),
        dtype=np.uint8
    )

    rgba[mask > 0] = (
        255,
        40,
        40,
        150
    )

    return png_response(
        rgba
    )


# ============================================================
# 3D CT GRAD-CAM HEATMAP
# ============================================================

@app.get("/gradcam/{scan_id}/{z}")
def get_gradcam(
    scan_id: str,
    z: int
):

    scan = get_scan(
        scan_id,
        z
    )

    cam = np.flipud(
        scan["gradcam"][:, :, z].T
    ).copy()

    cam = np.nan_to_num(
        cam,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    cam = np.clip(
        cam * 255,
        0,
        255
    ).astype(np.uint8)

    import cv2

    heatmap = cv2.applyColorMap(
        cam,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    rgba = np.zeros(
        (*cam.shape, 4),
        dtype=np.uint8
    )

    rgba[:, :, :3] = heatmap

    rgba[:, :, 3] = (
        cam.astype(np.float32) * 0.65
    ).clip(0, 180).astype(np.uint8)

    return png_response(rgba)


# ============================================================
# REVIEWER DECISION MODEL
# ============================================================

class DecisionIn(BaseModel):

    scan_id: str

    decision: Literal[
        "approved",
        "rejected"
    ]

    reviewer: str = Field(
        min_length=1,
        max_length=100
    )

    comment: str = Field(
        default="",
        max_length=1000
    )


# ============================================================
# REVIEWER DECISION ENDPOINT
# ============================================================

@app.post("/decision")
def post_decision(
    body: DecisionIn
):

    scan = scans.get(
        body.scan_id
    )

    if scan is None:

        raise HTTPException(
            status_code=404,
            detail="Unknown or expired scan ID"
        )

    try:

        return audit.record_decision(
            scan_id=body.scan_id,
            decision=body.decision,
            reviewer=body.reviewer,
            comment=body.comment,
            filename=scan["filename"],
            confidence=scan["confidence"],
            tumour_volume_ml=scan["volume_ml"],
            model_checkpoint=SEGMENTATION_PATH.name
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================
# AUDIT LOG
# ============================================================

@app.get("/audit")
def get_audit(
    limit: int = 50
):

    return audit.list_decisions(
        max(1, min(limit, 500))
    )


# ============================================================
# AUDIT INTEGRITY VERIFICATION
# ============================================================

@app.get("/audit/verify")
def verify_audit():

    ok, rows, bad = audit.verify()

    return {
        "intact": bool(ok),
        "rows": int(rows),
        "first_bad_row_id": bad
    }


# ============================================================
# ORTHANC RESULTS
# ============================================================

@app.get("/orthanc/results")
def get_orthanc_results():

    results_file = (
        BASE_DIR /
        "data" /
        "orthanc_results.json"
    )

    if not results_file.exists():

        return {
            "total": 0,
            "results": {}
        }

    try:

        with open(
            results_file,
            "r",
            encoding="utf-8"
        ) as file:

            results = json.load(file)

        return {
            "total": len(results),
            "results": results
        }

    except json.JSONDecodeError:

        raise HTTPException(
            status_code=500,
            detail="Orthanc results file contains invalid JSON"
        )