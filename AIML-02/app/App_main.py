"""
FastAPI backend for the 3D lung-tumour segmentation model.

Run:   uvicorn main:app --host 0.0.0.0 --port 8000
Needs: best_3d_unet_v3.pth in the same folder (or set CHECKPOINT=/path/to.pth)

Endpoints
  GET  /health                   -> model/device status
  POST /predict  (file=.nii/.nii.gz CT scan)  -> {scan_id, shape, tumour stats}
  GET  /slice/{scan_id}/{z}      -> grayscale PNG of axial slice z
  GET  /mask/{scan_id}/{z}       -> transparent red PNG of predicted tumour
  POST /decision                 -> record radiologist Approve/Reject (audit log)
  GET  /audit?limit=50           -> recent decisions
  GET  /audit/verify             -> check the audit log hash chain
"""
import io
import os
import shutil
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

import audit

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged,
                              Orientationd, ScaleIntensityRanged, Spacingd)


SPACING = (1.5, 1.5, 2.0)
HU_MIN, HU_MAX = -1000, 400
PATCH_SIZE = (96, 96, 96)

CHECKPOINT = os.environ.get("CHECKPOINT", str(Path(__file__).parent / "best_3d_unet_v3.pth"))
THRESHOLD = float(os.environ.get("THRESHOLD", 0.7))   
OVERLAP = float(os.environ.get("OVERLAP", 0.5))
MAX_CACHED_SCANS = 3                                   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(
    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2),
    num_res_units=2, norm="instance", dropout=0.1,
).to(device)

model_loaded = False
if os.path.exists(CHECKPOINT):
    ck = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model_loaded = True
    print(f"Loaded {CHECKPOINT} (epoch {ck.get('epoch')}, val Dice {ck.get('best_dice')})")
else:
    print(f"WARNING: checkpoint not found at {CHECKPOINT}")
model.eval()

preprocess = Compose([
    LoadImaged(keys="image"),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=SPACING, mode="bilinear"),
    ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX,
                         b_min=0.0, b_max=1.0, clip=True),
])

scans = OrderedDict()  
audit.init_db()

app = FastAPI(title="Lung tumour segmentation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded, "device": str(device),
            "threshold": THRESHOLD}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(503, "Model checkpoint not loaded on the server")
    name = file.filename or "scan.nii"
    if not name.endswith((".nii", ".nii.gz")):
        raise HTTPException(400, "Upload a .nii or .nii.gz CT volume")

    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, name)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            data = preprocess({"image": path})
        except Exception as e:
            raise HTTPException(400, f"Could not read NIfTI file: {e}")

        image = data["image"].unsqueeze(0).float().to(device)   # [1,1,X,Y,Z]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = sliding_window_inference(
                image, PATCH_SIZE, sw_batch_size=4, predictor=model,
                overlap=OVERLAP, mode="gaussian")
        prob = torch.softmax(logits.float(), dim=1)[0, 1]
        mask_t = prob > THRESHOLD
        
        confidence = float(prob[mask_t].mean().item()) if mask_t.any() else None
        mask = mask_t.cpu().numpy().astype(np.uint8)
        img = (image[0, 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    scan_id = uuid.uuid4().hex[:12]
    voxels = int(mask.sum())
    voxel_mm3 = SPACING[0] * SPACING[1] * SPACING[2]
    scans[scan_id] = {"img": img, "mask": mask, "filename": name,
                      "confidence": confidence,
                      "volume_ml": round(voxels * voxel_mm3 / 1000.0, 2)}
    while len(scans) > MAX_CACHED_SCANS:
        scans.popitem(last=False)

    slices_with_tumour = np.where(mask.sum(axis=(0, 1)) > 0)[0]
    return {
        "scan_id": scan_id,
        "filename": name,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "shape": list(img.shape),
        "num_slices": int(img.shape[2]),
        "tumour_voxels": voxels,
        "tumour_volume_ml": round(voxels * voxel_mm3 / 1000.0, 2),
        "tumour_detected": voxels > 0,
        "first_tumour_slice": int(slices_with_tumour[0]) if voxels else None,
        "last_tumour_slice": int(slices_with_tumour[-1]) if voxels else None,
        "best_slice": int(np.argmax(mask.sum(axis=(0, 1)))) if voxels else int(img.shape[2] // 2),
        "device": str(device),
    }


def _get(scan_id, z):
    s = scans.get(scan_id)
    if s is None:
        raise HTTPException(404, "Unknown or expired scan_id - upload again")
    if not 0 <= z < s["img"].shape[2]:
        raise HTTPException(404, "Slice out of range")
    return s


def _png(arr, mode):
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/slice/{scan_id}/{z}")
def get_slice(scan_id: str, z: int):
    s = _get(scan_id, z)
    return _png(np.flipud(s["img"][:, :, z].T).copy(), "L")


@app.get("/mask/{scan_id}/{z}")
def get_mask(scan_id: str, z: int):
    s = _get(scan_id, z)
    m = np.flipud(s["mask"][:, :, z].T)
    rgba = np.zeros((*m.shape, 4), dtype=np.uint8)
    rgba[m > 0] = (255, 40, 40, 150)
    return _png(rgba, "RGBA")


class DecisionIn(BaseModel):
    scan_id: str
    decision: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1, max_length=100)
    comment: str = Field(default="", max_length=1000)


@app.post("/decision")
def post_decision(body: DecisionIn):
    s = scans.get(body.scan_id)
    if s is None:
        raise HTTPException(404, "Unknown or expired scan_id - run the scan again")
    try:
        return audit.record_decision(
            scan_id=body.scan_id, decision=body.decision, reviewer=body.reviewer,
            comment=body.comment, filename=s["filename"],
            confidence=s["confidence"], tumour_volume_ml=s["volume_ml"],
            model_checkpoint=os.path.basename(CHECKPOINT))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/audit")
def get_audit(limit: int = 50):
    return audit.list_decisions(max(1, min(limit, 500)))


@app.get("/audit/verify")
def verify_audit():
    ok, n, bad = audit.verify()
    return {"intact": ok, "rows": n, "first_bad_row_id": bad}
