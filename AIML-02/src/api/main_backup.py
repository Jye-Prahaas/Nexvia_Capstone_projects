from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from io import BytesIO

import torch
import numpy as np

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from models.classifier import load_classifier
from models.segmentation import load_segmentation_model

from src.services.classification_service import predict_classification
from src.services.segmentation_service import predict_segmentation
from src.services.Grad_cam import generate_gradcam




BASE_DIR = Path(__file__).resolve().parents[2]

CLASSIFIER_PATH = BASE_DIR / "models" / "densenet121.pth"
SEGMENTATION_PATH = BASE_DIR / "models" / "unet_segmentation.pth"

DEVICE = torch.device("cpu")

classifier = None
segmentation_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, segmentation_model

    print("Loading DenseNet121...")
    classifier = load_classifier(CLASSIFIER_PATH, DEVICE)

    print("Loading U-Net...")
    segmentation_model = load_segmentation_model(
        SEGMENTATION_PATH, DEVICE
    )

    print("Both models loaded successfully!")

    yield

    classifier = None
    segmentation_model = None


app = FastAPI(
    title="AI Medical Imaging API",
    description="Chest X-ray classification and segmentation prototype",
    version="1.0.0",
    lifespan=lifespan
)

def save_uploaded_image(contents):

    temp_file = NamedTemporaryFile(
        suffix=".png",
        delete=False
    )

    temp_file.write(contents)
    temp_file.close()

    return Path(temp_file.name)


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "classifier_loaded": classifier is not None,
        "segmentation_loaded": segmentation_model is not None
    }

@app.post("/predict/classification")
async def classification_endpoint(
    file: UploadFile = File(...)
):

    if classifier is None:
        raise HTTPException(503, "Classifier not loaded")

    image_path = None

    try:
        contents = await file.read()
        image_path = save_uploaded_image(contents)

        result = predict_classification(
            classifier, image_path, DEVICE
        )

        return {
            "task": "classification",
            "result": result
        }

    except Exception as e:
        raise HTTPException(400, str(e))

    finally:
        if image_path and image_path.exists():
            image_path.unlink()


@app.post("/predict/segmentation")
async def segmentation_endpoint(
    file: UploadFile = File(...)
):

    if segmentation_model is None:
        raise HTTPException(503, "Segmentation model not loaded")

    image_path = None

    try:
        contents = await file.read()
        image_path = save_uploaded_image(contents)

        predicted_mask = predict_segmentation(
            segmentation_model, image_path, DEVICE
        )

        mask_image = Image.fromarray(
            (predicted_mask * 255).astype(np.uint8)
        )

        buffer = BytesIO()
        mask_image.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/png"
        )

    except Exception as e:
        raise HTTPException(400, str(e))

    finally:
        if image_path and image_path.exists():
            image_path.unlink()



@app.post("/predict/gradcam")
async def gradcam_endpoint(
    file: UploadFile = File(...)
):

    if classifier is None:
        raise HTTPException(503, "Classifier not loaded")

    image_path = None

    try:
        contents = await file.read()
        image_path = save_uploaded_image(contents)

        result = generate_gradcam(
            classifier, image_path, DEVICE
        )

        original = result["original_image"].astype(np.uint8)
        heatmap = result["heatmap"]

        
        import cv2

        heatmap_uint8 = (heatmap * 255).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
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

        buffer = BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "X-Prediction": result["prediction"],
                "X-Probability": str(result["probability"])
            }
        )

    except Exception as e:
        raise HTTPException(400, str(e))

    finally:
        if image_path and image_path.exists():
            image_path.unlink()