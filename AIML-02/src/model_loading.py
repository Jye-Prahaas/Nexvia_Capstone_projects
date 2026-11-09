import torch
from pathlib import Path

from models.classifier import load_classifier
from models.segmentation import load_segmentation_model

BASE_DIR = Path(__file__).resolve().parent.parent

CLASSIFIER_PATH = BASE_DIR / "models" / "densenet121.pth"
SEGMENTATION_PATH = BASE_DIR / "models" / "unet_segmentation.pth"

device = torch.device("cpu")

print("\nLoading DenseNet121...")

classifier = load_classifier(CLASSIFIER_PATH, device)

print("DenseNet121 loaded successfully!")

print("\nLoading U-Net...")

unet = load_segmentation_model(SEGMENTATION_PATH, device)

print("U-Net loaded successfully!")

print("\nBoth trained models are ready!")