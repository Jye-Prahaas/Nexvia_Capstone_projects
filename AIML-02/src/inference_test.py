import torch
from pathlib import Path

from models.classifier import load_classifier
from models.segmentation import load_segmentation_model


BASE_DIR = Path(__file__).resolve().parent.parent

CLASSIFIER_PATH = BASE_DIR / "models" / "densenet121.pth"
SEGMENTATION_PATH = BASE_DIR / "models" / "unet_segmentation.pth"

device = torch.device("cpu")


print("Loading DenseNet121...")
classifier = load_classifier(CLASSIFIER_PATH, device)

print("Loading U-Net...")
unet = load_segmentation_model(SEGMENTATION_PATH, device)


classification_input = torch.randn(1, 3, 224, 224)
segmentation_input = torch.randn(1, 1, 256, 256)

with torch.inference_mode():

    classification_output = classifier(classification_input)

    classification_probability = torch.sigmoid(
        classification_output
    )

    segmentation_output = unet(segmentation_input)

    segmentation_probability = torch.sigmoid(
        segmentation_output
    )


print("Classification output shape:",
      classification_output.shape)

print("Classification probability:",
      classification_probability.item())

print("Segmentation output shape:",
      segmentation_output.shape)

print("Segmentation probability shape:",
      segmentation_probability.shape)

print("\nBoth models completed inference successfully!")