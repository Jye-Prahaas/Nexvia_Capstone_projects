import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from models.classifier import load_classifier
from models.segmentation import load_segmentation_model

from src.services.classification_service import predict_classification
from src.services.segmentation_service import predict_segmentation
from src.services.Grad_cam import generate_gradcam



BASE_DIR = Path(__file__).resolve().parent.parent

CLASSIFIER_PATH = BASE_DIR / "models" / "densenet121.pth"
SEGMENTATION_PATH = BASE_DIR / "models" / "unet_segmentation.pth"

IMAGE_PATH = BASE_DIR / "samples" / "sample_xray.png"
MASK_PATH = BASE_DIR / "samples" / "sample_mask.png"

device = torch.device("cpu")



print("Loading DenseNet121...")
classifier = load_classifier(CLASSIFIER_PATH, device)

print("Loading U-Net...")
segmentation_model = load_segmentation_model(
    SEGMENTATION_PATH, device
)



print("\nRunning classification...")

classification_result = predict_classification(
    classifier, IMAGE_PATH, device
)

print("\nClassification Result:")
print(classification_result)

# Grad-CAM visualization
print("\nGenerating Grad-CAM heatmap...")

gradcam_result = generate_gradcam(
    classifier,
    IMAGE_PATH,
    device
)

print("\nGrad-CAM Result:")
print("Prediction:", gradcam_result["prediction"])
print("Probability:", gradcam_result["probability"])

# Display original image and heatmap overlay
original_image = gradcam_result["original_image"]
heatmap = gradcam_result["heatmap"]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap="gray")
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(original_image, cmap="gray")
plt.imshow(heatmap, cmap="jet", alpha=0.45)
plt.title("DenseNet121 Grad-CAM")
plt.axis("off")

plt.tight_layout()

# Save visualization
OUTPUT_PATH = BASE_DIR / "reports" / "gradcam_overlay.png"
plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=150)

plt.show()

print("Grad-CAM saved to:", OUTPUT_PATH)


print("\nRunning segmentation...")

predicted_mask = predict_segmentation(
    segmentation_model, IMAGE_PATH, device
)

print("Predicted mask shape:", predicted_mask.shape)
print("Predicted positive pixel percentage:",
      round(predicted_mask.mean() * 100, 2), "%")



xray = Image.open(IMAGE_PATH).convert("L")
ground_truth = Image.open(MASK_PATH).convert("L")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(xray, cmap="gray")
plt.title("Original X-ray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(ground_truth, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(predicted_mask, cmap="gray")
plt.title("U-Net Predicted Mask")
plt.axis("off")

plt.tight_layout()
plt.show()


import numpy as np


ground_truth = Image.open(MASK_PATH).convert("L")
ground_truth = ground_truth.resize(
    (256, 256),
    Image.Resampling.NEAREST
)

gt_mask = (np.array(ground_truth) > 0).astype(np.uint8)
pred_mask = (predicted_mask > 0).astype(np.uint8)


intersection = np.logical_and(gt_mask, pred_mask).sum()

dice = (
    2.0 * intersection /
    (gt_mask.sum() + pred_mask.sum() + 1e-8)
)


union = np.logical_or(gt_mask, pred_mask).sum()

iou = intersection / (union + 1e-8)


print(f"Dice Score: {dice:.4f}")
print(f"IoU Score:  {iou:.4f}")