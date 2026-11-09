import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from pathlib import Path
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, CropForegroundd,
    SpatialPadd, ScaleIntensityRanged
)
from monai.data import NibabelReader

from models.segmentation3d import load_3d_segmentation_model




IMAGE_PATH = ROOT / "data" / "Task06_Lung" / "imagesTr" / "lung_001.nii" / "lung_001.nii"

LABEL_PATH = ROOT / "data" / "Task06_Lung" / "labelsTr" / "lung_001.nii" / "lung_001.nii"
MODEL_PATH = ROOT / "models" / "best_3d_unet_v3.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLD = 0.7
PATCH_SIZE = (96, 96, 96)


preprocess = Compose([
    LoadImaged(
        keys=("image", "label"),
        reader=NibabelReader()
    ),
    EnsureChannelFirstd(keys=("image", "label")),
    Orientationd(keys=("image", "label"), axcodes="RAS"),
    Spacingd(
        keys=("image", "label"),
        pixdim=(1.5, 1.5, 2.0),
        mode=("bilinear", "nearest")
    ),
    CropForegroundd(
        keys=("image", "label"),
        source_key="image",
        margin=8
    ),
    SpatialPadd(
        keys=("image", "label"),
        spatial_size=PATCH_SIZE
    ),
    ScaleIntensityRanged(
        keys="image",
        a_min=-1000,
        a_max=400,
        b_min=0.0,
        b_max=1.0,
        clip=True
    )
])

def main():

    print("Loading 3D U-Net...")

    model = load_3d_segmentation_model(
        MODEL_PATH,
        DEVICE
    )

    print("Preprocessing CT and ground truth...")

    data = preprocess({
        "image": str(IMAGE_PATH),
        "label": str(LABEL_PATH)
    })

    image = data["image"].unsqueeze(0).float().to(DEVICE)

    ground_truth = (
        data["label"][0].cpu().numpy() > 0
    )

    print("Running segmentation inference...")

    with torch.inference_mode():

        logits = sliding_window_inference(
            image,
            PATCH_SIZE,
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
            mode="gaussian"
        )

        probabilities = torch.softmax(
            logits.float(), dim=1
        )[0, 1]

    predicted_mask = (
        probabilities > THRESHOLD
    ).cpu().numpy()

    
    intersection = (
        predicted_mask & ground_truth
    ).sum()

    total = predicted_mask.sum() + ground_truth.sum()

    dice = (
        2.0 * intersection / total
        if total > 0 else 1.0
    )

   
    print(f"CT scan: lung_001.nii")
    print(f"Device: {DEVICE}")
    print(f"Prediction voxels: {predicted_mask.sum()}")
    print(f"Ground truth voxels: {ground_truth.sum()}")
    print(f"Intersection voxels: {intersection}")
    print(f"Dice score: {dice:.4f}")
    

if __name__ == "__main__":
    main()