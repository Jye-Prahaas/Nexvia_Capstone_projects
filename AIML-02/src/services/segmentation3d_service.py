import torch
import torch.nn.functional as F
import numpy as np

from monai.inferers import sliding_window_inference

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd
)

from monai.data import NibabelReader


SPACING = (1.5, 1.5, 2.0)
PATCH_SIZE = (96, 96, 96)

HU_MIN = -1000
HU_MAX = 400

THRESHOLD = 0.7
OVERLAP = 0.5


preprocess = Compose([

    LoadImaged(
        keys="image",
        reader=NibabelReader()
    ),

    EnsureChannelFirstd(
        keys="image"
    ),

    Orientationd(
        keys="image",
        axcodes="RAS"
    ),

    Spacingd(
        keys="image",
        pixdim=SPACING,
        mode="bilinear"
    ),

    CropForegroundd(
        keys="image",
        source_key="image",
        margin=8
    ),

    SpatialPadd(
        keys="image",
        spatial_size=PATCH_SIZE
    ),

    ScaleIntensityRanged(
        keys="image",
        a_min=HU_MIN,
        a_max=HU_MAX,
        b_min=0.0,
        b_max=1.0,
        clip=True
    )

])


def generate_3d_gradcam(model, image, probabilities):

    device = image.device

    # Select the voxel with the highest tumour probability.
    max_position = torch.argmax(probabilities)

    center = np.unravel_index(
        max_position.item(),
        probabilities.shape
    )

    spatial_shape = image.shape[2:]

    starts = []
    ends = []

    # Extract a 96x96x96 patch around the selected voxel.
    for axis in range(3):

        start = max(
            0,
            center[axis] - PATCH_SIZE[axis] // 2
        )

        start = min(
            start,
            spatial_shape[axis] - PATCH_SIZE[axis]
        )

        end = start + PATCH_SIZE[axis]

        starts.append(start)
        ends.append(end)

    slices = tuple(
        slice(starts[i], ends[i])
        for i in range(3)
    )

    patch = image[
        :,
        :,
        slices[0],
        slices[1],
        slices[2]
    ].detach()

    captured = {}

    # First encoder block of the MONAI U-Net.
    target_layer = model.model[0]

    def forward_hook(module, inputs, output):

        if not torch.is_tensor(output):
            raise RuntimeError(
                "Grad-CAM target layer did not return a tensor."
            )

        captured["activation"] = output

        output.register_hook(
            lambda grad: captured.__setitem__(
                "gradient",
                grad
            )
        )

    hook = target_layer.register_forward_hook(
        forward_hook
    )

    try:

        model.zero_grad(set_to_none=True)

        with torch.enable_grad():

            logits = model(patch)

            if "activation" not in captured:
                raise RuntimeError(
                    "Grad-CAM activation was not captured."
                )

            class_difference = (
                logits[:, 1] - logits[:, 0]
            )

            patch_probabilities = torch.softmax(
                logits.detach(),
                dim=1
            )[:, 1]

            roi = (
                patch_probabilities[0] > THRESHOLD
            )

            # Fallback if the selected patch has no
            # voxels above the segmentation threshold.
            if not roi.any():

                best_voxel = torch.argmax(
                    patch_probabilities[0]
                )

                roi = torch.zeros_like(
                    patch_probabilities[0],
                    dtype=torch.bool
                )

                roi.view(-1)[best_voxel] = True

            score = class_difference[0][roi].mean()

            score.backward()

        activations = captured["activation"].detach()

        gradients = captured["gradient"].detach()

        # Global average pooling of gradients.
        weights = gradients.mean(
            dim=(2, 3, 4),
            keepdim=True
        )

        # Weighted combination of feature maps.
        cam = torch.relu(
            (weights * activations).sum(dim=1, keepdim=True)
        )

        # Upsample heatmap to the original patch size.
        cam = F.interpolate(
            cam,
            size=PATCH_SIZE,
            mode="trilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        cam_max = cam.max()

        if cam_max > 0:

            cam = cam / cam_max

        cam = cam.clamp(0, 1)

        # Place the localized CAM into the complete volume.
        full_cam = torch.zeros(
            spatial_shape,
            device=device,
            dtype=torch.float32
        )

        full_cam[
            slices[0],
            slices[1],
            slices[2]
        ] = cam

        return (
            full_cam
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    finally:

        hook.remove()

        model.zero_grad(set_to_none=True)


def predict_3d_segmentation(model, image_path, device):

    data = preprocess({
        "image": str(image_path)
    })

    image = (
        data["image"]
        .unsqueeze(0)
        .float()
        .to(device)
    )

    # Normal 3D segmentation inference.
    with torch.inference_mode():

        with torch.amp.autocast(
            device_type="cuda",
            enabled=device.type == "cuda"
        ):

            logits = sliding_window_inference(
                image,
                PATCH_SIZE,
                sw_batch_size=4,
                predictor=model,
                overlap=OVERLAP,
                mode="gaussian"
            )

        probabilities = torch.softmax(
            logits.float(),
            dim=1
        )[0, 1]

    predicted_mask = (
        probabilities > THRESHOLD
    )

    mask = (
        predicted_mask
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    image_array = (
        image[0, 0]
        .clamp(0, 1)
        .cpu()
        .numpy()
    )

    confidence = (

        float(
            probabilities[predicted_mask]
            .mean()
            .item()
        )

        if predicted_mask.any()
        else None
    )

    # Generate 3D Grad-CAM.
    gradcam = generate_3d_gradcam(
        model,
        image,
        probabilities
    )

    return {

        "image": image_array,

        "mask": mask,

        "confidence": confidence,

        "probability": (
            probabilities
            .cpu()
            .numpy()
        ),

        "gradcam": gradcam

    }