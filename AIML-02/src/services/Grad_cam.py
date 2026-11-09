import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from torchvision import transforms



gradcam_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def generate_gradcam(model, image_path, device):

    model.eval()

    image = Image.open(image_path).convert("RGB")
    original_image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    image_tensor = gradcam_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    activations = {}
    gradients = {}

    
    target_layer = model.features.denseblock4

    def forward_hook(module, inputs, output):
        activations["value"] = output

        output.register_hook(
            lambda grad: gradients.update({"value": grad})
        )

    hook = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():

            output = model(image_tensor)

            probability = torch.sigmoid(output).item()

            if probability >= 0.5:
                prediction = "Lung Opacity"
                score = output[:, 0]
            else:
                prediction = "No Lung Opacity"
                score = -output[:, 0]

            score.backward()

       
        feature_maps = activations["value"]
        grads = gradients["value"]

        
        weights = grads.mean(dim=(2, 3), keepdim=True)

       
        cam = (weights * feature_maps).sum(dim=1)
        cam = F.relu(cam)

        
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "heatmap": cam,
            "original_image": np.array(original_image)
        }

    finally:
        hook.remove()