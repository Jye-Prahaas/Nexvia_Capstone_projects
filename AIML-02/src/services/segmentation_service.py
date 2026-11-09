import torch
import numpy as np
from PIL import Image


def predict_segmentation(model, image_path, device):

    image = Image.open(image_path).convert("L")

    image = image.resize(
        (256, 256),
        Image.Resampling.BILINEAR
    )

    image_array = np.array(image, dtype=np.float32) / 255.0

    # Match training normalization
    mean = image_array.mean()
    std = image_array.std()

    image_array = (image_array - mean) / (std + 1e-6)

    image_tensor = torch.from_numpy(image_array)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        output = model(image_tensor)
        probability = torch.sigmoid(output)

    predicted_mask = (probability > 0.5).float()

    return predicted_mask.squeeze().cpu().numpy()