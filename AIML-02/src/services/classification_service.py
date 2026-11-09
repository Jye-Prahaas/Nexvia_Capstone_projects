import torch
from PIL import Image
from torchvision import transforms


classification_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_classification(model, image_path, device):

    image = Image.open(image_path).convert("RGB")

    image_tensor = classification_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()

    prediction = (
        "Lung Opacity" if probability >= 0.5
        else "No Lung Opacity"
    )

    return {
        "prediction": prediction,
        "probability": round(probability, 4)
    }