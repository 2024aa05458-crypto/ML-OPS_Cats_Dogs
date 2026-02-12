import torch
from PIL import Image
from torchvision import transforms
from src.models.model import SimpleCNN
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None  # Lazy loading


def load_model():
    global model

    if model is None:
        model = SimpleCNN().to(device)

        if os.path.exists("model.pt"):
            model.load_state_dict(torch.load("model.pt", map_location=device))
        else:
            # For CI: initialize random weights
            model.eval()

        model.eval()


# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict_image(image_path):

    load_model()

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities.cpu().numpy().tolist()
    }
