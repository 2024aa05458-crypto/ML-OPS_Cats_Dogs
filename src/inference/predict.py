import torch
from PIL import Image
from torchvision import transforms
from src.models.model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict_image(image_path):

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
