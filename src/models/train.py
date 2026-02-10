import torch
import torch.nn as nn
import torch.optim as optim

from src.data.preprocess import get_data_loaders
from src.models.model import SimpleCNN


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_data_loaders("data/raw")

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 2  # keep small for now

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "model.pt")
    print("Model saved as model.pt")


if __name__ == "__main__":
    train()
