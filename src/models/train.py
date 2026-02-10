import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from src.data.preprocess import get_data_loaders
from src.models.model import SimpleCNN


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_data_loaders("data/raw")

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 2

    # Set MLflow experiment
    mlflow.set_experiment("Cats_vs_Dogs_Experiment")

    with mlflow.start_run():

        # Log hyperparameters
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", 32)

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

            avg_loss = running_loss / len(train_loader)

            # Log training loss
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Evaluate on validation set
            val_accuracy = evaluate(model, val_loader, device)

            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Loss: {avg_loss:.4f} "
                  f"Val Accuracy: {val_accuracy:.4f}")

        # Save model
        torch.save(model.state_dict(), "model.pt")

        # Log model artifact
        mlflow.log_artifact("model.pt")

        print("Model saved and logged to MLflow")


if __name__ == "__main__":
    train()
