End-to-End MLOps Pipeline – Cats vs Dogs Classification

This repository implements a complete end-to-end MLOps workflow for an image classification problem (Cats vs Dogs). The objective of this project is to demonstrate how a machine learning model can be developed, tracked, deployed, containerized, and automatically validated using modern MLOps practices.

The system includes model training using PyTorch, dataset versioning with DVC, experiment tracking with MLflow, a REST inference API built using FastAPI, Docker-based containerization, and automated testing using GitHub Actions.

Project Structure

ML-OPS_Cats_Dogs/
│
├── data/ # Dataset (tracked using DVC)
├── src/
│ ├── data/ # Data preprocessing
│ ├── models/ # CNN model & training script
│ └── inference/ # API & prediction logic
│
├── tests/ # Unit tests
├── .github/workflows/ # CI configuration
├── Dockerfile # Docker configuration
├── requirements.txt # Dependencies
├── pytest.ini
└── README.md

Technologies Used

Python 3.10+

PyTorch

DVC

MLflow

FastAPI

Docker

Pytest

GitHub Actions

How to Run the Entire Project
1. Clone the Repository

git clone <your-repository-url>
cd ML-OPS_Cats_Dogs

2. Install Dependencies

Ensure Python 3.10 or higher is installed.

pip install -r requirements.txt

3. Train the Model

This step will:

Load dataset

Apply preprocessing

Train the CNN model

Log experiments in MLflow

Save trained weights as model.pt

Run:

python -m src.models.train

After training completes, a file named model.pt will be created in the project root.

4. View Experiment Tracking (MLflow)

Start MLflow UI:

mlflow ui

Open in browser:

http://127.0.0.1:5000

You can inspect:

Training metrics

Hyperparameters

Logged artifacts

5. Run the Inference API Locally (Without Docker)

Start FastAPI server:

uvicorn src.inference.app:app --reload

Open in browser:

Health check:
http://127.0.0.1:8000/health

Swagger UI:
http://127.0.0.1:8000/docs

Use the /predict endpoint to upload an image and receive classification results.

6. Run Using Docker (Recommended)

Build the Docker image:

docker build -t cats-dogs-mlops .

Run the container:

docker run -p 8000:8000 cats-dogs-mlops

Access the API at:

http://localhost:8000/docs

Running Tests

To execute unit tests:

pytest

The tests validate:

Data loader functionality

Prediction output structure

Continuous Integration (CI)

GitHub Actions automatically:

Installs dependencies

Runs unit tests

The pipeline triggers on every push to the master branch.
You can view pipeline status in the "Actions" tab of the repository.

API Endpoints

GET /health
Returns service health status.

GET /metrics
Returns total request count and average latency.

POST /predict
Accepts an image file and returns:

{
"predicted_class": 0,
"probabilities": [[0.85, 0.15]]
}

MLOps Components Implemented

Dataset versioning using DVC

Experiment tracking using MLflow

Modular training pipeline

REST API serving with FastAPI

Logging and basic monitoring

Docker containerization

CI automation using GitHub Actions

Cross-platform test compatibility

Notes

The trained model file (model.pt) is generated after training and is not committed to GitHub.

The CI pipeline is designed to run independently of the dataset and trained weights.

The Docker image excludes the dataset using .dockerignore.