from fastapi import FastAPI, UploadFile, File, Request
import shutil
import os
import time
import logging
import csv
from datetime import datetime

from src.inference.predict import predict_image

app = FastAPI()

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------
# In-Memory Metrics
# ---------------------------
request_count = 0
total_latency = 0.0


# ---------------------------
# Middleware for Monitoring
# ---------------------------
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    global request_count
    global total_latency

    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    request_count += 1
    total_latency += latency

    logger.info(
        f"Request #{request_count} | "
        f"Path: {request.url.path} | "
        f"Method: {request.method} | "
        f"Status: {response.status_code} | "
        f"Latency: {latency:.4f}s"
    )

    return response


# ---------------------------
# Health Endpoint
# ---------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ---------------------------
# Metrics Endpoint
# ---------------------------
@app.get("/metrics")
def metrics():
    avg_latency = total_latency / request_count if request_count > 0 else 0
    return {
        "total_requests": request_count,
        "average_latency_seconds": round(avg_latency, 4)
    }


# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    temp_path = "temp.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(temp_path)

    os.remove(temp_path)

    prediction_id = str(int(time.time() * 1000))

    # Log prediction
    with open("predictions_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            prediction_id,
            datetime.now().isoformat(),
            result["predicted_class"],
            ""  # true label placeholder
        ])

    logger.info(
        f"Prediction logged | ID: {prediction_id} | Class: {result['predicted_class']}"
    )

    return {
        "prediction_id": prediction_id,
        "predicted_class": result["predicted_class"]
    }
@app.post("/submit-label")
def submit_label(prediction_id: str, true_label: str):

    rows = []
    updated = False

    with open("predictions_log.csv", "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for row in rows:
        if row[0] == prediction_id:
            row[3] = true_label
            updated = True

    if not updated:
        return {"message": "Prediction ID not found"}

    with open("predictions_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return {"message": "True label updated successfully"}
@app.get("/performance")
def performance():

    total = 0
    correct = 0

    try:
        with open("predictions_log.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                prediction = row[2]
                true_label = row[3]

                if true_label:
                    total += 1
                    if prediction == true_label:
                        correct += 1

        accuracy = correct / total if total > 0 else 0

        return {
            "evaluated_samples": total,
            "correct_predictions": correct,
            "accuracy": round(accuracy, 4)
        }

    except FileNotFoundError:
        return {"message": "No predictions logged yet"}

