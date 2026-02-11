from fastapi import FastAPI, UploadFile, File
import shutil
import os
import time
import logging

from src.inference.predict import predict_image

app = FastAPI()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Simple in-memory metrics
request_count = 0
total_latency = 0.0


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():
    avg_latency = total_latency / request_count if request_count > 0 else 0
    return {
        "request_count": request_count,
        "average_latency_seconds": avg_latency
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    global request_count
    global total_latency

    start_time = time.time()

    temp_path = "temp.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(temp_path)

    os.remove(temp_path)

    end_time = time.time()
    latency = end_time - start_time

    request_count += 1
    total_latency += latency

    logging.info(f"Prediction made | Latency: {latency:.4f}s | Class: {result['predicted_class']}")

    return result
