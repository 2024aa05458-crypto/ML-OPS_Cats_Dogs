from fastapi import FastAPI, UploadFile, File, Request
import shutil
import os
import time
import logging

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

    logger.info(
        f"Prediction made | Predicted class: {result['predicted_class']}"
    )

    return result
