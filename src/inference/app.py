from fastapi import FastAPI, UploadFile, File
import shutil
import os

from src.inference.predict import predict_image

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    temp_path = "temp.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict_image(temp_path)

    os.remove(temp_path)

    return result
