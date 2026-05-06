from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

app = FastAPI()

# Constants
IMG_SIZE = 224
THRESHOLD = 0.4
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_saved")

model = None

@app.on_event("startup")
def load_model():
    global model
    print("BASE_DIR:", BASE_DIR)
    print("MODEL_PATH:", MODEL_PATH)
    print("Files in BASE_DIR:", os.listdir(BASE_DIR))
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully")
    except Exception as e:
        print("ERROR LOADING MODEL:", e)

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

@app.get("/")
def home():
    return {"message": "Solar Panel Defect Detection API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded"}
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img = preprocess_image(image)
    prob = model.predict(img)[0][0]
    prediction = "Defective" if prob > THRESHOLD else "Normal"
    return {
        "prediction": prediction,
        "probability": float(prob)
    }
