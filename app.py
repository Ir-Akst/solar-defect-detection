from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

# Load model once
model = tf.keras.models.load_model("model.keras", compile=False)

IMG_SIZE = 224
THRESHOLD = 0.4  # your optimized threshold

# Preprocess function
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "Solar Panel Defect Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = preprocess_image(image)

    prob = model.predict(img)[0][0]

    prediction = "Defective" if prob > THRESHOLD else "Normal"

    return {
        "prediction": prediction,
        "probability": float(prob)
    }
