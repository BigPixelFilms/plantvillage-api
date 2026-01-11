from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
import io

import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )


# -------------------------------
# Load model ONCE at startup
# -------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = MODEL_PATH = "models/ModelPatato.h5"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)



# CHANGE THESE to match your training labels
CLASS_NAMES = [
    "Early_Blight",
    "Late_Blight",
    "Healthy"
]

app = FastAPI(title="PlantVillage Disease Detection API")

# -------------------------------
# Health check
# -------------------------------
@app.get("/")
def health():
    return {"status": "API running", "model": "ModelPatato.h5"}

# -------------------------------
# Image preprocessing
# -------------------------------
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))  # must match training size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img
from fastapi import UploadFile, File
from PIL import Image
import numpy as np
import tensorflow as tf
import io

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize to model input
        image = image.resize((255, 255))

        # Convert to numpy
        img_array = np.array(image) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {
            "error": str(e)
        }
