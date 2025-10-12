# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import os
from typing import Tuple
import cv2

app = FastAPI()

from tensorflow.keras.models import load_model

MODEL_PATH = "Model4_23Epochs.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = load_model(MODEL_PATH)


# def preprocess_image_bytes(image_bytes: bytes,
#                            target_size: Tuple[int,int] = (224, 224),
#                            to_rgb: bool = True) -> np.ndarray:
#     try:
#         img = Image.open(BytesIO(image_bytes))
#     except Exception as e:
#         raise ValueError("Cannot open image") from e

#     if to_rgb:
#         img = img.convert("RGB")
#     img = img.resize(target_size)

#     arr = np.array(img).astype(np.float32) / 255.0  
#     if arr.ndim == 3:
#         arr = np.expand_dims(arr, axis=0)  
#     return arr

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")/255.0
    img = np.expand_dims(img, axis=0)
    
    return img

@app.get("/")
def read_root():
    return {"message": "Welcome to the Deepfake Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    print(img)
    # x = preprocess_image_bytes(contents, target_size=(224,224))
    x = preprocess(img)
    

    try:
        pred = model.predict(x)  
        # prob = (pred  > 0.5).astype("int32")
        # prob = 1 if pred >= 0.7 else 0

        print(pred)
        label = ''
        if pred >= 0.5:
            label = "Real"
        else:
            label = "Fake"
        
        print("pred", pred)
        print("Confidence: ", np.round(pred * 100, 2))
        return {"prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


