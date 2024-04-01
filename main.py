from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from io import BytesIO
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Model Loading
model_num = load('model_num.pkl')
model = load_model('model_img.h5')

# CORS Middleware Configuration
origins = ["*"]  
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for data validation
class Item(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Predict endpoint
@app.post('/predict_logisticregression/')
def predict(item: Item):
    data = item.dict()
    list_data = list(data.values())
    prediction = model_num.predict([list_data])
    if int(prediction[0]) == 1:
        res_pred = "ํYou have diabetes."
    else:
        res_pred = "You don't have diabetes."
    return {'prediction': res_pred}

# Classification labels
class_labels = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}

# Classify image endpoint
@app.post("/classify_densenet/")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = Image.open(BytesIO(contents)).convert('RGB')
    img = np.array(pil_image)
    X = cv2.resize(img, (224, 224))
    X = X / 255.0
    X = np.expand_dims(X, axis=0)
    predictions = model.predict(X)
    class_indices = np.argmax(predictions, axis=1)
    predicted_label = class_labels[class_indices[0]]
    return JSONResponse(content={"predicted_label": predicted_label})
