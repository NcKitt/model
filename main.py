from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from joblib import load
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from io import BytesIO
from PIL import Image


app = FastAPI()

model_num = load('model_num.pkl')
model = load_model('model_img.h5')

class Item(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin :float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post('/predict/')
def predict(item: Item):
    data = item.dict()
    list_data=list(data.values())
    prediction = model_num.predict([list_data])
    if int(prediction[0])==1:
        res_pred="ํYou have diabetes."
    else:
        res_pred="You don't have diabetes."
    return {'prediction': res_pred}

class_labels = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}

@app.post("/classify/")
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

