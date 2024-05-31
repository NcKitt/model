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
import tensorflow as tf
import io


class DWT2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DWT2D, self).__init__(**kwargs)
        self.db_filter_ll = tf.constant([0.482962913145, 0.836516303738, 0.224143868042, -0.129409522551], dtype=tf.float32)
        self.db_filter_lh = tf.constant([-0.129409522551, -0.224143868042, 0.836516303738, -0.482962913145], dtype=tf.float32)
        self.db_filter_hl = tf.constant([-0.129409522551, 0.224143868042, 0.836516303738, 0.482962913145], dtype=tf.float32)
        self.db_filter_hh = tf.constant([0.482962913145, -0.836516303738, 0.224143868042, 0.129409522551], dtype=tf.float32)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.db_filter_ll = tf.reshape(self.db_filter_ll, [4, 1, 1, 1])
        self.db_filter_ll = tf.tile(self.db_filter_ll, [1, 1, channels, 1])
        self.db_filter_lh = tf.reshape(self.db_filter_lh, [4, 1, 1, 1])
        self.db_filter_lh = tf.tile(self.db_filter_lh, [1, 1, channels, 1])
        self.db_filter_hl = tf.reshape(self.db_filter_hl, [4, 1, 1, 1])
        self.db_filter_hl = tf.tile(self.db_filter_hl, [1, 1, channels, 1])
        self.db_filter_hh = tf.reshape(self.db_filter_hh, [4, 1, 1, 1])
        self.db_filter_hh = tf.tile(self.db_filter_hh, [1, 1, channels, 1])

    def call(self, inputs):
        ll = tf.nn.depthwise_conv2d(inputs, self.db_filter_ll, strides=[1, 2, 2, 1], padding='SAME')
        lh = tf.nn.depthwise_conv2d(inputs, self.db_filter_lh, strides=[1, 2, 2, 1], padding='SAME')
        hl = tf.nn.depthwise_conv2d(inputs, self.db_filter_hl, strides=[1, 2, 2, 1], padding='SAME')
        hh = tf.nn.depthwise_conv2d(inputs, self.db_filter_hh, strides=[1, 2, 2, 1], padding='SAME')
        output = ll
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, input_shape[3])

    def get_config(self):
        config = super(DWT2D, self).get_config()
        return config


# Initialize FastAPI app
app = FastAPI()

# Model Loading
model_num = load('model_num.pkl')
model_img_densenet = load_model('model_image_densenet.h5')
model_img_incept = load_model('model_image_inception.h5', custom_objects={'DWT2D': DWT2D})


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
    predictions = model_img_densenet.predict(X)
    class_indices = np.argmax(predictions, axis=1)
    predicted_label = class_labels[class_indices[0]]
    return JSONResponse(content={"predicted_label": predicted_label})

@app.post("/classify_inceptionplusdwt/")
async def classify_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).resize((128, 128))  # Resize to the expected input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model_img_incept.predict(image)
    predicted_class = np.argmax(predictions, axis=1)
    if int(predicted_class[0]) == 0: 
        class_res ="No DR signs"
    elif int(predicted_class[0]) == 1: 
        class_res ="Mild (or early) NPDR"
    elif int(predicted_class[0]) == 2: 
        class_res ="Moderate NPDR"
    elif int(predicted_class[0]) == 3: 
        class_res ="Severe NPDR"
    elif int(predicted_class[0]) == 4: 
        class_res ="Very Severe NPDR"
    elif int(predicted_class[0]) == 5:
        class_res ="PDR"
    elif int(predicted_class[0]) == 6: 
        class_res ="Advanced PDR"
    return {"filename": file.filename, "predicted_class": class_res}

