from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

app = FastAPI()

# Define request schema
class InputData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "ML model is live"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}
