from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize app
app = FastAPI(title="FicZon Lead Prediction API")

# Load trained artifacts
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")

# Input schema (matches training columns)
class LeadInput(BaseModel):
    Product_ID: str
    Source: str
    Sales_Agent: str
    Location: str
    Delivery_Mode: str

@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: LeadInput):

    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Encode categorical features
    for col in df.columns:
     if col in encoders:
        le = encoders[col]
        df[col] = df[col].apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        df[col] = le.transform(df[col])


    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    prob = model.predict_proba(X_scaled)[0][1]
    label = "High Potential" if prob >= 0.6 else "Low Potential"

    return {
        "prediction": label,
        "probability": round(prob, 3)
    }
