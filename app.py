from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json

# -----------------------------
# Load model, scaler, and feature names
# -----------------------------
model = joblib.load("fraud_detector_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

app = FastAPI()

# -----------------------------
# Define expected input schema
# -----------------------------
class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float; V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float; V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(transaction: Transaction):
    # Convert Pydantic model to DataFrame
    df = pd.DataFrame([transaction.dict()])

    # Scale Time and Amount
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])

    # Ensure all features exist and are in correct order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Predict
    pred_proba = model.predict_proba(df)[:, 1][0]
    pred_class = model.predict(df)[0]

    return {
        "fraud_probability": float(pred_proba),
        "predicted_class": int(pred_class)
    }
