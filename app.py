from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np

from src.preprocess import clean_raw_df

app = FastAPI()

# Load model and features once on startup
model = joblib.load("models/model.pkl")
feature_cols = joblib.load("models/features.pkl")


app.add_middleware(
    CORSMiddleware,
    allow_origins= ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
class LoanApplication(BaseModel):
    Loan_ID: str
    Gender: str               # "Male" or "Female"
    Married: str              # "Yes" or "No"
    Dependents: Optional[str] # "0", "1", "2", or "3+"
    Education: str            # "Graduate" or "Not Graduate"
    Self_Employed: str        # "Yes" or "No"
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: Optional[float]        # can be None
    Loan_Amount_Term: Optional[float]  # can be None
    Credit_History: Optional[float]    # 1.0, 0.0, or None
    Property_Area: str        # "Urban", "Semiurban", "Rural"
  
@app.post("/predict")
def predict_loan_status(application: LoanApplication):
    df = pd.DataFrame([application.model_dump()])
    df_clean = clean_raw_df(df)
    X_new = df_clean.drop(columns=['Loan_ID', 'Loan_Status'], errors='ignore')
    pred_class = model.predict(X_new)[0]
    pred_prob = model.predict_proba(X_new)[0][1]

    prediction = int(pred_prob >= 0.5)  # Threshold 0.5 , should I use this?

    return {
        "prediction": int(pred_class),
        "probability": float(pred_prob)
    }
   
