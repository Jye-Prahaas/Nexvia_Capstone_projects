

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_fraud_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURE_NAMES = artifact["feature_names"]
DEFAULT_THRESHOLD = artifact.get("decision_threshold", 0.5)

app = FastAPI(
    title="Financial Fraud Detection API",
    description="API for predicting potentially fraudulent transactions.",
    version="1.1.0",
)


class Transaction(BaseModel):
    step: int = Field(ge=1, description="Simulation time step")
    transaction_type: str = Field(
        description="CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER"
    )
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)

    # Supply these from account history; do not assume they are zero.
    previous_transaction_count: int = Field(
        ge=0,
        description="Number of transactions before the current transaction",
    )
    previous_average_amount: float = Field(
        ge=0,
        description="Average amount of prior transactions for this account",
    )


@app.get("/")
def home():
    return {
        "message": "Financial Fraud Detection API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(FEATURE_NAMES),
    }


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    transaction_type = transaction.transaction_type.upper().strip()

    valid_types = {
        "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"
    }

    if transaction_type not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid transaction_type. Choose one of: {sorted(valid_types)}",
        )

    amount = transaction.amount
    oldbalance_org = transaction.oldbalanceOrg
    oldbalance_dest = transaction.oldbalanceDest

    row = {
        "step": transaction.step,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "oldbalanceDest": oldbalance_dest,
        "amount_log": np.log1p(amount),
        "amount_to_origin_balance": amount / (oldbalance_org + 1),
        "amount_exceeds_origin_balance": int(amount > oldbalance_org),
        "origin_balance_is_zero": int(oldbalance_org == 0),
        "is_transfer": int(transaction_type == "TRANSFER"),
        "is_cash_out": int(transaction_type == "CASH_OUT"),
        "transaction_step": transaction.step,
        "type_CASH_IN": int(transaction_type == "CASH_IN"),
        "type_CASH_OUT": int(transaction_type == "CASH_OUT"),
        "type_DEBIT": int(transaction_type == "DEBIT"),
        "type_PAYMENT": int(transaction_type == "PAYMENT"),
        "type_TRANSFER": int(transaction_type == "TRANSFER"),
        "previous_transaction_count": transaction.previous_transaction_count,
        "previous_average_amount": transaction.previous_average_amount,
    }

    missing = set(FEATURE_NAMES) - set(row)
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"API feature-generation mismatch: {sorted(missing)}",
        )

    features = pd.DataFrame(
        [[row[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )

    probability = float(model.predict_proba(features)[0, 1])
    prediction = (
        "FRAUD" if probability >= DEFAULT_THRESHOLD else "LEGITIMATE"
    )

    return {
        "prediction": prediction,
        "fraud_probability": round(probability, 6),
        "threshold": DEFAULT_THRESHOLD,
        "transaction_type": transaction_type,
        "model_version": "xgboost_fraud_model",
        "disclaimer": (
            "Experimental model trained on synthetic PaySim data. "
            "Account-history values must be computed from prior transactions."
        ),
    }
