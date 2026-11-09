from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preparation import load_data, validate_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def engineer_features(df):
    """
    Create transaction-level features for fraud detection.

    Note:
    Only information available when a transaction is scored
    should be used in a real-time production model.
    """

    features = df.copy()

    
    features["amount_log"] = np.log1p(features["amount"])

    # Transaction type
    features["type"] = features["type"].astype("category")

    #Origin account balance change
    features["origin_balance_change"] = (
        features["oldbalanceOrg"] - features["newbalanceOrig"]
    )

    #Destination account balance change
    features["destination_balance_change"] = (
        features["newbalanceDest"] - features["oldbalanceDest"]
    )

    
    features["amount_to_origin_balance"] = (
        features["amount"]
        / (features["oldbalanceOrg"] + 1)
    )

    
    features["amount_exceeds_origin_balance"] = (
        features["amount"] > features["oldbalanceOrg"]
    ).astype("int8")

    # Flag transactions involving an empty origin balance
    features["origin_balance_is_zero"] = (
        features["oldbalanceOrg"] == 0
    ).astype("int8")

    #Transaction type indicators
    features["is_transfer"] = (
        features["type"].astype(str) == "TRANSFER"
    ).astype("int8")

    features["is_cash_out"] = (
        features["type"].astype(str) == "CASH_OUT"
    ).astype("int8")

    #Transaction time
    features["transaction_step"] = features["step"]

    
    return features


def prepare_model_data(df):
    

    features = engineer_features(df)

    excluded_columns = [
    "isFraud",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
    "type",
    "newbalanceOrig",
    "newbalanceDest",
    "origin_balance_change",
    "destination_balance_change",
]

   
    features = pd.get_dummies(
        features,
        columns=["type"],
        dtype="int8",
    )

   
    X = features.drop(
        columns=excluded_columns,
        errors="ignore",
    )

    y = df["isFraud"].astype("int8")

    return X, y


def save_processed_data(X, y):
    
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_path = PROCESSED_DATA_DIR / "features.parquet"
    y_path = PROCESSED_DATA_DIR / "labels.parquet"

    X.to_parquet(X_path, index=False)
    y.to_frame(name="isFraud").to_parquet(
        y_path,
        index=False,
    )

    print(f"Features saved to: {X_path}")
    print(f"Labels saved to: {y_path}")


if __name__ == "__main__":
    print("Loading transaction dataset...")

    transactions = load_data()
    validate_data(transactions)

    print("Engineering features...")

    X, y = prepare_model_data(transactions)

    print("\nFEATURE ENGINEERING SUMMARY")
    print("-" * 40)
    print(f"Transactions: {len(X):,}")
    print(f"Model features: {X.shape[1]}")
    print(f"Fraudulent transactions: {int(y.sum()):,}")
    print(f"Fraud percentage: {y.mean() * 100:.4f}%")

    print("\nSample model features:")
    print(X.head())

    save_processed_data(X, y)