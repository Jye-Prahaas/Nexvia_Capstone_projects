from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "Transactions.csv"
)


def load_data():
    """Load the transaction dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATASET_PATH}"
        )

    return pd.read_csv(DATASET_PATH)


def validate_data(df):
    """Validate required columns and fraud labels."""

    required_columns = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df["isFraud"].isnull().any():
        raise ValueError("The isFraud column contains missing values.")

    if not df["isFraud"].isin([0, 1]).all():
        raise ValueError("The isFraud column must contain only 0 and 1.")

    if df["amount"].isnull().any():
        raise ValueError("The amount column contains missing values.")

    if (df["amount"] < 0).any():
        raise ValueError("The amount column contains negative values.")

    return True


def print_dataset_summary(df):
    """Print basic dataset statistics."""

    total_transactions = len(df)
    fraud_count = int(df["isFraud"].sum())
    fraud_rate = df["isFraud"].mean() * 100

    print("\nDATASET SUMMARY")
    print("-" * 40)
    print(f"Total transactions: {total_transactions:,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Legitimate transactions: {total_transactions - fraud_count:,}")
    print(f"Fraudulent transactions: {fraud_count:,}")
    print(f"Fraud percentage: {fraud_rate:.4f}%")


if __name__ == "__main__":
    transactions = load_data()

    validate_data(transactions)

    print("Dataset loaded and validated successfully.")
    print(f"Dataset path: {DATASET_PATH}")

    print_dataset_summary(transactions)