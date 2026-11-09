
from pathlib import Path

import pandas as pd

from src.data_preparation import load_data, validate_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    print("Loading transaction data...")

    df = load_data()
    validate_data(df)

    # Re-create the engineered feature using pre-transaction data.
    df["amount_exceeds_origin_balance"] = (
        df["amount"] > df["oldbalanceOrg"]
    ).astype("int8")

    # 1. Fraud rate by transaction type.
    type_summary = (
        df.groupby("type", observed=True)
        .agg(
            transactions=("isFraud", "size"),
            fraud_cases=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
        )
        .reset_index()
    )

    type_summary["fraud_rate_percent"] = (
        type_summary["fraud_rate"] * 100
    )

    print("\nFRAUD RATE BY TRANSACTION TYPE")
    print("-" * 65)
    print(
        type_summary[
            ["type", "transactions", "fraud_cases", "fraud_rate_percent"]
        ].to_string(index=False, formatters={
            "fraud_rate_percent": "{:.4f}%".format
        })
    )

    # 2. Fraud rate by amount-versus-origin-balance condition.
    balance_summary = (
        df.groupby("amount_exceeds_origin_balance")
        .agg(
            transactions=("isFraud", "size"),
            fraud_cases=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
        )
        .reset_index()
    )

    balance_summary["fraud_rate_percent"] = (
        balance_summary["fraud_rate"] * 100
    )

    print("\nFRAUD RATE BY AMOUNT/ORIGIN-BALANCE CONDITION")
    print("-" * 65)
    print(
        balance_summary[
            [
                "amount_exceeds_origin_balance",
                "transactions",
                "fraud_cases",
                "fraud_rate_percent",
            ]
        ].to_string(index=False, formatters={
            "fraud_rate_percent": "{:.4f}%".format
        })
    )

    # 3. Check the condition specifically for fraud and legitimate cases.
    print("\nCONDITION DISTRIBUTION BY CLASS")
    print("-" * 65)

    condition_counts = pd.crosstab(
        df["isFraud"],
        df["amount_exceeds_origin_balance"],
        normalize="index",
    ) * 100

    condition_counts.index = ["Legitimate", "Fraud"]
    condition_counts.columns = [
        "Amount <= starting balance",
        "Amount > starting balance",
    ]

    print(condition_counts.round(2).to_string())

    # Save reports.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    type_summary.to_csv(
        REPORTS_DIR / "fraud_by_transaction_type.csv",
        index=False,
    )

    balance_summary.to_csv(
        REPORTS_DIR / "fraud_by_balance_condition.csv",
        index=False,
    )

    print("\nAnalysis reports saved in:", REPORTS_DIR)


if __name__ == "__main__":
    main()
