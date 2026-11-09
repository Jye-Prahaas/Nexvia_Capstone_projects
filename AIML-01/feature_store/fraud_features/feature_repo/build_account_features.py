
from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_DIR.parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Transactions.csv"
FEAST_DATA_PATH = REPO_DIR / "data" / "fraud_features.parquet"
OUTPUT_PATH = REPO_DIR / "data" / "fraud_features.parquet"


def main():
    print("Loading account IDs and transaction history...")

    raw = pd.read_csv(
        RAW_PATH,
        usecols=["step", "amount", "nameOrig"],
        dtype={
            "step": "int32",
            "amount": "float64",
            "nameOrig": "string",
        },
    )

    # Preserve original row order to align with Feast's transaction_id.
    raw["transaction_id"] = np.arange(len(raw), dtype=np.int64)

    print("Aggregating transactions by account and simulation step...")

    # All transactions by an account in the same step receive features
    # based only on transactions from earlier steps.
    account_step = (
        raw.groupby(["nameOrig", "step"], sort=False, observed=True)
        .agg(
            step_transaction_count=("amount", "size"),
            step_amount_sum=("amount", "sum"),
        )
        .reset_index()
        .sort_values(["nameOrig", "step"], kind="mergesort")
    )

    account_groups = account_step.groupby(
        "nameOrig", sort=False, observed=True
    )

    # Exclude the current step to avoid same-step information leakage.
    account_step["previous_transaction_count"] = (
        account_groups["step_transaction_count"].cumsum()
        - account_step["step_transaction_count"]
    )

    account_step["previous_amount_sum"] = (
        account_groups["step_amount_sum"].cumsum()
        - account_step["step_amount_sum"]
    )

    previous_count = account_step["previous_transaction_count"]

    account_step["previous_average_amount"] = np.divide(
        account_step["previous_amount_sum"],
        previous_count,
        out=np.zeros(len(account_step), dtype=np.float64),
        where=previous_count.to_numpy() > 0,
    )

    print("Mapping account history to transactions...")

    raw = raw.merge(
        account_step[
            [
                "nameOrig",
                "step",
                "previous_transaction_count",
                "previous_average_amount",
            ]
        ],
        on=["nameOrig", "step"],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    raw = raw.sort_values("transaction_id", kind="mergesort")

    history = raw[
        [
            "transaction_id",
            "previous_transaction_count",
            "previous_average_amount",
        ]
    ].copy()

    del raw, account_step, account_groups

    print("Loading existing Feast feature data...")
    features = pd.read_parquet(FEAST_DATA_PATH)

    if len(features) != len(history):
        raise ValueError(
            f"Row mismatch: Feast={len(features)}, history={len(history)}"
        )

    if not np.array_equal(
        features["transaction_id"].to_numpy(),
        history["transaction_id"].to_numpy(),
    ):
        raise ValueError("Transaction IDs do not align; refusing to merge.")

    features["previous_transaction_count"] = (
        history["previous_transaction_count"].to_numpy(dtype=np.int64)
    )
    features["previous_average_amount"] = (
        history["previous_average_amount"].to_numpy(dtype=np.float64)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)

    print("\nAccount-history features created successfully.")
    print("Rows:", len(features))
    print(
        features[
            [
                "transaction_id",
                "step",
                "previous_transaction_count",
                "previous_average_amount",
            ]
        ].head(10).to_string(index=False)
    )
    print(
        "\nTransactions with previous history:",
        int((features["previous_transaction_count"] > 0).sum()),
    )
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
     main()
