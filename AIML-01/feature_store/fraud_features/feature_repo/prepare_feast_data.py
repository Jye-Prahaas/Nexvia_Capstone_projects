from pathlib import Path

import pandas as pd


# Project root:
# Financial_Fraud_Classifier/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "labels.parquet"

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_PATH = OUTPUT_DIR / "fraud_features.parquet"


def main():
    print("Loading engineered features...")

    features = pd.read_parquet(FEATURES_PATH)
    labels = pd.read_parquet(LABELS_PATH)

    if len(features) != len(labels):
        raise ValueError(
            f"Row count mismatch: features={len(features)}, "
            f"labels={len(labels)}"
        )

    # Preserve the original row order from the existing pipeline.
    # This ID identifies rows in this dataset, not real-world transactions.
    features = features.copy()
    features["transaction_id"] = range(len(features))

    # Feast requires an event timestamp.
    # PaySim's step is a simulation time index, not a real timestamp.
    # This synthetic timestamp preserves step ordering for offline retrieval.
    features["event_timestamp"] = (
        pd.Timestamp("2026-01-01", tz="UTC")
        + pd.to_timedelta(features["step"] - 1, unit="h")
    )

    # Keep the target separate from model input features.
    features["isFraud"] = labels["isFraud"].to_numpy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features.to_parquet(OUTPUT_PATH, index=False)

    print("\nFeast source data prepared.")
    print(f"Rows: {len(features):,}")
    print(f"Columns: {len(features.columns)}")
    print(f"Output: {OUTPUT_PATH}")
    print("\nFraud label distribution:")
    print(features["isFraud"].value_counts().to_string())
    print("\nFirst 3 rows:")
    print(features.head(3).to_string())


if __name__ == "__main__":
    main()