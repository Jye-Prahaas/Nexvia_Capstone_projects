
import json
from pathlib import Path

import joblib
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from scipy.stats import ks_2samp



ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "xgboost_fraud_model.joblib"

REFERENCE_PATH = (
    ROOT
    / "feature_store"
    / "fraud_features"
    / "feature_repo"
    / "data"
    / "fraud_features.parquet"
)

CURRENT_DATA_PATH = (
    ROOT / "data" / "monitoring" / "kafka_transactions.parquet"
)

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "data_drift_report.html"
TRIGGER_PATH = REPORT_DIR / "retraining_trigger.json"


SAMPLE_SIZE = 20_000
DRIFT_THRESHOLD = 0.30
P_VALUE_THRESHOLD = 0.05



print("Starting data drift monitoring...")

for path in [MODEL_PATH, REFERENCE_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

if not CURRENT_DATA_PATH.exists():
    raise FileNotFoundError(
        "Kafka monitoring data not found.\n"
        f"Expected location: {CURRENT_DATA_PATH}\n"
        "The Kafka consumer must first save processed "
        "transaction features to this file."
    )

artifact = joblib.load(MODEL_PATH)
FEATURE_COLUMNS = list(artifact["feature_names"])

if not FEATURE_COLUMNS:
    raise ValueError("The saved model contains no feature names.")

print(f"Model features loaded: {len(FEATURE_COLUMNS)}")



print("\nLoading Feast reference dataset...")

reference_df = pd.read_parquet(REFERENCE_PATH)

print("Loading incoming Kafka monitoring dataset...")

current_df = pd.read_parquet(CURRENT_DATA_PATH)


missing_reference = set(FEATURE_COLUMNS) - set(reference_df.columns)
missing_current = set(FEATURE_COLUMNS) - set(current_df.columns)

if missing_reference:
    raise ValueError(
        f"Reference dataset is missing features: "
        f"{sorted(missing_reference)}"
    )

if missing_current:
    raise ValueError(
        f"Kafka monitoring dataset is missing features: "
        f"{sorted(missing_current)}"
    )


reference_df = reference_df[FEATURE_COLUMNS].copy()
current_df = current_df[FEATURE_COLUMNS].copy()


for column in FEATURE_COLUMNS:
    reference_df[column] = pd.to_numeric(
        reference_df[column], errors="coerce"
    )
    current_df[column] = pd.to_numeric(
        current_df[column], errors="coerce"
    )


reference_df = reference_df.replace(
    [float("inf"), float("-inf")], float("nan")
).fillna(0)

current_df = current_df.replace(
    [float("inf"), float("-inf")], float("nan")
).fillna(0)

if reference_df.empty:
    raise ValueError("Feast reference dataset is empty.")

if current_df.empty:
    raise ValueError("Kafka monitoring dataset is empty.")


reference = reference_df.head(SAMPLE_SIZE).copy()
current = current_df.tail(SAMPLE_SIZE).copy()

print("\nDATASET SUMMARY")
print("-" * 40)
print(f"Reference rows: {len(reference):,}")
print(f"Current rows:   {len(current):,}")
print(f"Features:       {len(FEATURE_COLUMNS)}")


print("\nGenerating Evidently drift report...")

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

snapshot = report.run(
    reference_data=reference,
    current_data=current,
)

snapshot.save_html(str(REPORT_PATH))


print("\nCalculating feature drift...")

drifted_columns = []

for column in FEATURE_COLUMNS:

    reference_values = reference[column].to_numpy()
    current_values = current[column].to_numpy()

   
    if (
        reference[column].nunique() <= 1
        and current[column].nunique() <= 1
    ):
        continue

    _, p_value = ks_2samp(
        reference_values,
        current_values,
    )

    if p_value < P_VALUE_THRESHOLD:
        drifted_columns.append(column)


drift_share = (
    len(drifted_columns) / len(FEATURE_COLUMNS)
)

dataset_drift = drift_share >= DRIFT_THRESHOLD

retraining_required = dataset_drift

trigger = {
    "retraining_required": bool(retraining_required),
    "dataset_drift_detected": bool(dataset_drift),
    "share_of_drifted_columns": round(drift_share, 4),
    "drifted_columns": drifted_columns,
    "feature_count": len(FEATURE_COLUMNS),
    "reference_rows": len(reference),
    "current_rows": len(current),
    "drift_threshold": DRIFT_THRESHOLD,
    "p_value_threshold": P_VALUE_THRESHOLD,
    "drift_method": "Kolmogorov-Smirnov test",
    "reference_dataset": str(REFERENCE_PATH),
    "current_dataset": str(CURRENT_DATA_PATH),
    "report": str(REPORT_PATH),
}

TRIGGER_PATH.write_text(
    json.dumps(trigger, indent=2),
    encoding="utf-8",
)


print("\n" + "=" * 45)
print("DATA DRIFT MONITORING RESULTS")
print("=" * 45)

print(f"Features checked:       {len(FEATURE_COLUMNS)}")
print(f"Drifted features:       {len(drifted_columns)}")
print(f"Drift share:            {drift_share:.2%}")
print(f"Drift threshold:        {DRIFT_THRESHOLD:.0%}")
print(f"Dataset drift detected: {dataset_drift}")
print(f"Retraining required:    {retraining_required}")

print("\nDrifted features:")

if drifted_columns:
    for column in drifted_columns:
        print(f" - {column}")
else:
    print("No drifted features detected.")

print(f"\nEvidently report: {REPORT_PATH}")
print(f"Trigger file:     {TRIGGER_PATH}")

print("\nDrift monitoring completed.")
