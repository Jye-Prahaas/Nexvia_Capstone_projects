
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT
    / "feature_store"
    / "fraud_features"
    / "feature_repo"
    / "data"
    / "fraud_features.parquet"
)

MODEL_PATH = ROOT / "models" / "xgboost_fraud_model.joblib"
REPORT_DIR = ROOT / "reports"


# Load the same dataset used by the training script.
data = pd.read_parquet(DATA_PATH)

y = data["isFraud"].astype("int8")

# Match the chronological split used during training.
unique_steps = np.sort(data["step"].unique())
split_position = int(len(unique_steps) * 0.8)
split_position = min(max(split_position, 1), len(unique_steps) - 1)
cutoff_step = unique_steps[split_position]

test_mask = data["step"] >= cutoff_step
test_data = data.loc[test_mask].copy()
y_test = y.loc[test_mask]

# Reconstruct the model inputs in their saved order.
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_names = artifact["feature_names"]

X_test = test_data.drop(
    columns=[
        "transaction_id",
        "event_timestamp",
        "isFraud",
        "transaction_step",
    ],
    errors="ignore",
)

X_test = X_test.reindex(columns=feature_names)

if X_test.isnull().any().any():
    raise ValueError("Missing feature values found in test data.")

# Score the held-out test transactions.
probabilities = model.predict_proba(X_test)[:, 1]
threshold = artifact.get("decision_threshold", 0.5)
predictions = (probabilities >= threshold).astype("int8")

print(f"Test cutoff step: {cutoff_step}")
print(f"Test transactions: {len(y_test):,}")
print(f"Actual fraud cases: {int(y_test.sum()):,}")
print(f"Predicted fraud cases: {int(predictions.sum()):,}")
print(f"Decision threshold: {threshold}")

print("\nConfusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["LEGITIMATE", "FRAUD"],
        zero_division=0,
    )
)

# Save misclassified transactions for inspection.
test_data["actual_label"] = y_test
test_data["predicted_label"] = predictions
test_data["fraud_probability"] = probabilities

errors = test_data[
    test_data["actual_label"] != test_data["predicted_label"]
].copy()

errors["error_type"] = np.where(
    errors["actual_label"] == 1,
    "FALSE_NEGATIVE",
    "FALSE_POSITIVE",
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

error_path = REPORT_DIR / "misclassified_test_transactions.csv"
errors.to_csv(error_path, index=False)

print(f"\nMisclassified transactions: {len(errors):,}")
print(f"Saved misclassified rows to: {error_path}")

if not errors.empty:
    print("\nFirst 10 misclassified transactions:")
    columns = [
        "step",
        "amount",
        "actual_label",
        "predicted_label",
        "fraud_probability",
        "error_type",
    ]
    print(errors[columns].head(10).to_string(index=False))
