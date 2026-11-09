
from pathlib import Path
import joblib
import pandas as pd
import requests
import numpy as np

# Project paths
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "xgboost_fraud_model.joblib"
DATA_PATH = (
    ROOT / "feature_store" / "fraud_features"
    / "feature_repo" / "data" / "fraud_features.parquet"
)
API_URL = "http://127.0.0.1:8000/predict"

print("Loading model and labeled transactions...")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_names = artifact["feature_names"]
threshold = artifact.get("decision_threshold", 0.5)

df = pd.read_parquet(DATA_PATH)

required = {
    "step", "amount", "oldbalanceOrg", "oldbalanceDest",
    "previous_transaction_count", "previous_average_amount", "isFraud"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing dataset columns: {sorted(missing)}")

# Use the same chronological test period as the existing validation.
test = df[df["step"] >= 595].copy()
if test.empty:
    raise ValueError("No test rows found at step >= 595.")

# Build direct-model inputs using the exact saved feature order.
missing_features = set(feature_names) - set(test.columns)
if missing_features:
    raise ValueError(
        f"Dataset is missing model features: {sorted(missing_features)}"
    )

X = test.loc[:, feature_names]
y = test["isFraud"].astype(int).to_numpy()

direct_prob = model.predict_proba(X)[:, 1]
direct_pred = (direct_prob >= threshold).astype(int)

test["_actual"] = y
test["_direct_prob"] = direct_prob
test["_direct_pred"] = direct_pred

# Sample examples from each available outcome group.
groups = {
    "False positives": test[
        (test["_actual"] == 0) & (test["_direct_pred"] == 1)
    ],
    "False negatives": test[
        (test["_actual"] == 1) & (test["_direct_pred"] == 0)
    ],
    "True positives": test[
        (test["_actual"] == 1) & (test["_direct_pred"] == 1)
    ],
    "True negatives": test[
        (test["_actual"] == 0) & (test["_direct_pred"] == 0)
    ],
}

samples = []
for name, group in groups.items():
    if not group.empty:
        samples.append(group.sample(n=min(3, len(group)), random_state=42))

if not samples:
    raise ValueError("Could not select test transactions.")

sample = pd.concat(samples).drop_duplicates()
print(f"Test rows: {len(test):,}")
print(f"Transactions selected for API comparison: {len(sample)}")
print(f"Model threshold: {threshold}")
print("\nCalling the running API...")

type_columns = [
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

results = []

for idx, row in sample.iterrows():
    transaction_type = None

    for col in type_columns:
        if col in row.index and row[col] == 1:
            transaction_type = col.replace("type_", "")
            break

    if transaction_type is None:
        raise ValueError(
            f"Cannot determine transaction type for row {idx}. "
            f"Check the type one-hot columns in the parquet file."
        )

    payload = {
        "step": int(row["step"]),
        "transaction_type": transaction_type,
        "amount": float(row["amount"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "previous_transaction_count": int(
            row["previous_transaction_count"]
        ),
        "previous_average_amount": float(
            row["previous_average_amount"]
        ),
    }

    response = requests.post(API_URL, json=payload, timeout=15)
    response.raise_for_status()
    api_result = response.json()

    api_prob = float(api_result["fraud_probability"])
    api_pred = 1 if api_result["prediction"].upper() == "FRAUD" else 0
    direct_p = float(row["_direct_prob"])
    direct_label = int(row["_direct_pred"])
    actual = int(row["_actual"])

    results.append({
        "row_index": idx,
        "step": int(row["step"]),
        "type": transaction_type,
        "actual_label": actual,
        "direct_model": direct_label,
        "api_prediction": api_pred,
        "direct_probability": direct_p,
        "api_probability": api_prob,
        "probability_difference": abs(direct_p - api_prob),
        "api_matches_model": direct_label == api_pred,
        "api_matches_actual": api_pred == actual,
    })

results_df = pd.DataFrame(results)

print("\n--- API vs Saved Model ---")
print(results_df.to_string(index=False))

print("\n--- Summary ---")
print(f"API/model prediction matches: "
      f"{results_df['api_matches_model'].sum()}/{len(results_df)}")
print(f"API/model probabilities within 0.00001: "
      f"{(results_df['probability_difference'] <= 0.00001).sum()}"
      f"/{len(results_df)}")
print(f"API predictions matching actual labels: "
      f"{results_df['api_matches_actual'].sum()}/{len(results_df)}")

if not results_df["api_matches_model"].all():
    print("\nWARNING: API predictions differ from direct model predictions.")
elif (results_df["probability_difference"] > 0.00001).any():
    print("\nWARNING: Some API probabilities differ from direct predictions.")
else:
    print("\nPASS: API predictions match the saved model on these samples.")

results_path = ROOT / "reports" / "api_consistency_results.csv"
results_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(results_path, index=False)
print(f"\nDetailed results saved to: {results_path}")
