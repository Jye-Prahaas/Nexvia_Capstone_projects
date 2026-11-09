
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_data():
    X = pd.read_parquet(DATA_DIR / "features.parquet")
    y = pd.read_parquet(DATA_DIR / "labels.parquet")["isFraud"]
    return X, y


def chronological_split(X, y):
    steps = np.sort(X["step"].unique())
    cutoff = steps[int(len(steps) * 0.8)]

    train_mask = X["step"] < cutoff
    test_mask = X["step"] >= cutoff

    # Remove duplicate time feature; "step" is retained.
    X = X.drop(columns=["transaction_step"], errors="ignore")

    return (
        X.loc[train_mask].copy(),
        X.loc[test_mask].copy(),
        y.loc[train_mask],
        y.loc[test_mask],
    )


def evaluate_variant(name, X_train, X_test, y_train, y_test):
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negative_count / positive_count,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")

    result = {
        "model": name,
        "features": X_train.shape[1],
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(y_test, probabilities),
    }

    print(
        f"Precision: {result['precision']:.4f} | "
        f"Recall: {result['recall']:.4f} | "
        f"F1: {result['f1_score']:.4f} | "
        f"PR-AUC: {result['pr_auc']:.4f}"
    )

    return result


def main():
    print("Loading processed data...")
    X, y = load_data()

    X_train, X_test, y_train, y_test = chronological_split(X, y)

    # A: All features.
    all_features = list(X_train.columns)

    # B: Remove the strongest engineered feature.
    without_balance_flag = [
        col for col in all_features
        if col != "amount_exceeds_origin_balance"
    ]

    # C: Remove transaction type indicators and related flags.
    type_columns = [
        col for col in all_features
        if col.startswith("type_")
        or col in {"is_transfer", "is_cash_out"}
    ]
    without_type = [
        col for col in all_features
        if col not in type_columns
    ]

    variants = [
        ("A - All features", all_features),
        ("B - Without balance flag", without_balance_flag),
        ("C - Without transaction-type features", without_type),
    ]

    results = []

    for name, columns in variants:
        result = evaluate_variant(
            name,
            X_train[columns],
            X_test[columns],
            y_train,
            y_test,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORTS_DIR / "feature_ablation_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nFEATURE ABLATION RESULTS")
    print("-" * 80)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
