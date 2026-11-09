
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_fraud_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_names = artifact["feature_names"]

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        by="importance",
        ascending=False,
    )

    print("\nFEATURE IMPORTANCE")
    print("-" * 50)
    print(importance.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORTS_DIR / "feature_importance.csv"
    plot_path = REPORTS_DIR / "feature_importance.png"

    importance.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 7))
    plt.barh(
        importance["feature"].iloc[::-1],
        importance["importance"].iloc[::-1],
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("XGBoost Fraud Detection - Feature Importance")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nImportance table saved to: {csv_path}")
    print(f"Feature importance chart saved to: {plot_path}")


if __name__ == "__main__":
    main()
