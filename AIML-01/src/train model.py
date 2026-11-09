from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

EXPERIMENT_NAME = "Financial_Fraud_Classifier"
DECISION_THRESHOLD = 0.5


def setup_mlflow():
    """Configure MLflow to use a local SQLite tracking database."""
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")



def load_processed_data():
    """Load model features and labels from the Feast feature dataset."""
    feast_data_path = (
        PROJECT_ROOT
        / "feature_store"
        / "fraud_features"
        / "feature_repo"
        / "data"
        / "fraud_features.parquet"
    )

    if not feast_data_path.exists():
        raise FileNotFoundError(
            f"Feast dataset not found: {feast_data_path}"
        )

    print(f"Loading dataset: {feast_data_path}")
    data = pd.read_parquet(feast_data_path)

    required_columns = {
        "transaction_id",
        "event_timestamp",
        "step",
        "transaction_step",
        "isFraud",
        "previous_transaction_count",
        "previous_average_amount",
    }

    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(
            f"Feast dataset is missing required columns: {sorted(missing)}"
        )

    # The target is kept separate from the model inputs.
    y = data["isFraud"].astype("int8")

    # Exclude identifiers, timestamps, and the target from model inputs.
    X = data.drop(
        columns=["transaction_id", "event_timestamp", "isFraud"]
    )

    if len(X) != len(y):
        raise ValueError("Features and labels have different row counts.")

    if X.isnull().any().any():
        raise ValueError("Feature dataset contains missing values.")

    if not X.select_dtypes(include=["number"]).columns.equals(
        X.columns
    ):
        raise ValueError("All model features must be numeric.")

    print(f"Rows loaded: {len(X):,}")
    print(f"Fraud cases: {int(y.sum()):,}")
    print(f"Fraud rate: {y.mean():.4%}")
    print(f"Feature columns: {len(X.columns)}")
    print("Account-history features included:")
    print(" - previous_transaction_count")
    print(" - previous_average_amount")

    return X, y



def chronological_split(X, y):
    """Train on earlier transaction steps and test on later steps."""
    unique_steps = np.sort(X["step"].unique())

    if len(unique_steps) < 2:
        raise ValueError("At least two unique transaction steps are required.")

    split_position = int(len(unique_steps) * 0.8)
    split_position = min(max(split_position, 1), len(unique_steps) - 1)
    cutoff_step = unique_steps[split_position]

    train_mask = X["step"] < cutoff_step
    test_mask = X["step"] >= cutoff_step

    X_train = X.loc[train_mask].drop(columns=["transaction_step"])
    X_test = X.loc[test_mask].drop(columns=["transaction_step"])
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    print("\nCHRONOLOGICAL DATA SPLIT")
    print("-" * 40)
    print(f"Training cutoff step: {cutoff_step}")
    print(f"Training transactions: {len(X_train):,}")
    print(f"Testing transactions: {len(X_test):,}")
    print(f"Frauds in training data: {int(y_train.sum()):,}")
    print(f"Frauds in testing data: {int(y_test.sum()):,}")

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Training and testing data must each contain both classes. "
            "Check the chronological split."
        )

    return X_train, X_test, y_train, y_test, int(cutoff_step)


def train_model(X_train, y_train):
    """Train XGBoost with class-imbalance weighting."""
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError("Training data contains no fraud examples.")

    scale_pos_weight = negative_count / positive_count

    print("\nMODEL CONFIGURATION")
    print("-" * 40)
    print(f"Negative examples: {negative_count:,}")
    print(f"Positive examples: {positive_count:,}")
    print(f"Scale positive weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, dataset_name):
    """Evaluate one dataset and return metrics plus a text report."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype("int8")

    metrics = {
        f"{dataset_name}_precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        f"{dataset_name}_recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        f"{dataset_name}_f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        f"{dataset_name}_pr_auc": float(
            average_precision_score(y, probabilities)
        ),
    }

    print(f"\n{dataset_name.upper()} PERFORMANCE")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions, labels=[0, 1]))

    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report)

    return metrics, report


def save_model(model, feature_names):
    """Save the joblib artifact in the format expected by the API."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "xgboost_fraud_model.joblib"

    artifact = {
        "model": model,
        "feature_names": list(feature_names),
        "decision_threshold": DECISION_THRESHOLD,
    }
    joblib.dump(artifact, model_path)

    print(f"\nModel saved to: {model_path}")
    return model_path


def main():
    setup_mlflow()

    print("\nLoading processed transaction data...")
    X, y = load_processed_data()

    X_train, X_test, y_train, y_test, cutoff_step = chronological_split(X, y)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="xgboost_fraud_baseline"):
        model = train_model(X_train, y_train)

        params = {
            "model_type": "XGBClassifier",
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": float(
                model.get_params()["scale_pos_weight"]
            ),
            "random_state": 42,
            "tree_method": "hist",
            "decision_threshold": DECISION_THRESHOLD,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_fraud_count": int(y_train.sum()),
            "test_fraud_count": int(y_test.sum()),
            "chronological_cutoff_step": cutoff_step,
            "feature_count": X_train.shape[1],
        }
        mlflow.log_params(params)

        print("\n========== TRAINING PERFORMANCE ==========")
        train_metrics, train_report = evaluate_model(
            model, X_train, y_train, "train"
        )

        print("\n========== TEST PERFORMANCE ==========")
        test_metrics, test_report = evaluate_model(
            model, X_test, y_test, "test"
        )

        mlflow.log_metrics({**train_metrics, **test_metrics})

        train_report_path = REPORT_DIR / "train_classification_report.txt"
        test_report_path = REPORT_DIR / "test_classification_report.txt"
        train_report_path.write_text(train_report, encoding="utf-8")
        test_report_path.write_text(test_report, encoding="utf-8")

        mlflow.log_artifact(str(train_report_path))
        mlflow.log_artifact(str(test_report_path))

        model_path = save_model(model, X_train.columns)
        mlflow.log_artifact(str(model_path), artifact_path="model_files")

        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"MLflow experiment: {EXPERIMENT_NAME}")
        print(f"Tracking database: {MLFLOW_DB_PATH}")


if __name__ == "__main__":
    main()

