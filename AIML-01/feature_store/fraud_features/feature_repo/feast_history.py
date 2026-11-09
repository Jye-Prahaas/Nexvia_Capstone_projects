
from pathlib import Path
import pandas as pd
from feast import FeatureStore

REPO_DIR = Path(__file__).resolve().parent
DATA_PATH = REPO_DIR / "data" / "fraud_features.parquet"


def main():
    print("Loading Feast feature store...")
    store = FeatureStore(repo_path=str(REPO_DIR))

    print("Loading sample transactions...")
    data = pd.read_parquet(
        DATA_PATH,
        columns=["transaction_id", "event_timestamp", "isFraud"]
    )

    # Select 10 transactions for a quick test
    sample = data.head(10).copy()

    print("Retrieving historical features...")
    result = store.get_historical_features(
        entity_df=sample[["transaction_id", "event_timestamp"]],
        features=[
            "fraud_transaction_features:amount",
            "fraud_transaction_features:oldbalanceOrg",
            "fraud_transaction_features:amount_log",
            "fraud_transaction_features:is_transfer",
            "fraud_transaction_features:is_cash_out",
            "fraud_transaction_features:previous_transaction_count",
            "fraud_transaction_features:previous_average_amount",
        ],
    ).to_df()

    # Add labels only after feature retrieval
    result = result.merge(
        sample[["transaction_id", "isFraud"]],
        on="transaction_id",
        how="left",
    )

    print("\nRetrieved features:")
    print(result.head())
    print("\nRows retrieved:", len(result))
    print("Missing values in retrieved features:")
    print(result.isnull().sum())


if __name__ == "__main__":
    main()
