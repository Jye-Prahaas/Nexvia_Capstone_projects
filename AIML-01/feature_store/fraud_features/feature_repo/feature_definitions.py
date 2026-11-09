from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64


REPO_DIR = Path(__file__).resolve().parent
DATA_PATH = REPO_DIR / "data" / "fraud_features.parquet"

transaction = Entity(
    name="transaction",
    join_keys=["transaction_id"],
    description="Dataset-local transaction row for offline feature retrieval",
)

transaction_source = FileSource(
    name="fraud_transaction_source",
    path=str(DATA_PATH),
    timestamp_field="event_timestamp",
)

transaction_features = FeatureView(
    name="fraud_transaction_features",
    entities=[transaction],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="step", dtype=Int64),
        Field(name="amount", dtype=Float64),
        Field(name="oldbalanceOrg", dtype=Float64),
        Field(name="oldbalanceDest", dtype=Float64),
        Field(name="amount_log", dtype=Float64),
        Field(name="amount_to_origin_balance", dtype=Float64),
        Field(name="amount_exceeds_origin_balance", dtype=Int64),
        Field(name="origin_balance_is_zero", dtype=Int64),
        Field(name="is_transfer", dtype=Int64),
        Field(name="is_cash_out", dtype=Int64),
        Field(name="type_CASH_IN", dtype=Int64),
        Field(name="type_CASH_OUT", dtype=Int64),
        Field(name="type_DEBIT", dtype=Int64),
        Field(name="type_PAYMENT", dtype=Int64),
        Field(name="type_TRANSFER", dtype=Int64),
        Field(name="previous_transaction_count", dtype=Int64),
        Field(name="previous_average_amount", dtype=Float64),
    ],
    source=transaction_source,
    online=True,
    tags={"project": "financial_fraud_classifier"},
)