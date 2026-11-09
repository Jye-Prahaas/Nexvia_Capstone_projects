
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "Transactions.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "graph_sample.parquet"

CHUNK_SIZE = 250_000
FRAUD_SAMPLE_SIZE = 8_213
LEGITIMATE_SAMPLE_SIZE = 100_000
RANDOM_SEED = 42

required_columns = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "oldbalanceDest",
    "nameDest",
    "isFraud",
]

print("Scanning transactions in chunks...")

fraud_chunks = []
legitimate_reservoir = pd.DataFrame()
legitimate_seen = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        RAW_PATH,
        usecols=required_columns,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):
    fraud = chunk[chunk["isFraud"] == 1]
    legitimate = chunk[chunk["isFraud"] == 0]

    if not fraud.empty:
        fraud_chunks.append(fraud)

    # Reservoir-style sampling keeps the legitimate sample
    # approximately uniform across the entire input stream.
    combined = pd.concat(
        [legitimate_reservoir, legitimate],
        ignore_index=True,
    )

    legitimate_seen += len(legitimate)

    if len(combined) > LEGITIMATE_SAMPLE_SIZE:
        legitimate_reservoir = combined.sample(
            n=LEGITIMATE_SAMPLE_SIZE,
            random_state=RANDOM_SEED + chunk_number,
        ).reset_index(drop=True)
    else:
        legitimate_reservoir = combined

    if chunk_number % 5 == 0:
        print(
            f"Chunks scanned: {chunk_number} | "
            f"Fraud rows collected: "
            f"{sum(len(frame) for frame in fraud_chunks):,} | "
            f"Legitimate rows seen: {legitimate_seen:,}"
        )

fraud_df = pd.concat(fraud_chunks, ignore_index=True)

if len(fraud_df) > FRAUD_SAMPLE_SIZE:
    fraud_df = fraud_df.sample(
        n=FRAUD_SAMPLE_SIZE,
        random_state=RANDOM_SEED,
    )

sample_df = pd.concat(
    [fraud_df, legitimate_reservoir],
    ignore_index=True,
)

sample_df = (
    sample_df.drop_duplicates()
    .sort_values(["step"])
    .reset_index(drop=True)
)

sample_df.to_parquet(OUTPUT_PATH, index=False)

print("\nGraph sample prepared.")
print(f"Rows: {len(sample_df):,}")
print(f"Fraud cases: {int(sample_df['isFraud'].sum()):,}")
print(f"Legitimate cases: {(sample_df['isFraud'] == 0).sum():,}")
print(f"Steps covered: {sample_df['step'].min()} to {sample_df['step'].max()}")
print(f"Saved to: {OUTPUT_PATH}")
