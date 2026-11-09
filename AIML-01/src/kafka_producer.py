import json
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from kafka import KafkaProducer


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Transactions.csv"

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "fraud-transactions"

BATCH_SIZE = 1000

REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "oldbalanceDest",
}


def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=5,
    )


def replay_transactions(limit=None, delay=0.0):
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"PaySim dataset not found: {DATA_PATH}"
        )

    producer = create_producer()

    replay_id = datetime.now().strftime("%Y%m%d%H%M%S")
    published = 0
    pending = []

    print("Starting PaySim transaction replay...")
    print(f"Dataset: {DATA_PATH}")
    print(f"Kafka topic: {TOPIC}")
    print(f"Replay ID: {replay_id}")
    print(f"Message limit: {limit if limit else 'Full dataset'}")

    try:
        with DATA_PATH.open(
            "r",
            encoding="utf-8",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])

            if missing:
                raise ValueError(
                    f"Dataset is missing columns: {sorted(missing)}"
                )

            for row_number, row in enumerate(reader, start=1):

                if limit is not None and published >= limit:
                    break

                account_id = row["nameOrig"]

                if not account_id:
                    print(
                        f"Skipping row {row_number}: missing account ID"
                    )
                    continue

                transaction = {
                    "transaction_id": (
                        f"PAY-{replay_id}-{row_number:09d}"
                    ),
                    "nameOrig": account_id,
                    "step": int(float(row["step"])),
                    "transaction_type": row["type"].strip().upper(),
                    "amount": float(row["amount"]),
                    "oldbalanceOrg": float(row["oldbalanceOrg"]),
                    "oldbalanceDest": float(row["oldbalanceDest"]),
                }

                future = producer.send(
                    TOPIC,
                    key=account_id,
                    value=transaction,
                )

                pending.append(future)
                published += 1

                if published <= 5:
                    print(
                        f"Queued {transaction['transaction_id']} "
                        f"| Type: {transaction['transaction_type']} "
                        f"| Amount: {transaction['amount']}"
                    )

                if len(pending) >= BATCH_SIZE:
                    for result in pending:
                        result.get(timeout=30)

                    pending.clear()

                    print(
                        f"Published transactions: {published:,}"
                    )

                if delay > 0:
                    time.sleep(delay)

        # Confirm all remaining messages reached Kafka.
        for result in pending:
            result.get(timeout=30)

        producer.flush()

        print("\nTransaction replay completed.")
        print(f"Total transactions published: {published:,}")

    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
        producer.flush()

    finally:
        producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay PaySim transactions into Kafka."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum transactions to publish. Omit for full replay.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between transactions.",
    )

    args = parser.parse_args()

    replay_transactions(
        limit=args.limit,
        delay=args.delay,
    )
