
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
from kafka import KafkaConsumer, TopicPartition
from kafka.structs import OffsetAndMetadata


PROJECT_ROOT = Path(__file__).resolve().parent.parent

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "fraud-transactions"
KAFKA_GROUP_ID = "fraud-prediction-consumer-v3"

API_URL = "http://127.0.0.1:8000/predict"

MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"
MONITORING_PATH = MONITORING_DIR / "kafka_transactions.parquet"
LEGACY_MONITORING_PATH = (
    MONITORING_DIR / "kafka_transactions_legacy.parquet"
)

DATABASE_PATH = MONITORING_DIR / "kafka_consumer_state.db"

FEATURE_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_log",
    "amount_to_origin_balance",
    "amount_exceeds_origin_balance",
    "origin_balance_is_zero",
    "is_transfer",
    "is_cash_out",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "previous_transaction_count",
    "previous_average_amount",
]

POLL_TIMEOUT_MS = 1000
API_TIMEOUT_SECONDS = 15

MONITORING_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("fraud-kafka-consumer")

db = sqlite3.connect(
    DATABASE_PATH,
    timeout=30,
)

db.execute("PRAGMA journal_mode=WAL")
db.execute("PRAGMA synchronous=FULL")

db.execute("""
CREATE TABLE IF NOT EXISTS account_history (
    replay_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    last_step INTEGER,
    previous_count INTEGER NOT NULL DEFAULT 0,
    previous_sum REAL NOT NULL DEFAULT 0,
    current_step INTEGER,
    current_step_count INTEGER NOT NULL DEFAULT 0,
    current_step_sum REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (replay_id, account_id)
)
""")

db.execute("""
CREATE TABLE IF NOT EXISTS processed_offsets (
    partition_id INTEGER PRIMARY KEY,
    last_offset INTEGER NOT NULL
)
""")

db.execute("""
CREATE TABLE IF NOT EXISTS monitoring_records (
    partition_id INTEGER NOT NULL,
    message_offset INTEGER NOT NULL,
    feature_json TEXT NOT NULL,
    PRIMARY KEY (partition_id, message_offset)
)
""")

db.execute("""
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
)
""")

db.commit()


account_history = {}


def get_replay_id(transaction_id):
    """
    Extract replay ID from transaction IDs formatted as:
    PAY-{replay_id}-{row_number}
    """

    if not transaction_id:
        return "default"

    match = re.match(
        r"^PAY-(.+)-\d{9}$",
        str(transaction_id),
    )

    if match:
        return match.group(1)

    return "default"


def get_history_key(replay_id, account_id):
    return str(replay_id), str(account_id)


def get_history_state(replay_id, account_id):
    key = get_history_key(replay_id, account_id)

    if key not in account_history:
        account_history[key] = {
            "last_step": None,
            "previous_count": 0,
            "previous_sum": 0.0,
            "current_step": None,
            "current_step_count": 0,
            "current_step_sum": 0.0,
        }

    return account_history[key]


def calculate_previous_history(
    replay_id,
    account_id,
    step,
    amount,
):
    """
    Calculate history using only completed earlier steps.

    Transactions from the current step are not included
    in previous_transaction_count or previous_average_amount.
    """

    state = get_history_state(replay_id, account_id)

    step = int(step)
    amount = float(amount)

    if state["last_step"] is not None:
        if step < state["last_step"]:
            return None

    if state["current_step"] is None:
        state["current_step"] = step

    elif step > state["current_step"]:

        state["previous_count"] += state["current_step_count"]

        state["previous_sum"] += state["current_step_sum"]

        state["current_step"] = step

        state["current_step_count"] = 0

        state["current_step_sum"] = 0.0

    previous_count = state["previous_count"]

    if previous_count > 0:
        previous_average = (
            state["previous_sum"] / previous_count
        )
    else:
        previous_average = 0.0

    return {
        "previous_transaction_count": previous_count,
        "previous_average_amount": previous_average,
    }


def update_history_after_prediction(
    replay_id,
    account_id,
    step,
    amount,
):
    state = get_history_state(replay_id, account_id)

    state["current_step_count"] += 1

    state["current_step_sum"] += float(amount)

    state["last_step"] = int(step)

def load_history_from_database():

    global account_history

    rows = db.execute("""
        SELECT
            replay_id,
            account_id,
            last_step,
            previous_count,
            previous_sum,
            current_step,
            current_step_count,
            current_step_sum
        FROM account_history
    """).fetchall()

    account_history = {}

    for row in rows:

        replay_id, account_id = row[0], row[1]

        account_history[(replay_id, account_id)] = {
            "last_step": row[2],
            "previous_count": row[3],
            "previous_sum": row[4],
            "current_step": row[5],
            "current_step_count": row[6],
            "current_step_sum": row[7],
        }

    logger.info(
        "Loaded %s account-history records from SQLite.",
        len(account_history),
    )


def save_history_to_database():

    with db:

        db.execute("DELETE FROM account_history")

        for (replay_id, account_id), state in account_history.items():

            db.execute("""
                INSERT INTO account_history (
                    replay_id,
                    account_id,
                    last_step,
                    previous_count,
                    previous_sum,
                    current_step,
                    current_step_count,
                    current_step_sum
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                replay_id,
                account_id,
                state["last_step"],
                state["previous_count"],
                state["previous_sum"],
                state["current_step"],
                state["current_step_count"],
                state["current_step_sum"],
            ))

def rebuild_history(main_consumer):

    marker = db.execute("""
        SELECT value
        FROM metadata
        WHERE key = 'history_rebuilt'
    """).fetchone()

    if marker is not None:
        logger.info(
            "History recovery already completed. "
            "Loading saved SQLite history."
        )
        load_history_from_database()
        return

    logger.info(
        "Rebuilding account history from Kafka committed offsets..."
    )

    partitions = main_consumer.assignment()

    if not partitions:
        raise RuntimeError(
            "Kafka consumer has no assigned partitions."
        )

    target_offsets = {}

    beginning_offsets = main_consumer.beginning_offsets(partitions)

    for tp in partitions:

        committed = main_consumer.committed(tp)

        if committed is None:
            target_offsets[tp] = beginning_offsets[tp]
        else:
            target_offsets[tp] = committed

    recovery_consumer = KafkaConsumer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        consumer_timeout_ms=1000,
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
    )

    recovery_consumer.assign(list(partitions))

    for tp in partitions:
        recovery_consumer.seek(
            tp,
            beginning_offsets[tp],
        )

    account_history.clear()

    completed = set()

    try:

        while len(completed) < len(partitions):

            records = recovery_consumer.poll(
                timeout_ms=POLL_TIMEOUT_MS,
            )

            for tp, messages in records.items():

                target = target_offsets[tp]

                for message in messages:

                    if message.offset >= target:
                        continue

                    data = message.value

                    account_id = (
                        data.get("nameOrig")
                        or data.get("account_id")
                    )

                    if not account_id:
                        continue

                    transaction_id = data.get("transaction_id")

                    replay_id = get_replay_id(transaction_id)

                    step = data.get("step")
                    amount = data.get("amount")

                    if step is None or amount is None:
                        continue

                    state = get_history_state(
                        replay_id,
                        account_id,
                    )

                    step = int(step)
                    amount = float(amount)

                    if (
                        state["last_step"] is not None
                        and step < state["last_step"]
                    ):
                        continue

                    if state["current_step"] is None:
                        state["current_step"] = step

                    elif step > state["current_step"]:

                        state["previous_count"] += (
                            state["current_step_count"]
                        )

                        state["previous_sum"] += (
                            state["current_step_sum"]
                        )

                        state["current_step"] = step

                        state["current_step_count"] = 0

                        state["current_step_sum"] = 0.0

                    state["current_step_count"] += 1

                    state["current_step_sum"] += amount

                    state["last_step"] = step

            for tp in partitions:

                if tp in completed:
                    continue

                if recovery_consumer.position(tp) >= target_offsets[tp]:
                    completed.add(tp)

        with db:

            db.execute("DELETE FROM account_history")

            db.execute("DELETE FROM processed_offsets")

            for (replay_id, account_id), state in account_history.items():

                db.execute("""
                    INSERT INTO account_history (
                        replay_id,
                        account_id,
                        last_step,
                        previous_count,
                        previous_sum,
                        current_step,
                        current_step_count,
                        current_step_sum
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    replay_id,
                    account_id,
                    state["last_step"],
                    state["previous_count"],
                    state["previous_sum"],
                    state["current_step"],
                    state["current_step_count"],
                    state["current_step_sum"],
                ))

            for tp, target in target_offsets.items():

                if target > beginning_offsets[tp]:

                    db.execute("""
                        INSERT OR REPLACE INTO processed_offsets (
                            partition_id,
                            last_offset
                        )
                        VALUES (?, ?)
                    """, (
                        tp.partition,
                        target - 1,
                    ))

            db.execute("""
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('history_rebuilt', '1')
            """)

        logger.info(
            "History recovery completed. Recovered %s account records.",
            len(account_history),
        )

    finally:
        recovery_consumer.close()


def initialize_monitoring_base():

    marker = db.execute("""
        SELECT value
        FROM metadata
        WHERE key = 'legacy_archived'
    """).fetchone()

    if marker is None:

        if MONITORING_PATH.exists():

            if not LEGACY_MONITORING_PATH.exists():

                import shutil

                shutil.copy2(
                    MONITORING_PATH,
                    LEGACY_MONITORING_PATH,
                )

                logger.info(
                    "Existing monitoring dataset archived: %s",
                    LEGACY_MONITORING_PATH,
                )

        with db:

            db.execute("""
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES ('legacy_archived', '1')
            """)

    if not MONITORING_PATH.exists():

        pd.DataFrame(
            columns=FEATURE_COLUMNS
        ).to_parquet(
            MONITORING_PATH,
            index=False,
        )

    logger.info(
        "Active monitoring dataset initialized."
    )


def build_monitoring_features(
    data,
    history_features,
):

    transaction_type = str(
        data.get("transaction_type", "")
    ).upper()

    amount = float(data["amount"])

    oldbalance_org = float(data["oldbalanceOrg"])

    oldbalance_dest = float(data["oldbalanceDest"])

    features = {
        "step": int(data["step"]),
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "oldbalanceDest": oldbalance_dest,

        "amount_log": float(
            __import__("math").log1p(max(amount, 0))
        ),

        "amount_to_origin_balance": (
            amount / oldbalance_org
            if oldbalance_org > 0
            else 0.0
        ),

        "amount_exceeds_origin_balance": int(
            amount > oldbalance_org
        ),

        "origin_balance_is_zero": int(
            oldbalance_org == 0
        ),

        "is_transfer": int(
            transaction_type == "TRANSFER"
        ),

        "is_cash_out": int(
            transaction_type == "CASH_OUT"
        ),

        "type_CASH_IN": int(
            transaction_type == "CASH_IN"
        ),

        "type_CASH_OUT": int(
            transaction_type == "CASH_OUT"
        ),

        "type_DEBIT": int(
            transaction_type == "DEBIT"
        ),

        "type_PAYMENT": int(
            transaction_type == "PAYMENT"
        ),

        "type_TRANSFER": int(
            transaction_type == "TRANSFER"
        ),

        "previous_transaction_count": (
            history_features["previous_transaction_count"]
        ),

        "previous_average_amount": (
            history_features["previous_average_amount"]
        ),
    }

    return features


def predict_transaction(data, history_features):

    payload = {
        "step": int(data["step"]),
        "transaction_type": str(data["transaction_type"]),
        "amount": float(data["amount"]),
        "oldbalanceOrg": float(data["oldbalanceOrg"]),
        "oldbalanceDest": float(data["oldbalanceDest"]),

        "previous_transaction_count": int(
            history_features["previous_transaction_count"]
        ),

        "previous_average_amount": float(
            history_features["previous_average_amount"]
        ),
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=API_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    return response.json()


def persist_successful_transaction(
    tp,
    offset,
    replay_id,
    account_id,
    step,
    amount,
    monitoring_features,
):

    state = get_history_state(
        replay_id,
        account_id,
    )

    state["current_step_count"] += 1

    state["current_step_sum"] += float(amount)

    state["last_step"] = int(step)

    feature_json = json.dumps(
        monitoring_features
    )

    with db:

        db.execute("""
            INSERT OR REPLACE INTO account_history (
                replay_id,
                account_id,
                last_step,
                previous_count,
                previous_sum,
                current_step,
                current_step_count,
                current_step_sum
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            replay_id,
            account_id,
            state["last_step"],
            state["previous_count"],
            state["previous_sum"],
            state["current_step"],
            state["current_step_count"],
            state["current_step_sum"],
        ))

        db.execute("""
            INSERT OR REPLACE INTO monitoring_records (
                partition_id,
                message_offset,
                feature_json
            )
            VALUES (?, ?, ?)
        """, (
            tp.partition,
            offset,
            feature_json,
        ))

        db.execute("""
            INSERT OR REPLACE INTO processed_offsets (
                partition_id,
                last_offset
            )
            VALUES (?, ?)
        """, (
            tp.partition,
            offset,
        ))


def persist_skipped_offset(tp, offset):

    with db:

        db.execute("""
            INSERT OR REPLACE INTO processed_offsets (
                partition_id,
                last_offset
            )
            VALUES (?, ?)
        """, (
            tp.partition,
            offset,
        ))


def export_monitoring():

    rows = db.execute("""
        SELECT feature_json
        FROM monitoring_records
        ORDER BY partition_id, message_offset
    """).fetchall()

    if rows:

        records = [
            json.loads(row[0])
            for row in rows
        ]

        new_data = pd.DataFrame(
            records,
            columns=FEATURE_COLUMNS,
        )

    else:

        new_data = pd.DataFrame(
            columns=FEATURE_COLUMNS
        )

    temporary_path = MONITORING_DIR / (
        f"kafka_export_{os.getpid()}_{time.time_ns()}.tmp.parquet"
    )

    try:

        new_data.to_parquet(
            temporary_path,
            index=False,
        )

        os.replace(
            temporary_path,
            MONITORING_PATH,
        )

        logger.info(
            "Active monitoring dataset exported: %s verified records.",
            len(new_data),
        )

    finally:

        if temporary_path.exists():
            temporary_path.unlink()



def main():

    logger.info("Starting Kafka fraud prediction consumer.")

    consumer = KafkaConsumer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda value: json.loads(
            value.decode("utf-8")
        ),
    )

    consumer.subscribe([KAFKA_TOPIC])

    try:

        
        while not consumer.assignment():
            consumer.poll(timeout_ms=POLL_TIMEOUT_MS)

        logger.info(
            "Assigned Kafka partitions: %s",
            consumer.assignment(),
        )

        initialize_monitoring_base()

        rebuild_history(consumer)

    
        export_monitoring()

        logger.info(
            "Listening to Kafka topic: %s",
            KAFKA_TOPIC,
        )

        while True:

            records = consumer.poll(
                timeout_ms=POLL_TIMEOUT_MS,
            )

            for tp, messages in records.items():

                for message in messages:

                    # Skip messages already stored in SQLite.
                    stored_offset = db.execute("""
                        SELECT last_offset
                        FROM processed_offsets
                        WHERE partition_id = ?
                    """, (
                        tp.partition,
                    )).fetchone()

                    if (
                        stored_offset is not None
                        and message.offset <= stored_offset[0]
                    ):

                        consumer.commit({
                            tp: OffsetAndMetadata(
                                message.offset + 1,
                                "",
                            )
                        })

                        continue

                    data = message.value

                    account_id = (
                        data.get("nameOrig")
                        or data.get("account_id")
                    )

                    transaction_id = data.get("transaction_id")

                    if not account_id:

                        logger.warning(
                            "Skipping message without account ID. "
                            "Partition=%s Offset=%s",
                            tp.partition,
                            message.offset,
                        )

                        persist_skipped_offset(
                            tp,
                            message.offset,
                        )

                        consumer.commit({
                            tp: OffsetAndMetadata(
                                message.offset + 1,
                                "",
                            )
                        })

                        continue

                    required_fields = [
                        "step",
                        "transaction_type",
                        "amount",
                        "oldbalanceOrg",
                        "oldbalanceDest",
                    ]

                    missing_fields = [
                        field
                        for field in required_fields
                        if field not in data
                    ]

                    if missing_fields:

                        logger.warning(
                            "Skipping invalid message %s. Missing: %s",
                            transaction_id,
                            missing_fields,
                        )

                        persist_skipped_offset(
                            tp,
                            message.offset,
                        )

                        consumer.commit({
                            tp: OffsetAndMetadata(
                                message.offset + 1,
                                "",
                            )
                        })

                        continue

                    replay_id = get_replay_id(
                        transaction_id
                    )

                    step = int(data["step"])

                    amount = float(data["amount"])

                    
                    history_features = calculate_previous_history(
                        replay_id,
                        account_id,
                        step,
                        amount,
                    )

                    if history_features is None:

                        logger.warning(
                            "Skipping out-of-order transaction %s.",
                            transaction_id,
                        )

                        persist_skipped_offset(
                            tp,
                            message.offset,
                        )

                        consumer.commit({
                            tp: OffsetAndMetadata(
                                message.offset + 1,
                                "",
                            )
                        })

                        continue

                    try:

                        prediction = predict_transaction(
                            data,
                            history_features,
                        )

                        monitoring_features = (
                            build_monitoring_features(
                                data,
                                history_features,
                            )
                        )

                        persist_successful_transaction(
                            tp,
                            message.offset,
                            replay_id,
                            account_id,
                            step,
                            amount,
                            monitoring_features,
                        )

                        
                        consumer.commit({
                            tp: OffsetAndMetadata(
                                message.offset + 1,
                                "",
                            )
                        })

                        logger.info(
                            "Transaction=%s | Prediction=%s | "
                            "Fraud Probability=%s | Previous Count=%s",
                            transaction_id,
                            prediction.get("prediction"),
                            prediction.get("fraud_probability"),
                            history_features[
                                "previous_transaction_count"
                            ],
                        )

                    except requests.RequestException:

                        logger.exception(
                            "FastAPI prediction failed for transaction %s. "
                            "Stopping so this message can be retried.",
                            transaction_id,
                        )

                        raise

                    except Exception:

                        logger.exception(
                            "Processing failed for transaction %s. "
                            "Stopping to avoid committing an unprocessed message.",
                            transaction_id,
                        )

                        raise

    except KeyboardInterrupt:

        logger.info(
            "Consumer stopped by user."
        )

    finally:

        try:
            export_monitoring()

        except Exception:

            logger.exception(
                "Monitoring export failed. SQLite records remain saved."
            )

        consumer.close()

        db.close()

        logger.info(
            "Kafka consumer and SQLite connection closed."
        )


if __name__ == "__main__":
    main()
