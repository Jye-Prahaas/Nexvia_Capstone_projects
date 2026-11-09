import json
from datetime import datetime
from pathlib import Path

import plotly.express as px

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
CASES_PATH = REPORTS_DIR / "analyst_cases.json"
DRIFT_PATH = REPORTS_DIR / "retraining_trigger.json"
KAFKA_DATA_PATH = ROOT / "data" / "monitoring" / "kafka_transactions.parquet"

API_URL = "http://127.0.0.1:8000/predict"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)


def load_cases():
    if not CASES_PATH.exists():
        return []

    try:
        return json.loads(CASES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_cases(cases):
    CASES_PATH.write_text(
        json.dumps(cases, indent=2),
        encoding="utf-8",
    )


def load_drift():
    if not DRIFT_PATH.exists():
        return None

    try:
        return json.loads(DRIFT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_kafka_data():
    if not KAFKA_DATA_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_parquet(KAFKA_DATA_PATH)
    except Exception as exc:
        st.error(f"Could not load Kafka monitoring data: {exc}")
        return pd.DataFrame()


st.title("🛡️ Financial Fraud Detection Platform")
st.caption("Real-time scoring • Fraud alerts • Case management • Drift monitoring")

cases = load_cases()
drift = load_drift()
kafka_data = load_kafka_data()


total_cases = len(cases)
fraud_cases = sum(
    str(case.get("prediction", "")).upper() == "FRAUD"
    for case in cases
)
open_cases = sum(
    case.get("status", "Open") != "Resolved"
    for case in cases
)
confirmed_fraud = sum(
    case.get("feedback") == "Confirmed Fraud"
    for case in cases
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Transactions Scored", total_cases)
col2.metric("Fraud Alerts", fraud_cases)
col3.metric("Unresolved Cases", open_cases)
col4.metric("Confirmed Fraud", confirmed_fraud)

st.divider()

st.subheader("Kafka Transaction Overview")

if not kafka_data.empty:
    total_transactions = len(kafka_data)

    fraud_count = 0
    legitimate_count = 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Kafka Transactions", total_transactions)
    col2.metric("Account History Available", 
                int((kafka_data["previous_transaction_count"] > 0).sum()))
    col3.metric("Account History Missing",
                int((kafka_data["previous_transaction_count"] == 0).sum()))

    st.write("Recent Kafka Transactions")

    st.dataframe(
        kafka_data.tail(20).iloc[::-1],
        use_container_width=True,
        hide_index=True
    )
    st.divider()

    st.subheader("Transaction Analytics")


    transaction_types = {
        "CASH_IN": "type_CASH_IN",
        "CASH_OUT": "type_CASH_OUT",
        "TRANSFER": "type_TRANSFER",
        "PAYMENT": "type_PAYMENT",
        "DEBIT": "type_DEBIT",
    }

    type_counts = {
        name: int(kafka_data[column].sum())
        for name, column in transaction_types.items()
    }

    type_df = pd.DataFrame(
        list(type_counts.items()),
        columns=["Transaction Type", "Count"]
    )

    col1, col2 = st.columns(2)

    with col1:
        
        fig = px.pie(
        type_df,
        names="Transaction Type",
        values="Count",
        title="Transaction Type Distribution",
        hole=0.5
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

    with col2:
        

        amount_bins = pd.cut(
            kafka_data["amount"],
            bins=10
        ).value_counts(sort=False)

        

    amount_bins = pd.cut(
        kafka_data["amount"],
        bins=10
    ).value_counts(sort=False)

    amount_df = pd.DataFrame({
        "Amount Range": amount_bins.index.astype(str),
        "Transactions": amount_bins.values
    })

    fig = px.bar(
        amount_df,
        x="Amount Range",
        y="Transactions",
        title="Transaction Amount Distribution",
        text="Transactions"
    )

    fig.update_layout(
        xaxis_title="Transaction Amount Range",
        yaxis_title="Number of Transactions",
        height=450,
        bargap=0.3
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.write("### Transaction Volume by Step")

    step_counts = (
        kafka_data.groupby("step")
        .size()
        .reset_index(name="Transactions")
        .sort_values("step")
    )

    st.line_chart(
        step_counts.set_index("step"),
        y="Transactions"
    )

else:
    st.warning("No Kafka monitoring data available.")


st.subheader("Data Drift Monitoring")



if drift:
    drift_col1, drift_col2, drift_col3 = st.columns(3)

    drift_col1.metric(
        "Drifted Feature Share",
        f"{drift.get('share_of_drifted_columns', 0):.1%}",
    )

    drift_col2.metric(
        "Dataset Drift",
        "Detected" if drift.get("dataset_drift_detected") else "Not detected",
    )

    drift_col3.metric(
        "Retraining Trigger",
        "Required" if drift.get("retraining_required") else "Not required",
    )

    st.caption(
        "This is a monitoring signal, not an automatic retraining process. "
        "Review the drift report before deciding to retrain."
    )

    report_path = REPORTS_DIR / "data_drift_report.html"

    if report_path.exists():
        st.info(f"Drift report available at: {report_path}")
else:
    st.info(
        "No drift trigger found. Run `python src\\drift_monitor.py` "
        "to generate the drift report and trigger."
    )

st.divider()


st.subheader("Score a Transaction")

with st.form("transaction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        step = st.number_input(
            "Transaction step",
            min_value=1,
            value=600,
        )
        transaction_type = st.selectbox(
            "Transaction type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
        )

    with col2:
        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=5000.0,
        )
        oldbalance_org = st.number_input(
            "Origin's previous balance",
            min_value=0.0,
            value=10000.0,
        )

    with col3:
        oldbalance_dest = st.number_input(
            "Destination's previous balance",
            min_value=0.0,
            value=2000.0,
        )
        previous_count = st.number_input(
            "Previous transaction count",
            min_value=0,
            value=0,
        )
        previous_average = st.number_input(
            "Previous average amount",
            min_value=0.0,
            value=0.0,
        )

    submitted = st.form_submit_button(
        "Analyze Transaction",
        type="primary",
    )

if submitted:
    payload = {
        "step": int(step),
        "transaction_type": transaction_type,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalance_org),
        "oldbalanceDest": float(oldbalance_dest),
        "previous_transaction_count": int(previous_count),
        "previous_average_amount": float(previous_average),
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        prediction = str(result.get("prediction", "UNKNOWN")).upper()
        probability = float(result.get("fraud_probability", 0.0))

        new_case = {
            "case_id": f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "transaction_type": transaction_type,
            "amount": float(amount),
            "prediction": prediction,
            "fraud_probability": probability,
            "status": "Open" if prediction == "FRAUD" else "Resolved",
            "feedback": "Pending",
            "payload": payload,
        }

        cases = load_cases()
        cases.insert(0, new_case)
        save_cases(cases)

        if prediction == "FRAUD":
            st.error(
                f"🚨 FRAUD ALERT — Estimated fraud probability: "
                f"{probability:.2%}"
            )
        elif prediction == "LEGITIMATE":
            st.success(
                f"Transaction classified as LEGITIMATE. "
                f"Estimated fraud probability: {probability:.2%}"
            )
        else:
            st.warning(f"Model returned: {prediction}")

        st.json(result)

    except requests.RequestException as exc:
        st.error(
            "Could not reach the prediction API. Start FastAPI first "
            "and check that it is running at http://127.0.0.1:8000."
        )
        st.caption(str(exc))

    except (ValueError, TypeError) as exc:
        st.error(f"Could not process the prediction response: {exc}")

st.divider()

st.divider()

st.subheader("Fraud Analytics")

cases = load_cases()

if cases:
    fraud_count = sum(
        case.get("prediction") == "FRAUD"
        for case in cases
    )

    legitimate_count = sum(
        case.get("prediction") == "LEGITIMATE"
        for case in cases
    )

    confirmed_count = sum(
        case.get("feedback") == "Confirmed Fraud"
        for case in cases
    )

    false_positive_count = sum(
        case.get("feedback") == "False Positive"
        for case in cases
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Fraud Predictions", fraud_count)
    col2.metric("Legitimate Predictions", legitimate_count)
    col3.metric("Confirmed Fraud", confirmed_count)
    col4.metric("False Positives", false_positive_count)

    st.write("### Prediction Distribution")

    prediction_df = pd.DataFrame({
        "Prediction": ["FRAUD", "LEGITIMATE"],
        "Count": [fraud_count, legitimate_count]
    })

    fig = px.bar(
    prediction_df,
    x="Prediction",
    y="Count",
    title="Prediction Distribution",
    text="Count",
    color="Prediction",
    color_discrete_map={
        "FRAUD": "#E74C3C",
        "LEGITIMATE": "#2ECC71"
    }
)


    fig.update_layout(
        xaxis_title="Prediction",
        yaxis_title="Number of Transactions",
        bargap=0.5
    )

    fig.update_xaxes(type="category")

    st.plotly_chart(fig, use_container_width=True)

    st.write("### Fraud Probability by Case")

    probability_df = pd.DataFrame([
        {
            "Case ID": case["case_id"][-8:],
            "Fraud Probability": case["fraud_probability"]
        }
        for case in cases
    ])

    probability_df["Case ID"] = probability_df["Case ID"].astype(str)

    probability_df = probability_df.sort_values(
    "Fraud Probability",
    ascending=True
)

    fig = px.bar(
        probability_df,
        x="Fraud Probability",
        y="Case ID",
        orientation="h",
        title="Fraud Probability by Case",
        text="Fraud Probability",
        color="Fraud Probability",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 1]
    )

    fig.update_layout(
        xaxis_title="Fraud Probability",
        yaxis_title="Case ID",
        xaxis=dict(range=[0, 1]),
        height=350,
        showlegend=False
    )

    fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("### Analyst Feedback Distribution")

    feedback_df = pd.DataFrame({
        "Feedback": [
            "Confirmed Fraud",
            "False Positive",
            "Pending"
        ],
        "Count": [
            confirmed_count,
            false_positive_count,
            sum(
                case.get("feedback", "Pending") == "Pending"
                for case in cases
            )
        ]
    })

    fig = px.bar(
        feedback_df,
        x="Feedback",
        y="Count",
        title="Analyst Feedback Distribution",
        text="Count"
    )

    fig.update_layout(
        xaxis_title="Analyst Feedback",
        yaxis_title="Number of Cases",
        bargap=0.5
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No transaction predictions available yet.")

st.subheader("Fraud Alerts & Case Management")

cases = load_cases()

fraud_alerts = [
    case for case in cases
    if case.get("prediction") == "FRAUD"
]

if not fraud_alerts:
    st.info("No fraud alerts recorded yet. Score a transaction above to begin.")
else:
    display_rows = [
        {
            "Case ID": case["case_id"],
            "Created": case["created_at"],
            "Type": case["transaction_type"],
            "Amount": case["amount"],
            "Fraud Probability": case["fraud_probability"],
            "Status": case.get("status", "Open"),
            "Feedback": case.get("feedback", "Pending"),
        }
        for case in fraud_alerts
    ]

    st.dataframe(
        pd.DataFrame(display_rows),
        use_container_width=True,
        hide_index=True,
    )

    selected_id = st.selectbox(
        "Select a case to update",
        [case["case_id"] for case in fraud_alerts],
    )

    selected_case = next(
        case for case in fraud_alerts
        if case["case_id"] == selected_id
    )

    with st.form("case_update_form"):
        new_status = st.selectbox(
            "Case status",
            ["Open", "Investigating", "Resolved"],
            index=["Open", "Investigating", "Resolved"].index(
                selected_case.get("status", "Open")
            ),
        )

        new_feedback = st.selectbox(
            "Analyst feedback",
            [
                "Pending",
                "Confirmed Fraud",
                "False Positive",
            ],
            index=[
                "Pending",
                "Confirmed Fraud",
                "False Positive",
            ].index(selected_case.get("feedback", "Pending")),
        )

        update_submitted = st.form_submit_button("Save Case Update")

    if update_submitted:
        cases = load_cases()

        for case in cases:
            if case["case_id"] == selected_id:
                case["status"] = new_status
                case["feedback"] = new_feedback
                case["updated_at"] = datetime.now().isoformat(
                    timespec="seconds"
                )
                break

        save_cases(cases)
        st.success("Case updated successfully.")
        st.rerun()

st.caption(
    "Case records are stored locally in reports/analyst_cases.json. "
    "Analyst feedback is recorded for review; it does not automatically "
    "retrain the model."
)

if st.button("Refresh Dashboard"):
    st.rerun()