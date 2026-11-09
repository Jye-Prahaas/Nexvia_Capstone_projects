
from fastapi.testclient import TestClient

from src.Api import app

client = TestClient(app)

VALID_TRANSACTION = {
    "step": 600,
    "transaction_type": "TRANSFER",
    "amount": 5000,
    "oldbalanceOrg": 10000,
    "oldbalanceDest": 2000,
    "previous_transaction_count": 3,
    "previous_average_amount": 2500,
}


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_prediction():
    response = client.post("/predict", json=VALID_TRANSACTION)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in ["FRAUD", "LEGITIMATE"]
    assert 0 <= result["fraud_probability"] <= 1
    assert result["threshold"] == 0.5


def test_invalid_transaction_type():
    payload = VALID_TRANSACTION.copy()
    payload["transaction_type"] = "INVALID"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_missing_history_features():
    payload = {
        key: value
        for key, value in VALID_TRANSACTION.items()
        if key not in {
            "previous_transaction_count",
            "previous_average_amount",
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_negative_amount():
    payload = VALID_TRANSACTION.copy()
    payload["amount"] = -100

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
