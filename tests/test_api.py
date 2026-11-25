from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import CustomerFeatures, app
from src.modeling import FEATURES, predict

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200



def test_predict(client):
    customer = {
        "recency": 10,
        "total_orders": 5,
        "total_spent": 1794.24,
        "avg_order": 449.31,
        "purchased_months": 4
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200



def test_predict_invalid(client):
    customer1 = {
        "recency": 10,
        "total_orders": 5,
        "total_spent": 1794.24,
        "avg_order": 449.31
    }

    customer2 = {
        "recency": 10,
        "total_orders": 5,
        "total_spent": 1794.24,
        "avg_order": 449.31,
        "purchased_months": -40
    }

    response = client.post("/predict", json=customer1)
    assert response.status_code == 422

    response = client.post("/predict", json=customer2)
    assert response.status_code == 422



def test_parity(client):
    customer = {
            "recency": 10,
            "total_orders": 4,
            "total_spent": 1794.24,
            "avg_order": 449.31,
            "purchased_months": 3
        }

    df = pd.DataFrame([customer])
    model = joblib.load(MODEL_PATH)

    label, prob = predict(model, df)

    response = client.post('/predict', json=customer)
    assert response.status_code == 200

    result = response.json()

    assert result['label'] == label[0]
    assert result['prob'] == pytest.approx(prob[0], abs=0.001)



def test_schema(client):

    assert list(CustomerFeatures.model_fields) == FEATURES

