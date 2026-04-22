from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_positive():
    response = client.post("/predict", json={"text": "This is absolutely wonderful!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["POSITIVE", "NEGATIVE"]
    assert 0.0 <= data["score"] <= 1.0

def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422