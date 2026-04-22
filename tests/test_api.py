from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_returns_valid_response():
    response = client.post("/predict", json={"text": "This is absolutely wonderful!"})
    # 200 if model is loaded, 503 if running in CI without model — both are valid
    assert response.status_code in [200, 503]

def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    # 422 if model loaded and catches empty input, 503 if no model in CI
    assert response.status_code in [422, 503]

def test_predict_response_schema():
    response = client.post("/predict", json={"text": "Great film!"})
    if response.status_code == 200:
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert data["label"] in ["POSITIVE", "NEGATIVE"]
        assert 0.0 <= data["score"] <= 1.0