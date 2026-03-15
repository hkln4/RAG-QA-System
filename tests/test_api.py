import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

with patch("core.mlflow_tracker.setup_mlflow"), \
     patch("mlflow.set_tracking_uri"), \
     patch("mlflow.set_experiment"):
    from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "vector_store_loaded" in data


def test_query_without_vector_store():
    response = client.post("/query", json={
        "question": "What is this document about?",
        "k": 5
    })
    assert response.status_code == 400
    assert "Vector store not loaded" in response.json()["detail"]


def test_query_empty_question():
    response = client.post("/query", json={
        "question": "   ",
        "k": 5
    })
    assert response.status_code == 422


def test_ingest_no_pdfs():
    with patch("api.os.path.exists", return_value=False):
        response = client.post("/ingest", json={
            "chunk_size": 1000,
            "chunk_overlap": 200
        })
    assert response.status_code == 400
    assert "No PDFs found" in response.json()["detail"]


def test_query_invalid_k():
    response = client.post("/query", json={
        "question": "What is this?",
        "k": -1
    })
    assert response.status_code == 422