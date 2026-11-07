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

def test_query_retrieves_requested_k():
    with patch("api.qa_chain", MagicMock()), patch("api.retriever", MagicMock()), \
         patch("api.log_query"), \
         patch("api.ask_question", return_value=("answer", [])) as ask:
        response = client.post("/query", json={"question": "What is this?", "k": 12})
    assert response.status_code == 200
    assert ask.call_args.kwargs["k"] == 12


def test_ingest_updates_chunk_settings_logged_by_query():
    # ingest rebinds module globals; patch them so the change is undone after the test
    with patch("api.qa_chain", None), patch("api.retriever", None), \
         patch("api.vector_store_loaded", False), \
         patch("api.current_chunk_size", 1500), patch("api.current_chunk_overlap", 300):
        _ingest_then_query()


def _ingest_then_query():
    with patch("api.os.path.exists", return_value=True), \
         patch("api.os.listdir", return_value=["a.pdf"]), \
         patch("api.load_pdfs", return_value=[]), \
         patch("api.split_documents", return_value=[]), \
         patch("api.create_vector_store"), \
         patch("api.create_qa_chain", return_value=(MagicMock(), MagicMock())), \
         patch("api.log_ingestion"):
        response = client.post("/ingest", json={"chunk_size": 500, "chunk_overlap": 50})
    assert response.status_code == 200

    with patch("api.ask_question", return_value=("answer", [])), \
         patch("api.log_query") as log_query:
        client.post("/query", json={"question": "What is this?", "k": 3})
    assert log_query.call_args.kwargs["chunk_size"] == 500
    assert log_query.call_args.kwargs["chunk_overlap"] == 50
