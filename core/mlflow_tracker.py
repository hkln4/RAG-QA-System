import mlflow
from typing import Optional
import os

EXPERIMENT_NAME = "RAG Pipeline Evaluation"

def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

def log_query(
    question: str,
    answer: str,
    processing_time_ms: float,
    chunk_size: int,
    chunk_overlap: int,
    k: int,
    faithfulness: Optional[float] = None,
    answer_relevancy: Optional[float] = None,
):
    with mlflow.start_run():
        # Parameters
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("k", k)
        mlflow.log_param("question", question[:100])

        # Metrics
        mlflow.log_metric("processing_time_ms", processing_time_ms)

        if faithfulness is not None:
            mlflow.log_metric("faithfulness", faithfulness)

        if answer_relevancy is not None:
            mlflow.log_metric("answer_relevancy", answer_relevancy)


def log_ingestion(
    pages_loaded: int,
    chunks_created: int,
    processing_time_ms: float,
    chunk_size: int,
    chunk_overlap: int,
):
    with mlflow.start_run(run_name="ingestion"):
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_metric("pages_loaded", pages_loaded)
        mlflow.log_metric("chunks_created", chunks_created)
        mlflow.log_metric("processing_time_ms", processing_time_ms)
