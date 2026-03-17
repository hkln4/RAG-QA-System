import os
import mlflow

EXPERIMENT_NAME = "RAG Pipeline Evaluation"
_mlflow_initialized = False


def setup_mlflow():
    global _mlflow_initialized
    if _mlflow_initialized:
        return
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        _mlflow_initialized = True
    except Exception as e:
        print(f"MLflow connection failed: {e}. Continuing without tracking.")


def log_query(question, answer, processing_time_ms, chunk_size, chunk_overlap, k, faithfulness=None, answer_relevancy=None):
    try:
        setup_mlflow()
        with mlflow.start_run():
            mlflow.log_param("chunk_size", chunk_size)
            mlflow.log_param("chunk_overlap", chunk_overlap)
            mlflow.log_param("k", k)
            mlflow.log_param("question", question[:100])
            mlflow.log_metric("processing_time_ms", processing_time_ms)
            if faithfulness is not None:
                mlflow.log_metric("faithfulness", faithfulness)
            if answer_relevancy is not None:
                mlflow.log_metric("answer_relevancy", answer_relevancy)
    except Exception as e:
        print(f"MLflow logging failed: {e}")


def log_ingestion(pages_loaded, chunks_created, processing_time_ms, chunk_size, chunk_overlap):
    try:
        setup_mlflow()
        with mlflow.start_run(run_name="ingestion"):
            mlflow.log_param("chunk_size", chunk_size)
            mlflow.log_param("chunk_overlap", chunk_overlap)
            mlflow.log_metric("pages_loaded", pages_loaded)
            mlflow.log_metric("chunks_created", chunks_created)
            mlflow.log_metric("processing_time_ms", processing_time_ms)
    except Exception as e:
        print(f"MLflow logging failed: {e}")