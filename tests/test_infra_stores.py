import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rag_eval.infra.bench import exact_agreement, ground_truth, recall
from rag_eval.infra.stores import FaissStore, HNSWParams


def data(n=400, d=16, q=20, seed=0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, d)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = rng.normal(size=(q, d)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    meta = [[i, f"title {i}"] for i in range(n)]
    return vectors, meta, queries


def test_ground_truth_matches_full_sort():
    vectors, _, queries = data()
    truth, scores = ground_truth(vectors, queries, k=10)
    full = np.argsort(-(queries @ vectors.T), axis=1)[:, :10]
    assert (truth == full).all()
    assert (np.diff(scores, axis=1) <= 0).all()


def test_exact_agreement_counts_a_tie_at_the_cutoff_as_a_tie():
    corpus = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # items 0 and 1 are duplicates
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    truth, scores = np.array([[0]]), np.array([[1.0]])
    result = exact_agreement([[1]], truth, corpus, queries, scores)
    assert result["identical_top10_sets"] == 0 and result["differ_only_by_ties"] == 1


def check_store(store, vectors, meta, queries):
    store.load(vectors, meta)
    store.build_index(HNSWParams(m=16, ef_construction=100))
    truth, _ = ground_truth(vectors, queries, k=10)
    exact = [store.search_exact(q, 10) for q in queries]
    assert exact == truth.tolist()
    # with ef far above k on 400 points, HNSW should find (nearly) everything
    assert recall([store.search_ann(q, 10, 200) for q in queries], truth) >= 0.95


def test_faiss_store_exact_equals_brute_force():
    # in a fresh process: other test modules import rag_pipeline, which loads torch when
    # it is installed, and two OpenMP runtimes crash FAISS's threaded HNSW build (see FaissStore)
    code = ("from tests.test_infra_stores import check_store, data; "
            "from rag_eval.infra.stores import FaissStore; check_store(FaissStore(), *data())")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]


needs_db = pytest.mark.skipif(os.getenv("RUN_DB_TESTS") != "1",
                              reason="needs `docker compose --profile bench up`; overwrites the bench data")


@needs_db
def test_pgvector_store_exact_equals_brute_force():
    from rag_eval.infra.stores import PgvectorStore

    check_store(PgvectorStore(), *data())


@needs_db
def test_qdrant_store_exact_equals_brute_force():
    from rag_eval.infra.stores import QdrantStore

    check_store(QdrantStore(), *data())
