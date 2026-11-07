"""One VectorStore interface with FAISS, pgvector and Qdrant adapters.

All three use cosine similarity on normalized vectors (inner product), the
same HNSW M / ef_construction, and 4 CPU threads, so differences come from the
engines rather than from configuration.
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
THREADS = 4  # matches the CPU limit of the pgvector and Qdrant containers


@dataclass(frozen=True)
class HNSWParams:
    m: int
    ef_construction: int


def container_memory(name):
    """Resident memory of a Docker container as reported by docker stats, in MiB."""
    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name],
                         capture_output=True, text=True, timeout=30).stdout.strip()
    used = out.split("/")[0].strip()
    value, unit = float(used[:-3]), used[-3:]
    return round(value * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}[unit], 1)


class VectorStore:
    """load() inserts vectors; build_index() builds HNSW; searches return ids best first."""

    name = ""

    def load(self, vectors, meta): ...
    def build_index(self, params: HNSWParams): ...
    def search_exact(self, query, k): ...
    def search_ann(self, query, k, ef_search): ...
    def ping(self): ...
    def memory(self): ...
    def close(self): ...


class FaissStore(VectorStore):
    """In-process library: no server, no network, nothing to persist unless written to a file.

    Keep torch out of the process that runs this: both ship an OpenMP runtime, and
    with two loaded FAISS's multithreaded HNSW build crashes on macOS. The benchmark
    therefore avoids importing rag_pipeline, whose LangChain imports pull in torch.
    """

    name = "faiss"

    def __init__(self):
        import faiss

        self.faiss = faiss
        faiss.omp_set_num_threads(THREADS)

    def load(self, vectors, meta):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.flat = self.faiss.IndexFlatIP(self.vectors.shape[1])  # exact: compares every vector
        self.flat.add(self.vectors)

    def build_index(self, params):
        self.hnsw = self.faiss.IndexHNSWFlat(self.vectors.shape[1], params.m, self.faiss.METRIC_INNER_PRODUCT)
        self.hnsw.hnsw.efConstruction = params.ef_construction
        self.hnsw.add(self.vectors)

    def search_exact(self, query, k):
        return self.flat.search(query[None, :], k)[1][0].tolist()

    def search_ann(self, query, k, ef_search):
        self.hnsw.hnsw.efSearch = ef_search
        return self.hnsw.search(query[None, :], k)[1][0].tolist()

    def ping(self):
        pass  # in-process: there is no connection to pay for

    def memory(self):
        return {"index_mib": round(self.faiss.serialize_index(self.hnsw).nbytes / 2**20, 1)}

    def close(self):
        pass


class PgvectorStore(VectorStore):
    name = "pgvector"
    CONTAINER = "rag-qa-system-bench-pgvector-1"

    def __init__(self, dsn="postgresql://postgres:bench@localhost:5434/bench"):
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)

    def load(self, vectors, meta):
        self.conn.execute("DROP TABLE IF EXISTS items")
        self.conn.execute(f"CREATE TABLE items (id int PRIMARY KEY, embedding vector({vectors.shape[1]}), "
                          "article_id int, title text)")
        with self.conn.cursor() as cur, cur.copy(
                "COPY items (id, embedding, article_id, title) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["int4", "vector", "int4", "text"])
            for i, (vec, (article_id, title)) in enumerate(zip(vectors, meta)):
                copy.write_row((i, vec, article_id, title))
        self.conn.execute("ANALYZE items")

    def build_index(self, params):
        self.conn.execute("DROP INDEX IF EXISTS items_hnsw")
        self.conn.execute("SET maintenance_work_mem = '1GB'")  # build the graph in memory, not on disk
        self.conn.execute(f"SET max_parallel_maintenance_workers = {THREADS - 1}")  # + the leader = THREADS
        self.conn.execute(f"CREATE INDEX items_hnsw ON items USING hnsw (embedding vector_cosine_ops) "
                          f"WITH (m = {params.m}, ef_construction = {params.ef_construction})")

    SQL = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"

    def _search(self, settings, query, k):
        # each path is forced and verified by plan_uses_index: left to the planner,
        # exact and HNSW runs could silently measure the same plan
        with self.conn.transaction():
            for s in settings:
                self.conn.execute(s)
            return [r[0] for r in self.conn.execute(self.SQL, (query, k)).fetchall()]

    def search_exact(self, query, k):
        return self._search(["SET LOCAL enable_indexscan = off"], query, k)

    def search_ann(self, query, k, ef_search):
        return self._search(["SET LOCAL enable_seqscan = off", f"SET LOCAL hnsw.ef_search = {ef_search}"], query, k)

    def plan_uses_index(self, query, exact):
        setting = "SET LOCAL enable_indexscan = off" if exact else "SET LOCAL enable_seqscan = off"
        with self.conn.transaction():
            self.conn.execute(setting)
            plan = self.conn.execute("EXPLAIN " + self.SQL, (query, 10)).fetchall()
        return any("items_hnsw" in row[0] for row in plan)

    def ping(self):
        self.conn.execute("SELECT 1").fetchone()

    def memory(self):
        row = self.conn.execute("SELECT pg_relation_size('items'), pg_relation_size('items_hnsw')").fetchone()
        return {"table_mib": round(row[0] / 2**20, 1), "index_mib": round(row[1] / 2**20, 1),
                "container_mib": container_memory(self.CONTAINER)}

    def close(self):
        self.conn.close()


class QdrantStore(VectorStore):
    name = "qdrant"
    CONTAINER = "rag-qa-system-bench-qdrant-1"
    COLLECTION = "bench"

    def __init__(self, url="http://localhost:6333"):
        from qdrant_client import QdrantClient, models

        self.models = models
        self.client = QdrantClient(url=url, prefer_grpc=False, timeout=600)  # REST, like one request per query

    def load(self, vectors, meta):
        self.vectors, self.meta = vectors, meta  # Qdrant fixes HNSW params per collection; build_index reloads

    def build_index(self, params):
        m = self.models
        self.client.delete_collection(self.COLLECTION)
        self.client.create_collection(
            self.COLLECTION,
            vectors_config=m.VectorParams(size=self.vectors.shape[1], distance=m.Distance.COSINE),
            hnsw_config=m.HnswConfigDiff(m=params.m, ef_construct=params.ef_construction, max_indexing_threads=THREADS),
            # Qdrant skips HNSW for segments under indexing_threshold (default 10,000 KB) and
            # searches them by full scan, so an "HNSW" measurement could silently be exact
            # search; 1 KB makes every segment get an index (0 would disable indexing)
            optimizers_config=m.OptimizersConfigDiff(indexing_threshold=1),
        )
        self.client.upload_collection(
            self.COLLECTION, vectors=self.vectors, ids=list(range(len(self.vectors))),
            payload=[{"article_id": a, "title": t} for a, t in self.meta], batch_size=1000, wait=True)
        self.wait_until_indexed()

    def wait_until_indexed(self, poll=1.0, timeout=1800):
        """Qdrant builds HNSW in the background; timing or searching before it finishes measures a partial index."""
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            info = self.client.get_collection(self.COLLECTION)
            if info.status == self.models.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= len(self.vectors):
                return
            time.sleep(poll)
        raise TimeoutError(f"Qdrant indexed {info.indexed_vectors_count} of {len(self.vectors)} vectors in {timeout}s")

    def _search(self, query, k, params):
        points = self.client.query_points(self.COLLECTION, query=query.tolist(), limit=k, search_params=params).points
        return [p.id for p in points]

    def search_exact(self, query, k):
        return self._search(query, k, self.models.SearchParams(exact=True))

    def search_ann(self, query, k, ef_search):
        return self._search(query, k, self.models.SearchParams(hnsw_ef=ef_search))

    def ping(self):
        self.client.get_collections()  # lightest call through the same client and connection pool

    def memory(self):
        return {"container_mib": container_memory(self.CONTAINER)}

    def close(self):
        self.client.close()
