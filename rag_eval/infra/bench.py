"""Vector-DB benchmark: FAISS vs pgvector vs Qdrant on the same 100k vectors.

    python -m rag_eval.infra.prepare               # once: data + embeddings
    docker compose --profile bench up -d           # pgvector + Qdrant
    python -m rag_eval.infra.bench                 # -> rag_eval/results/infra.{json,md}

Ground truth is numpy brute force, independent of every engine. For each store:
exact top-10 agreement with it, then for each HNSW M: time to a searchable index,
recall@10 against the ground truth and latency at several ef_search values,
index size and memory. A no-op request measures each store's connection cost.
"""
import json
import platform
import random
import time
from pathlib import Path

import numpy as np

# not rag_eval.infra.prepare: it imports rag_pipeline, which pulls in torch (see FaissStore)
from rag_eval.infra.stores import DATA_DIR, THREADS, FaissStore, HNSWParams, PgvectorStore, QdrantStore

SEED = 0
K = 10
M_VALUES = (16, 32)
EF_CONSTRUCTION = 100  # same for all three; their defaults differ (FAISS 40, pgvector 64, Qdrant 100)
EF_SEARCH = (10, 20, 40, 80, 160)
WARMUP = 50
PINGS = 500
RESULTS = Path(__file__).parents[1] / "results"


def ground_truth(corpus, queries, k=K):
    scores = queries @ corpus.T
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(top, order, axis=1), np.sort(scores, axis=1)[:, ::-1][:, :k]


def timed_searches(fn, queries):
    for q in queries[:WARMUP]:
        fn(q)
    ids, ms = [], []
    for q in queries:
        start = time.perf_counter()
        ids.append(fn(q))
        ms.append((time.perf_counter() - start) * 1000)
    return ids, {"p50_ms": round(float(np.percentile(ms, 50)), 3), "p95_ms": round(float(np.percentile(ms, 95)), 3)}


def recall(ids, truth):
    return float(np.mean([len(set(a) & set(t)) / K for a, t in zip(ids, truth.tolist())]))


def exact_agreement(ids, truth, corpus, queries, truth_scores):
    """Set overlap with the ground truth, and whether any miss is only a tie at the cut-off."""
    sets_equal = sum(set(a) == set(t) for a, t in zip(ids, truth.tolist()))
    tie_only = 0
    for a, t, q, ts in zip(ids, truth.tolist(), queries, truth_scores):
        if set(a) != set(t):
            # a different 10th item with (numerically) the same score is a tie, not an error
            kth = ts[-1]
            tie_only += all(abs(float(corpus[i] @ q) - kth) < 1e-5 for i in set(a) ^ set(t))
    return {"identical_top10_sets": sets_equal, "differ_only_by_ties": tie_only, "n": len(ids),
            "recall@10": recall(ids, truth)}


def bench_store(store, corpus, meta, queries, truth, truth_scores):
    result = {"store": store.name, "builds": []}
    start = time.perf_counter()
    store.load(corpus, meta)
    load_s = time.perf_counter() - start
    for i, m in enumerate(M_VALUES):
        print(f"[{store.name}] building HNSW M={m}", flush=True)
        start = time.perf_counter()
        store.build_index(HNSWParams(m=m, ef_construction=EF_CONSTRUCTION))
        build = {"m": m, "ef_construction": EF_CONSTRUCTION, "build_s": round(time.perf_counter() - start, 1),
                 "load_s": round(load_s, 1) if store.name != "qdrant" else None}
        if i == 0:
            if isinstance(store, PgvectorStore):
                # verify each forced path really runs the intended plan
                assert not store.plan_uses_index(queries[0], exact=True), "exact path used the index"
                assert store.plan_uses_index(queries[0], exact=False), "ANN path did not use the index"
            print(f"[{store.name}] exact search", flush=True)
            ids, latency = timed_searches(lambda q: store.search_exact(q, K), queries)
            result["exact"] = {**exact_agreement(ids, truth, corpus, queries, truth_scores), **latency}
        build["ann"] = []
        for ef in EF_SEARCH:
            ids, latency = timed_searches(lambda q: store.search_ann(q, K, ef), queries)
            build["ann"].append({"ef_search": ef, "recall@10": round(recall(ids, truth), 4), **latency})
            print(f"[{store.name}] M={m} ef={ef} recall={build['ann'][-1]['recall@10']} p50={latency['p50_ms']}ms", flush=True)
        build["memory"] = store.memory()
        result["builds"].append(build)
    _, result["ping"] = timed_searches(lambda q: store.ping(), queries[:PINGS])
    return result


def versions(stores):
    import faiss
    out = {"faiss": faiss.__version__}
    for s in stores:
        if isinstance(s, PgvectorStore):
            out["postgres"] = s.conn.execute("SHOW server_version").fetchone()[0]
            out["pgvector"] = s.conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
        if isinstance(s, QdrantStore):
            out["qdrant"] = s.client.info().version
    return out


def to_markdown(r):
    lines = [
        "# Vector-DB benchmark",
        "",
        f"{r['data']['n_corpus']:,} Simple English Wikipedia chunks, {r['data']['n_queries']:,} held-out queries, "
        f"{r['data']['model']} ({r['data']['dims']} dims), k={K}, HNSW ef_construction={EF_CONSTRUCTION}, "
        f"{THREADS} threads each. {r['platform']}; versions: "
        + ", ".join(f"{k} {v}" for k, v in r["versions"].items()) + ".",
        "",
        "Recall is against numpy brute force. FAISS runs in the benchmark process; pgvector and Qdrant run in "
        "Docker containers (4 CPUs, 4 GB) and are reached over localhost, so their latencies include the "
        "round trip shown in the connection row.",
        "",
        "## Exact search",
        "",
        "| Store | Identical top-10 sets | Differ only by ties | recall@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for s in r["stores"]:
        e = s["exact"]
        lines.append(f"| {s['store']} | {e['identical_top10_sets']}/{e['n']} | {e['differ_only_by_ties']} | "
                     f"{e['recall@10']:.4f} | {e['p50_ms']:.2f} | {e['p95_ms']:.2f} |")
    lines += ["", "## HNSW: recall@10 and latency", "",
              "| Store | M | ef_search | recall@10 | p50 ms | p95 ms |", "|---|---|---|---|---|---|"]
    for s in r["stores"]:
        for b in s["builds"]:
            for a in b["ann"]:
                lines.append(f"| {s['store']} | {b['m']} | {a['ef_search']} | {a['recall@10']:.4f} | "
                             f"{a['p50_ms']:.2f} | {a['p95_ms']:.2f} |")
    lines += ["", "## Build, memory and connection cost", "",
              "| Store | M | Load s | Index build s | Memory | Connection p50 / p95 ms |", "|---|---|---|---|---|---|"]
    for s in r["stores"]:
        for b in s["builds"]:
            mem = ", ".join(f"{k.replace('_mib', '')} {v} MiB" for k, v in b["memory"].items())
            load = "incl. in build" if b["load_s"] is None else b["load_s"]
            lines.append(f"| {s['store']} | {b['m']} | {load} | {b['build_s']} | {mem} | "
                         f"{s['ping']['p50_ms']:.3f} / {s['ping']['p95_ms']:.3f} |")
    return "\n".join(lines) + "\n"


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    corpus = np.load(DATA_DIR / "corpus.npy")
    queries = np.load(DATA_DIR / "queries.npy")
    meta = json.loads((DATA_DIR / "corpus_meta.json").read_text())
    data = json.loads((DATA_DIR / "manifest.json").read_text())
    truth, truth_scores = ground_truth(corpus, queries)

    stores = [FaissStore(), PgvectorStore(), QdrantStore()]
    result = {"data": data, "platform": platform.platform(), "seed": SEED, "versions": versions(stores),
              "stores": [bench_store(s, corpus, meta, queries, truth, truth_scores) for s in stores]}
    for s in stores:
        s.close()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "infra.json").write_text(json.dumps(result, indent=2) + "\n")
    (RESULTS / "infra.md").write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == "__main__":
    main()
