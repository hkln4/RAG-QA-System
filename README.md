# RAG Document Q&A Service

Source-attributed question answering over PDFs: LangChain, FAISS and Gemini
behind a FastAPI API with a Gradio UI, MLflow tracking, Prometheus/Grafana
monitoring, tests and CI. The repo also contains a measured study of what
actually moves RAG quality, and of how much the choice of vector database
matters.

![Interface](assets/gradio_ui.png)

## Findings

- **Retrieval components set the quality.** Switching the embedding model
  changed recall@5 by up to 0.28, and hybrid search (BM25 + dense) cut false
  "not available" answers from 5/46 to 1/46. Chunk size (500 vs. 1000) and a
  cross-encoder reranker made no measurable difference.
- **The vector DB does not change results.** With the same embeddings, FAISS,
  pgvector and Qdrant return the same exact top 10 for 1000/1000 queries, and
  their HNSW indexes trade recall for latency in the same way. The differences
  are latency architecture (in-process vs. network) and operations.
- The API now uses the winning configuration, hybrid retrieval, by default.

## Running

```bash
cp .env.example .env        # add GOOGLE_API_KEY
docker compose up --build   # API :8000 (UI at /ui), MLflow :5001, Prometheus :9090, Grafana :3000
```

Local development: `pip install -r requirements.txt`, then
`uvicorn api:app --reload --port 8000`.

| Method | Endpoint   | Description                                           |
| ------ | ---------- | ----------------------------------------------------- |
| GET    | `/health`  | Service status and whether the vector store is loaded |
| POST   | `/ingest`  | Build the index from `pdfs/` (configurable chunking)  |
| POST   | `/query`   | Answer a question with sources, pages and latency     |
| GET    | `/metrics` | Prometheus metrics                                    |
| GET    | `/ui`      | Gradio interface                                      |

`RETRIEVAL_MODE` selects retrieval: `hybrid` (default) or `dense` (FAISS only).
Answers come from `gemini-3.1-flash-lite`, embeddings from `gemini-embedding-001`.

## Evaluation set

Three public-domain NIST publications (AI RMF 1.0, Generative AI Profile
AI 600-1, CSF 2.0; 144 pages) and 51 questions in `rag_eval/questions.json`:
46 answerable (including 7 identifier lookups, 4 multi-hop and 14 reworded to
share few words with their evidence) and 5 unanswerable ones that test refusal.

Answers are labelled with their page and verbatim evidence text rather than
chunk ids, since chunk boundaries move with chunk size; a retrieved chunk is
relevant when it contains at least half of an evidence span. Metrics: hit@k,
recall@k (share of evidence items found) and MRR for retrieval; RAGAS
faithfulness and answer relevancy for answers, judged by `claude-haiku-5-5`.

## Quality experiments

Each row changes one variable from the baseline (the API's former defaults:
chunk 1000/200, Gemini embeddings, dense FAISS, k = 5). Δ is the mean
per-question difference from the baseline with a paired bootstrap 95% CI; an
interval that contains 0 is not a measurable difference on 46 questions.

| Variable | Value | recall@5 | MRR | Δ recall@5 | Δ MRR | Identifier hit@5 |
|---|---|---|---|---|---|---|
| – | **baseline** | 0.83 | 0.63 | – | – | 0.57 |
| Chunk size | 500 / 100 | 0.83 | 0.59 | +0.00 [−0.11, +0.11] | −0.03 [−0.14, +0.07] | 0.86 |
| Chunk size | 2000 / 400 | *pending* | | | | |
| Embedding | bge-small-en-v1.5 | 0.54 | 0.43 | **−0.28** [−0.41, −0.15] | **−0.20** [−0.33, −0.07] | 0.14 |
| Embedding | bge-base-en-v1.5 | 0.59 | 0.46 | **−0.24** [−0.37, −0.12] | **−0.17** [−0.29, −0.05] | 0.00 |
| Retrieval | BM25 + dense (RRF) | 0.90 | 0.76 | +0.08 [−0.01, +0.18] | **+0.13** [+0.03, +0.24] | 1.00 |
| Reranker | MiniLM-L6 cross-encoder | 0.80 | 0.68 | −0.02 [−0.12, +0.08] | +0.05 [−0.06, +0.16] | 0.86 |

| Answers | Faithfulness | Answer relevancy | Answerable refused | Unanswerable refused |
|---|---|---|---|---|
| baseline | 0.96 | 0.60 | 5 / 46 | 5 / 5 |
| hybrid | 0.96 | 0.66 | **1 / 46** | 5 / 5 |

Hybrid's gains come mostly from identifier lookups ("What does GV-1.3-007
recommend?"), which dense embeddings match poorly and BM25 matches exactly:
three of the four answers it recovers are identifier lookups. False refusals
−0.09 [−0.17, −0.02], answer relevancy +0.06 [+0.00, +0.12].
Full tables: [`rag_eval/results/experiments.md`](rag_eval/results/experiments.md).

> **Pending.** Chunk size 2000/400 has not run yet: the Gemini free tier allows
> 1,000 embedding requests a day and it needs 219 more. A live `/query` smoke
> test with hybrid retrieval waits on the same quota; offline, the API's
> retriever returns the same top 5 as the evaluated hybrid run for 51/51
> questions. `python -m rag_eval.experiments` fills the row in from the cache.

## Vector DB benchmark

100,000 Simple English Wikipedia chunks (CC BY-SA), 1,000 held-out queries,
local `bge-small-en-v1.5` embeddings shared by all three stores (a different
model from the quality axis, to avoid API quotas). Ground truth is numpy brute
force; HNSW uses the same M and ef_construction everywhere and each store gets
4 CPUs. Apple M2; pgvector and Qdrant run in Docker. Latency is the median of
three passes.

| | Exact top 10 = ground truth | Exact p50 | HNSW recall@10 / p50 at ef 40 | at ef 160 | Build (M 16 / 32) | Connection p50 |
|---|---|---|---|---|---|---|
| FAISS 1.13 | 1000 / 1000 | 2.9 ms | 0.939 / 0.07 ms | 0.988 / 0.23 ms | 7 / 8 s | – |
| pgvector 0.8.6 | 1000 / 1000 | 14.4 ms | 0.954 / 1.6 ms | 0.991 / 2.6 ms | 16 / 29 s | 0.11 ms |
| Qdrant 1.20 | 1000 / 1000 | 7.0 ms | 0.941 / 3.1 ms | 0.988 / 3.1 ms | 24 / 22 s | 0.69 ms |

Full grid and memory: [`rag_eval/results/infra.md`](rag_eval/results/infra.md).
For HNSW search FAISS is 10–100× faster because it runs in process (2–5× for
exact search); at this size Qdrant's latency barely depends on ef_search, i.e.
it is dominated by the request path.

| | FAISS | pgvector | Qdrant |
|---|---|---|---|
| Runs as | Library in the app process | Postgres extension | Separate service |
| Metadata filters | Only by id (`IDSelector`) | SQL `WHERE`, applied after the HNSW scan | Payload filters inside the HNSW search |
| Persistence | Save and load index files yourself | Postgres (WAL, backups, replicas) | On-disk storage, WAL, snapshots |
| Deletes | Not in an HNSW index | Regular SQL | By id |
| Operations | None, but lives in app memory | One more database, or none if you run Postgres already | One more service to run and upgrade |
| Gotchas met here | Segfaults beside torch on macOS (two OpenMP runtimes) | Parallel HNSW build fails on Docker's 64 MB `/dev/shm` | Skips HNSW for small segments unless `indexing_threshold` is lowered; indexes in the background |

## Reproducing

```bash
pip install -r requirements-eval.txt        # adds torch, sentence-transformers, Qdrant/pgvector clients

python rag_eval/download_corpus.py          # NIST PDFs, checksum-pinned
python -m rag_eval.experiments              # quality axis -> rag_eval/results/experiments.md

python -m rag_eval.infra.prepare            # Wikipedia vectors (~10 min on an M2)
docker compose --profile bench up -d        # pgvector + Qdrant
python -m rag_eval.infra.bench              # -> rag_eval/results/infra.md
```

The quality axis needs `GOOGLE_API_KEY` and, for the judge, `ANTHROPIC_API_KEY`;
`JUDGE_BUDGET_USD` (default 0.30) caps judge spend. Every API result is cached
in `rag_eval/cache.sqlite` under a hash of its input, so reruns are free.
Quality runs log to MLflow when the compose MLflow is up.

## Design decisions

- **Evidence spans, not chunk ids, as labels**, so one label set serves every
  chunk size.
- **One variable per experiment, with paired bootstrap CIs**, because 46
  questions make small differences unreliable.
- **Retrieval metrics for every configuration, RAGAS only for the baseline and
  the winner.** Each experiment changes a retrieval component, which retrieval
  metrics measure directly; the free tier and the judge budget cannot cover
  more.
- **A judge from another vendor than the answer model**, so the model does not
  grade itself.
- **Reciprocal Rank Fusion for hybrid search**: it uses ranks only, so BM25 and
  cosine scores need no weights. The BM25 tokenizer keeps identifiers such as
  `gv-1.3-007` whole and unfolds PDF ligatures.
- **The app and the evaluation share one retrieval module**
  (`core/retrieval.py`), so the eval measures the code the API runs.
- **Identical vectors and an independent ground truth for the DB benchmark**,
  so any difference comes from the engine.

## Limitations

- One domain and 46 answerable questions; the questions were drafted by an LLM
  and reviewed by hand.
- LLM judges err. On the same 41 answers, the earlier flash-lite judge and
  Haiku agreed on faithfulness for 39 and each misjudged one correct answer.
  Faithfulness measures agreement with the retrieved context, not correctness,
  and RAGAS answer relevancy penalises the short answers the prompt asks for.
- The eval answers at temperature 0; the app uses 0.3.
- The DB benchmark uses one sequential client on 100k vectors that fit in
  memory: no concurrency, no disk-bound scale, and Docker on macOS adds its own
  overhead. Multithreaded HNSW builds are not deterministic, so recall moves in
  the third decimal between runs.

## Tests and CI

```bash
PYTHONPATH=. pytest tests/              # RUN_DB_TESTS=1 also checks the pgvector and Qdrant adapters
```

GitHub Actions runs the tests and builds the Docker image on every push and
pull request to `main`.
