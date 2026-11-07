# RAG eval: bge_base

2026-11-07T17:32:23+00:00 · 51 questions · 464 chunks · chunk 1000/200 · k=5 · embeddings `BAAI/bge-base-en-v1.5` · retrieval dense · reranker None · answers `models/gemini-3.1-flash-lite` (T=0.0) · judge `None`

## Retrieval

| Questions | n | hit@1 | hit@5 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| all | 46 | 0.348 | 0.630 | 0.587 | 0.696 | 0.458 |
| paraphrased | 14 | 0.357 | 0.571 | 0.571 | 0.643 | 0.451 |
| not_paraphrased | 32 | 0.344 | 0.656 | 0.594 | 0.719 | 0.461 |
| identifier | 7 | 0.000 | 0.000 | 0.000 | 0.286 | 0.041 |
| multi_hop | 4 | 0.250 | 1.000 | 0.625 | 0.750 | 0.458 |

Retrieval only: no answers generated.
