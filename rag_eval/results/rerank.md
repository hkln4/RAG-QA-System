# RAG eval: rerank

2026-11-07T18:06:37+00:00 · 51 questions · 464 chunks · chunk 1000/200 · k=5 · embeddings `models/gemini-embedding-001` · retrieval dense · reranker cross-encoder/ms-marco-MiniLM-L-6-v2 · answers `models/gemini-3.1-flash-lite` (T=0.0) · judge `None`

## Retrieval

| Questions | n | hit@1 | hit@5 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| all | 46 | 0.565 | 0.848 | 0.804 | 0.913 | 0.678 |
| paraphrased | 14 | 0.571 | 0.786 | 0.786 | 0.857 | 0.670 |
| not_paraphrased | 32 | 0.562 | 0.875 | 0.812 | 0.938 | 0.682 |
| identifier | 7 | 0.429 | 0.857 | 0.857 | 1.000 | 0.579 |
| multi_hop | 4 | 0.250 | 1.000 | 0.625 | 0.750 | 0.467 |

Retrieval only: no answers generated.
