# RAG eval: bge_small

2026-11-07T18:06:08+00:00 · 51 questions · 464 chunks · chunk 1000/200 · k=5 · embeddings `BAAI/bge-small-en-v1.5` · retrieval dense · reranker None · answers `models/gemini-3.1-flash-lite` (T=0.0) · judge `None`

## Retrieval

| Questions | n | hit@1 | hit@5 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| all | 46 | 0.304 | 0.587 | 0.543 | 0.598 | 0.425 |
| paraphrased | 14 | 0.286 | 0.357 | 0.357 | 0.429 | 0.318 |
| not_paraphrased | 32 | 0.312 | 0.688 | 0.625 | 0.672 | 0.472 |
| identifier | 7 | 0.000 | 0.143 | 0.143 | 0.143 | 0.029 |
| multi_hop | 4 | 0.000 | 0.750 | 0.375 | 0.625 | 0.328 |

Retrieval only: no answers generated.
