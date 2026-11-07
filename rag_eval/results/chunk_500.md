# RAG eval: chunk_500

2026-11-07T18:02:02+00:00 · 51 questions · 889 chunks · chunk 500/100 · k=5 · embeddings `models/gemini-embedding-001` · retrieval dense · reranker None · answers `models/gemini-3.1-flash-lite` (T=0.0) · judge `None`

## Retrieval

| Questions | n | hit@1 | hit@5 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| all | 46 | 0.435 | 0.848 | 0.826 | 0.957 | 0.593 |
| paraphrased | 14 | 0.500 | 0.929 | 0.929 | 0.929 | 0.693 |
| not_paraphrased | 32 | 0.406 | 0.812 | 0.781 | 0.969 | 0.549 |
| identifier | 7 | 0.429 | 0.857 | 0.857 | 1.000 | 0.597 |
| multi_hop | 4 | 0.250 | 0.750 | 0.500 | 1.000 | 0.398 |

Retrieval only: no answers generated.
