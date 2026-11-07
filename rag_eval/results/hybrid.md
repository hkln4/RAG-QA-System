# RAG eval: hybrid

2026-11-07T18:06:22+00:00 · 51 questions · 464 chunks · chunk 1000/200 · k=5 · embeddings `models/gemini-embedding-001` · retrieval hybrid · reranker None · answers `models/gemini-3.1-flash-lite` (T=0.0) · judge `claude-haiku-5-5`

## Retrieval

| Questions | n | hit@1 | hit@5 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| all | 46 | 0.630 | 0.913 | 0.902 | 0.989 | 0.757 |
| paraphrased | 14 | 0.643 | 1.000 | 1.000 | 1.000 | 0.776 |
| not_paraphrased | 32 | 0.625 | 0.875 | 0.859 | 0.984 | 0.749 |
| identifier | 7 | 0.857 | 1.000 | 1.000 | 1.000 | 0.905 |
| multi_hop | 4 | 0.250 | 0.250 | 0.250 | 0.875 | 0.363 |

## Generation

| Faithfulness | Answer relevancy | Unanswerable refused | Answerable refused |
|---|---|---|---|
| 0.957 (n=45) | 0.661 (n=46) | 1.000 | 0.022 |
