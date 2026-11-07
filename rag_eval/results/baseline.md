# RAG eval: baseline

2026-11-07T17:31:57+00:00 · 51 questions · 464 chunks · chunk 1000/200 · k=5 · embeddings `models/gemini-embedding-001` · retrieval dense · reranker None · answers `models/gemini-3.1-flash-lite` (T=0.0) · judge `claude-haiku-5-5`

## Retrieval

| Questions | n | hit@1 | hit@5 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| all | 46 | 0.500 | 0.848 | 0.826 | 0.913 | 0.627 |
| paraphrased | 14 | 0.571 | 0.857 | 0.857 | 0.929 | 0.675 |
| not_paraphrased | 32 | 0.469 | 0.844 | 0.812 | 0.906 | 0.605 |
| identifier | 7 | 0.429 | 0.571 | 0.571 | 0.857 | 0.506 |
| multi_hop | 4 | 0.500 | 0.750 | 0.500 | 0.750 | 0.598 |

## Generation

| Faithfulness | Answer relevancy | Unanswerable refused | Answerable refused |
|---|---|---|---|
| 0.955 (n=41) | 0.604 (n=46) | 1.000 | 0.109 |
