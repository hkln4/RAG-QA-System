# Quality experiments

46 answerable questions; each row changes one variable from the baseline (chunk 1000/200, gemini-embedding-001, dense FAISS, no reranker). Δ is the mean per-question difference from the baseline with a paired bootstrap 95% CI; an interval that contains 0 is not a measurable difference on this set. Run on macOS-15.7.3-arm64-arm-64bit-Mach-O.

| Variable | Value | Chunks | hit@1 | hit@5 | recall@5 | recall@10 | MRR | Δ recall@5 [95% CI] | Δ MRR [95% CI] | Identifier hit@5 | Reworded hit@5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | – | 464 | 0.500 | 0.848 | 0.826 | 0.913 | 0.627 | – | – | 0.571 | 0.857 |
| chunk size | 500 / 100 | 889 | 0.435 | 0.848 | 0.826 | 0.957 | 0.593 | +0.000 [-0.109, +0.109] | -0.033 [-0.137, +0.072] | 0.857 | 0.929 |
| embedding | bge-small-en-v1.5 | 464 | 0.304 | 0.587 | 0.543 | 0.598 | 0.425 | -0.283 [-0.413, -0.152] | -0.201 [-0.329, -0.075] | 0.143 | 0.357 |
| embedding | bge-base-en-v1.5 | 464 | 0.348 | 0.630 | 0.587 | 0.696 | 0.458 | -0.239 [-0.370, -0.120] | -0.168 [-0.291, -0.048] | 0.000 | 0.571 |
| retrieval | BM25 + dense (RRF) | 464 | 0.630 | 0.913 | 0.902 | 0.989 | 0.757 | +0.076 [-0.011, +0.185] | +0.131 [+0.028, +0.236] | 1.000 | 1.000 |
| reranker | MiniLM-L6 cross-encoder | 464 | 0.565 | 0.848 | 0.804 | 0.913 | 0.678 | -0.022 [-0.120, +0.076] | +0.052 [-0.060, +0.163] | 0.857 | 0.786 |
| chunk size | 2000 / 400 | pending: API quota or judge budget | | | | | | | | | |

## Generation (baseline and retrieval winner)

| Config | Faithfulness | Answer relevancy | Unanswerable refused | Answerable refused |
|---|---|---|---|---|
| baseline | 0.955 (n=41) | 0.604 (n=46) | 1.000 | 0.109 |
| hybrid | 0.957 (n=45) | 0.661 (n=46) | 1.000 | 0.022 |

Judge `claude-haiku-5-5`, answer-relevancy embeddings `BAAI/bge-base-en-v1.5`, answers `models/gemini-3.1-flash-lite` at temperature 0.0.
