# Vector-DB benchmark

100,000 Simple English Wikipedia chunks, 1,000 held-out queries, BAAI/bge-small-en-v1.5 (384 dims), k=10, HNSW ef_construction=100, 4 threads each. macOS-15.7.3-arm64-arm-64bit-Mach-O; versions: faiss 1.13.2, postgres 17.11 (Debian 17.11-1.pgdg12+2), pgvector 0.8.6, qdrant 1.20.0.

Latency: median of 3 passes over all queries, with the range of the per-pass p50. Recall is against numpy brute force. FAISS runs in the benchmark process; pgvector and Qdrant run in Docker containers (4 CPUs, 4 GB) and are reached over localhost, so their latencies include the round trip shown in the connection row.

## Exact search

| Store | Identical top-10 sets | Differ only by ties | recall@10 | p50 ms (range) | p95 ms |
|---|---|---|---|---|---|
| faiss | 1000/1000 | 0 | 1.0000 | 2.87 (2.86–2.88) | 3.05 |
| pgvector | 1000/1000 | 0 | 1.0000 | 14.42 (14.18–14.46) | 15.86 |
| qdrant | 1000/1000 | 0 | 1.0000 | 6.95 (6.95–7.08) | 8.75 |

## HNSW: recall@10 and latency

| Store | M | ef_search | recall@10 | p50 ms (range) | p95 ms |
|---|---|---|---|---|---|
| faiss | 16 | 10 | 0.7611 | 0.03 (0.03–0.04) | 0.06 |
| faiss | 16 | 20 | 0.8739 | 0.05 (0.05–0.05) | 0.07 |
| faiss | 16 | 40 | 0.9385 | 0.07 (0.07–0.07) | 0.12 |
| faiss | 16 | 80 | 0.9718 | 0.13 (0.13–0.13) | 0.20 |
| faiss | 16 | 160 | 0.9880 | 0.23 (0.23–0.24) | 0.35 |
| faiss | 32 | 10 | 0.8039 | 0.04 (0.04–0.04) | 0.06 |
| faiss | 32 | 20 | 0.8995 | 0.05 (0.05–0.06) | 0.09 |
| faiss | 32 | 40 | 0.9494 | 0.09 (0.09–0.10) | 0.14 |
| faiss | 32 | 80 | 0.9785 | 0.16 (0.15–0.16) | 0.25 |
| faiss | 32 | 160 | 0.9917 | 0.28 (0.28–0.28) | 0.44 |
| pgvector | 16 | 10 | 0.8261 | 1.23 (1.07–1.49) | 1.81 |
| pgvector | 16 | 20 | 0.9041 | 1.29 (1.26–1.41) | 1.75 |
| pgvector | 16 | 40 | 0.9542 | 1.63 (1.57–1.70) | 2.08 |
| pgvector | 16 | 80 | 0.9788 | 1.91 (1.86–2.23) | 2.42 |
| pgvector | 16 | 160 | 0.9908 | 2.59 (2.57–2.86) | 3.36 |
| pgvector | 32 | 10 | 0.8982 | 1.63 (1.54–1.65) | 2.22 |
| pgvector | 32 | 20 | 0.9481 | 1.72 (1.69–1.76) | 2.22 |
| pgvector | 32 | 40 | 0.9765 | 1.98 (1.97–2.02) | 2.48 |
| pgvector | 32 | 80 | 0.9890 | 2.42 (2.41–2.43) | 3.21 |
| pgvector | 32 | 160 | 0.9962 | 3.21 (3.18–3.23) | 4.50 |
| qdrant | 16 | 10 | 0.7686 | 2.88 (2.81–2.94) | 3.98 |
| qdrant | 16 | 20 | 0.8726 | 2.89 (2.79–2.97) | 3.99 |
| qdrant | 16 | 40 | 0.9407 | 3.05 (2.64–3.37) | 4.25 |
| qdrant | 16 | 80 | 0.9738 | 2.78 (2.73–4.81) | 11.01 |
| qdrant | 16 | 160 | 0.9878 | 3.14 (3.10–3.68) | 6.05 |
| qdrant | 32 | 10 | 0.7943 | 3.06 (2.74–3.14) | 4.32 |
| qdrant | 32 | 20 | 0.8884 | 2.36 (1.33–2.62) | 3.39 |
| qdrant | 32 | 40 | 0.9467 | 2.06 (2.05–2.52) | 3.22 |
| qdrant | 32 | 80 | 0.9773 | 2.08 (1.98–2.46) | 3.19 |
| qdrant | 32 | 160 | 0.9896 | 2.42 (2.11–2.49) | 3.51 |

## Build, memory and connection cost

| Store | M | Load s | Index build s | Memory | Connection p50 / p95 ms |
|---|---|---|---|---|---|
| faiss | 16 | 0.0 | 6.7 | index 160.2 MiB | 0.000 / 0.000 |
| faiss | 32 | 0.0 | 7.9 | index 172.4 MiB | 0.000 / 0.000 |
| pgvector | 16 | 1.6 | 15.6 | table 156.5 MiB, index 193.4 MiB, container 441.1 MiB | 0.111 / 0.131 |
| pgvector | 32 | 1.6 | 29.0 | table 156.5 MiB, index 200.6 MiB, container 444.8 MiB | 0.111 / 0.131 |
| qdrant | 16 | incl. in build | 23.5 | container 347.0 MiB | 0.691 / 1.046 |
| qdrant | 32 | incl. in build | 21.5 | container 343.0 MiB | 0.691 / 1.046 |
