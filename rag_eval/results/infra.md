# Vector-DB benchmark

100,000 Simple English Wikipedia chunks, 1,000 held-out queries, BAAI/bge-small-en-v1.5 (384 dims), k=10, HNSW ef_construction=100, 4 threads each. macOS-15.7.3-arm64-arm-64bit-Mach-O; versions: faiss 1.13.2, postgres 17.11 (Debian 17.11-1.pgdg12+2), pgvector 0.8.6, qdrant 1.20.0.

Recall is against numpy brute force. FAISS runs in the benchmark process; pgvector and Qdrant run in Docker containers (4 CPUs, 4 GB) and are reached over localhost, so their latencies include the round trip shown in the connection row.

## Exact search

| Store | Identical top-10 sets | Differ only by ties | recall@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| faiss | 1000/1000 | 0 | 1.0000 | 2.88 | 3.06 |
| pgvector | 1000/1000 | 0 | 1.0000 | 14.06 | 14.91 |
| qdrant | 1000/1000 | 0 | 1.0000 | 6.08 | 7.52 |

## HNSW: recall@10 and latency

| Store | M | ef_search | recall@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| faiss | 16 | 10 | 0.7599 | 0.03 | 0.05 |
| faiss | 16 | 20 | 0.8732 | 0.04 | 0.07 |
| faiss | 16 | 40 | 0.9362 | 0.07 | 0.12 |
| faiss | 16 | 80 | 0.9712 | 0.14 | 0.22 |
| faiss | 16 | 160 | 0.9875 | 0.23 | 0.35 |
| faiss | 32 | 10 | 0.8036 | 0.04 | 0.07 |
| faiss | 32 | 20 | 0.8991 | 0.06 | 0.09 |
| faiss | 32 | 40 | 0.9500 | 0.09 | 0.14 |
| faiss | 32 | 80 | 0.9792 | 0.17 | 0.28 |
| faiss | 32 | 160 | 0.9925 | 0.28 | 0.44 |
| pgvector | 16 | 10 | 0.8232 | 1.08 | 1.42 |
| pgvector | 16 | 20 | 0.9060 | 1.18 | 1.43 |
| pgvector | 16 | 40 | 0.9548 | 1.47 | 1.86 |
| pgvector | 16 | 80 | 0.9800 | 1.80 | 2.39 |
| pgvector | 16 | 160 | 0.9913 | 2.47 | 3.20 |
| pgvector | 32 | 10 | 0.8936 | 1.61 | 1.96 |
| pgvector | 32 | 20 | 0.9423 | 1.73 | 2.16 |
| pgvector | 32 | 40 | 0.9745 | 1.97 | 2.55 |
| pgvector | 32 | 80 | 0.9890 | 2.39 | 3.29 |
| pgvector | 32 | 160 | 0.9961 | 3.19 | 4.52 |
| qdrant | 16 | 10 | 0.7611 | 2.16 | 2.81 |
| qdrant | 16 | 20 | 0.8740 | 1.43 | 2.67 |
| qdrant | 16 | 40 | 0.9380 | 2.37 | 3.31 |
| qdrant | 16 | 80 | 0.9725 | 2.12 | 3.32 |
| qdrant | 16 | 160 | 0.9869 | 3.00 | 4.03 |
| qdrant | 32 | 10 | 0.7860 | 2.52 | 3.52 |
| qdrant | 32 | 20 | 0.8868 | 1.66 | 2.88 |
| qdrant | 32 | 40 | 0.9470 | 2.64 | 4.20 |
| qdrant | 32 | 80 | 0.9788 | 3.35 | 5.72 |
| qdrant | 32 | 160 | 0.9899 | 3.42 | 4.92 |

## Build, memory and connection cost

| Store | M | Load s | Index build s | Memory | Connection p50 / p95 ms |
|---|---|---|---|---|---|
| faiss | 16 | 0.0 | 6.5 | index 160.2 MiB | 0.000 / 0.000 |
| faiss | 32 | 0.0 | 8.0 | index 172.4 MiB | 0.000 / 0.000 |
| pgvector | 16 | 1.0 | 15.6 | table 156.5 MiB, index 193.4 MiB, container 425.2 MiB | 0.126 / 0.334 |
| pgvector | 32 | 1.0 | 25.8 | table 156.5 MiB, index 200.5 MiB, container 440.2 MiB | 0.126 / 0.334 |
| qdrant | 16 | incl. in build | 22.9 | container 395.6 MiB | 1.379 / 2.442 |
| qdrant | 32 | incl. in build | 23.3 | container 385.4 MiB | 1.379 / 2.442 |
