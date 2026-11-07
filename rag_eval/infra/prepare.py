"""Build the vector-DB benchmark data: 100k Simple English Wikipedia chunks.

Every database in the benchmark loads the same vectors from these files, so
differences between them cannot come from the embeddings.

    python -m rag_eval.infra.prepare   ->  rag_eval/infra/data/{corpus,queries}.npy, manifest.json

Corpus: articles in id order, split with the app's splitter (1000/200), the
first 100,000 chunks. Queries: the first chunk of each of the next 1,000
articles, so no query text is in the corpus. Embeddings: bge-small-en-v1.5,
normalized; queries carry the BGE query instruction, as in retrieval use.
"""
import hashlib
import json
import platform
import time
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from langchain_core.documents import Document

from rag_eval.infra.stores import DATA_DIR
from rag_eval.retrievers import LocalEmbeddings
from rag_pipeline import split_documents

REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"  # wikimedia/wikipedia, pinned
PARQUET = "20231101.simple/train-00000-of-00001.parquet"
SHA256 = "31bded16768a47c286becd292079122f5d7d4397a17b87d4250a00ccd581e6f0"
URL = f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{REVISION}/{PARQUET}"

MODEL = "BAAI/bge-small-en-v1.5"
N_CORPUS = 100_000
N_QUERIES = 1_000
SEED = 0


def download():
    path = DATA_DIR / "simplewiki.parquet"
    if not path.exists():
        print("downloading", URL, flush=True)
        urllib.request.urlretrieve(URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch for {path}: {digest}")
    return path


def select_chunks(path):
    table = pq.read_table(path, columns=["id", "title", "text"]).to_pylist()
    articles = sorted(table, key=lambda a: int(a["id"]))
    docs = [Document(page_content=a["text"], metadata={"article_id": int(a["id"]), "title": a["title"]}) for a in articles]
    chunks = split_documents(docs)  # keeps article order
    corpus = chunks[:N_CORPUS]
    last_corpus_article = corpus[-1].metadata["article_id"]
    queries, seen = [], set()
    for chunk in chunks[N_CORPUS:]:
        article = chunk.metadata["article_id"]
        # first chunk of each later article; skip the tail of the article the corpus ends in
        if article != last_corpus_article and article not in seen:
            seen.add(article)
            queries.append(chunk)
            if len(queries) == N_QUERIES:
                break
    return corpus, queries


def main():
    import torch  # only needed to embed; keeps this module importable without torch (CI)

    torch.manual_seed(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    corpus, queries = select_chunks(download())
    embedder = LocalEmbeddings(MODEL)

    start = time.time()
    corpus_vectors = np.asarray(embedder.embed_documents([c.page_content for c in corpus]), dtype=np.float32)
    corpus_seconds = time.time() - start
    query_vectors = np.asarray([embedder.embed_query(q.page_content) for q in queries], dtype=np.float32)

    np.save(DATA_DIR / "corpus.npy", corpus_vectors)
    np.save(DATA_DIR / "queries.npy", query_vectors)
    (DATA_DIR / "corpus_meta.json").write_text(json.dumps(
        [[c.metadata["article_id"], c.metadata["title"]] for c in corpus]))
    manifest = {
        "source": f"wikimedia/wikipedia {PARQUET} @ {REVISION}", "license": "CC BY-SA 3.0 / GFDL",
        "model": MODEL, "dims": int(corpus_vectors.shape[1]),
        "n_corpus": len(corpus), "n_queries": len(queries),
        "corpus_articles": len({c.metadata["article_id"] for c in corpus}),
        "embedding_seconds": round(corpus_seconds, 1), "device": str(embedder._encoder.device),
        "platform": platform.platform(), "torch": torch.__version__,
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
