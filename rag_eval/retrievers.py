"""Retrieval variants compared in the quality experiments.

Every variant returns chunk indices ranked best first, so the evaluation
scores them identically.
"""
import re
import unicodedata

from langchain_core.embeddings import Embeddings

RRF_K = 60  # standard Reciprocal Rank Fusion constant (Cormack et al., 2009)

# BGE v1.5 models are trained with this instruction on the query side only
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbeddings(Embeddings):
    """sentence-transformers model behind the LangChain Embeddings interface."""

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.query_prefix = BGE_QUERY_PREFIX if "bge-" in model_name else ""

    def embed_documents(self, texts):
        # normalized, so FAISS's L2 distance ranks exactly like cosine similarity
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()


def bm25_tokenize(text):
    """Lowercased NFKC tokens that keep identifiers such as gv-1.3-007 or id.am-04 whole.

    Splitting on every non-alphanumeric character would turn "GV-1.3-007" into
    "gv 1 3 007", and exact identifier matching is what BM25 adds over dense
    retrieval. NFKC also unfolds ligatures ("ﬁ" -> "fi") left by PDF extraction.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text)


class BM25:
    def __init__(self, texts):
        from rank_bm25 import BM25Okapi

        self.index = BM25Okapi([bm25_tokenize(t) for t in texts])

    def rank(self, query, n):
        scores = self.index.get_scores(bm25_tokenize(query))
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:n]


def reciprocal_rank_fusion(rankings, n, k=RRF_K):
    """Fuse ranked index lists by sum of 1 / (k + rank); uses ranks only, so the
    incomparable BM25 and cosine score scales need no normalization or weights."""
    fused = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=lambda i: -fused[i])[:n]


class Reranker:
    def __init__(self, model_name):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, texts):
        scores = self.model.predict([(query, texts[i]) for i in candidates])
        return [i for _, i in sorted(zip(scores, candidates), key=lambda pair: -pair[0])]
