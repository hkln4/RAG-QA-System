"""Retrieval variants compared in the quality experiments.

Every variant returns chunk indices ranked best first, so the evaluation
scores them identically. BM25 and fusion come from the app's core.retrieval,
so the experiments measure the code the API runs.
"""
from langchain_core.embeddings import Embeddings

from core.retrieval import BM25, RRF_K, bm25_tokenize, reciprocal_rank_fusion  # noqa: F401  (re-exported)

# BGE v1.5 models are trained with this instruction on the query side only
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbeddings(Embeddings):
    """sentence-transformers model behind the LangChain Embeddings interface."""

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer

        self.model = model_name  # a string: RAGAS reads it as the embedding model's name
        self._encoder = SentenceTransformer(model_name)
        self.query_prefix = BGE_QUERY_PREFIX if "bge-" in model_name else ""

    def embed_documents(self, texts):
        # normalized, so FAISS's L2 distance ranks exactly like cosine similarity
        return self._encoder.encode(texts, batch_size=32, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self._encoder.encode(self.query_prefix + text, normalize_embeddings=True).tolist()


class Reranker:
    def __init__(self, model_name):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, texts):
        scores = self.model.predict([(query, texts[i]) for i in candidates])
        return [i for _, i in sorted(zip(scores, candidates), key=lambda pair: -pair[0])]
