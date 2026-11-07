from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from core.retrieval import HybridRetriever

TEXTS = [f"general risk management text number {i}" for i in range(30)] + [
    "GV-1.3-007 Devise a plan to halt development or deployment of a GAI system.",
]


def make_retriever(k=5):
    # random-but-fixed embeddings: the dense half carries no signal, so BM25 decides
    store = FAISS.from_documents([Document(page_content=t) for t in TEXTS], DeterministicFakeEmbedding(size=16))
    return HybridRetriever(vector_store=store, k=k)


def test_bm25_brings_an_exact_identifier_match_into_the_results():
    # RRF rewards agreement, so the BM25-only hit need not be first, but it must
    # make the top k even when the dense half ranks it nowhere
    docs = make_retriever(k=5).invoke("What does GV-1.3-007 recommend?")
    assert any(d.page_content.startswith("GV-1.3-007") for d in docs)


def test_default_and_per_call_k():
    retriever = make_retriever(k=5)
    assert len(retriever.invoke("risk")) == 5
    assert len(retriever.invoke("risk", k=12)) == 12


def test_results_are_distinct_chunks():
    docs = make_retriever().invoke("risk management", k=20)
    assert len({d.page_content for d in docs}) == 20
