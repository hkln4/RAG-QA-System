from unittest.mock import MagicMock

from langchain_core.documents import Document

from rag_pipeline import ask_question


def make_fakes():
    docs = [Document(page_content="alpha"), Document(page_content="beta")]
    retriever = MagicMock()
    retriever.invoke.return_value = docs
    chain = MagicMock()
    chain.invoke.return_value = "answer"
    return docs, retriever, chain


def test_ask_question_retrieves_once():
    _, retriever, chain = make_fakes()
    ask_question(chain, retriever, "q")
    assert retriever.invoke.call_count == 1


def test_ask_question_answers_from_the_returned_sources():
    docs, retriever, chain = make_fakes()
    answer, sources = ask_question(chain, retriever, "q")
    assert answer == "answer"
    assert sources == docs
    chain.invoke.assert_called_once_with({"context": "alpha\n\nbeta", "question": "q"})


def test_ask_question_passes_k_to_the_retriever():
    _, retriever, chain = make_fakes()
    ask_question(chain, retriever, "q", k=3)
    retriever.invoke.assert_called_once_with("q", k=3)


def test_ask_question_without_k_uses_the_retriever_default():
    _, retriever, chain = make_fakes()
    ask_question(chain, retriever, "q")
    retriever.invoke.assert_called_once_with("q")


def test_per_call_k_is_honoured_by_a_real_faiss_retriever():
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import DeterministicFakeEmbedding

    store = FAISS.from_texts([f"text {i}" for i in range(10)], DeterministicFakeEmbedding(size=16))
    retriever = store.as_retriever(search_kwargs={"k": 7})
    _, _, chain = make_fakes()
    _, sources = ask_question(chain, retriever, "q", k=3)
    assert len(sources) == 3
