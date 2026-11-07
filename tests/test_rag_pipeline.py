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
