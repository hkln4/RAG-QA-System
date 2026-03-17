import os
import mlflow
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from datasets import Dataset
from rag_pipeline import load_pdfs, split_documents, create_vector_store, create_qa_chain, ask_question

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

test_questions = [
    {"question": "What is the main goal of this project?"},
    {"question": "Which dataset was used for training?"},
    {"question": "Which architectures were compared?"},
    {"question": "What is ejection fraction?"},
    {"question": "What are the limitations of manual EF calculation?"},
]

CONFIGS = [
    #{"chunk_size": 500,  "chunk_overlap": 50,  "k": 3},
    #{"chunk_size": 1000, "chunk_overlap": 200, "k": 5},
    {"chunk_size": 1500, "chunk_overlap": 300, "k": 7},
]


def collect_answers(qa_chain, retriever, questions):
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
    }
    for item in questions:
        answer, sources = ask_question(qa_chain, retriever, item["question"])
        contexts = [doc.page_content for doc in sources]
        data["question"].append(item["question"])
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        print(f"Answered: {item['question'][:50]}...")
    return Dataset.from_dict(data)


def run_evaluation(dataset):
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    wrapped_llm = LangchainLLMWrapper(llm)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings
    )
    return results


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("RAG Optimization")

    documents = load_pdfs()

    for config in CONFIGS:
        print(f"\nTesting config: {config}")

        chunks = split_documents(
            documents,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )
        vs = create_vector_store(chunks)
        qa_chain, retriever = create_qa_chain(vs, k=config["k"])

        dataset = collect_answers(qa_chain, retriever, test_questions)
        results = run_evaluation(dataset)

        with mlflow.start_run(run_name=f"chunk_{config['chunk_size']}_k_{config['k']}"):
            mlflow.log_param("chunk_size", config["chunk_size"])
            mlflow.log_param("chunk_overlap", config["chunk_overlap"])
            mlflow.log_param("k", config["k"])
            mlflow.log_metric("faithfulness", sum(results["faithfulness"]) / len(results["faithfulness"])) # faithfulness returns a list. also results["faithfulness"][0] can be used
            mlflow.log_metric("answer_relevancy", sum(results["answer_relevancy"]) / len(results["answer_relevancy"]))

        print(f"Faithfulness: {sum(results["faithfulness"]) / len(results["faithfulness"]):.4f}")
        print(f"Answer Relevancy: {sum(results["answer_relevancy"]) / len(results["answer_relevancy"]):.4f}")