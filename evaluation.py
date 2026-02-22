import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from datasets import Dataset
from rag_pipeline import load_pdfs, split_documents, create_vector_store, load_vector_store, create_qa_chain, ask_question

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

test_questions = [
    {
        "question": "What is the main goal of this project?",
        "reference": "The main goal is to automate left ventricle segmentation and estimate ejection fraction using echocardiography videos."
    },
    {
        "question": "Which dataset was used for training?",
        "reference": "The CAMUS dataset was used for training, with fine-tuning on EchoNet-Dynamic dataset."
    },
    {
        "question": "What architectures were compared?",
        "reference": "U-Net and nnU-Net architectures were compared."
    },
    {
        "question": "What is ejection fraction?",
        "reference": "Ejection fraction is a measure of how much blood the heart pumps with each contraction."
    },
    {
        "question": "What are the limitations of manual EF calculation?",
        "reference": "Manual EF calculation is time-consuming and subject to inter-expert variability."
    },
]

def collect_answers(qa_chain, retriever, questions):
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "reference": [],
    }

    for item in questions:
        answer, sources = ask_question(qa_chain, retriever, item["question"])
        contexts = [doc.page_content for doc in sources]

        data["question"].append(item["question"])
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["reference"].append(item["reference"])

        print(f"Answered: {item['question'][:50]}...")

    return Dataset.from_dict(data)

def run_evaluation(dataset):
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import faithfulness, answer_relevancy

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
    vector_store = load_vector_store()
    qa_chain, retriever = create_qa_chain(vector_store)

    print("Collecting answers...")
    dataset = collect_answers(qa_chain, retriever, test_questions)

    print("Running evaluation..")
    results = run_evaluation(dataset)

    print("\nEvaluation Results")
    print(results)