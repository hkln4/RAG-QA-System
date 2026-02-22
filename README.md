# RAG-Based Document Q&A System

A retrieval-augmented generation (RAG) pipeline that enables semantic question answering over uploaded PDF documents. Built with LangChain, FAISS, Google Gemini, and Gradio.

## Demo

![UI Screenshot](assets/ui_screenshot.png)

---

## Features

- Upload multiple PDFs and ask natural language questions
- Semantic search using FAISS vector store and Gemini embeddings
- Source attribution — see which page and document the answer came from
- Evaluated with RAGAS framework: **0.96 Faithfulness / 0.81 Answer Relevancy**

---

## Architecture
```
PDF(s) → PyPDFLoader → RecursiveCharacterTextSplitter
       → GoogleGenerativeAIEmbeddings → FAISS Vector Store
       → Retriever → Gemini 2.5 Flash → Answer + Sources
```

---

## Tech Stack

- **LangChain** — pipeline orchestration
- **Google Gemini** — LLM (gemini-2.5-flash-lite) and embeddings (gemini-embedding-001)
- **FAISS** — vector store for semantic search
- **Gradio** — interactive UI
- **RAGAS** — evaluation framework

---

## Installation
```bash
git clone https://github.com/hkln4/RAG-QA-System.git
cd rag-qa-system

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Setup

Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com).

---

## Usage
```bash
python app.py
```

Open `http://localhost:7860` in your browser, upload your PDFs, and start asking questions.

To run the RAGAS evaluation:
```bash
python evaluation.py
```

---

## Evaluation

The pipeline was evaluated using the [RAGAS](https://docs.ragas.io) framework:

| Metric | Score |
|--------|-------|
| Faithfulness | 0.96 |
| Answer Relevancy | 0.81 |

**Faithfulness** measures whether the generated answers are grounded in the retrieved context. **Answer Relevancy** measures how relevant the answers are to the given questions.

---

## Project Structure
```
rag-qa-system/
├── app.py              # Gradio UI
├── rag_pipeline.py     # Core RAG pipeline
├── evaluation.py       # RAGAS evaluation
├── pdfs/               # Upload your PDFs here
├── faiss_index/        # Saved vector store
├── assets/             # Screenshots for README
├── .env                # API key (not committed)
├── .env.example        # API key template
└── requirements.txt
```
