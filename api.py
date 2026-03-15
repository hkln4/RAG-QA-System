import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from models.schemas import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    IngestRequest,
    IngestResponse,
    HealthResponse
)
from core.logger import get_logger
from rag_pipeline import (
    load_pdfs,
    split_documents,
    create_vector_store,
    load_vector_store,
    create_qa_chain,
    ask_question
)

from core.mlflow_tracker import setup_mlflow, log_query, log_ingestion

load_dotenv()
logger = get_logger(__name__)
setup_mlflow()

qa_chain = None
retriever = None
vector_store_loaded = False

current_chunk_size = 1000
current_chunk_overlap = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_chain, retriever, vector_store_loaded

    if os.path.exists("faiss_index"):
        logger.info("Loading existing vector store...")
        vs = load_vector_store()
        qa_chain, retriever = create_qa_chain(vs)
        vector_store_loaded = True
        logger.info("Vector store loaded successfully.")
    else:
        logger.warning("No vector store found. Please ingest PDFs first.")
    
    yield

app = FastAPI(
    title= "RAG Q&A API",
    description="Semantic question answering over PDF documents.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model= HealthResponse)
async def health():
    return HealthResponse(
        status="OK",
        vector_store_loaded=vector_store_loaded
    )

@app.post("/ingest", response_model= IngestResponse)
async def ingest(request: IngestRequest):
    global qa_chain, retriever, vector_store_loaded
    current_chunk_size = request.chunk_size
    current_chunk_overlap = request.chunk_overlap

    start_time = time.time()

    if not os.path.exists("pdfs") or not os.listdir("pdfs"):
        raise HTTPException(
            status_code= 400,
            detail= "No PDFs found in pdfs/ directory"
        )
    
    logger.info("Starting PDF ingestion...")

    documents = load_pdfs()
    chunks = split_documents(
        documents,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )
    
    vs = create_vector_store(chunks)
    qa_chain, retriever = create_qa_chain(vs)
    vector_store_loaded = True

    processing_time = (time.time() - start_time) * 1000

    logger.info("Ingestion complete.", extra={"extra": {
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "processing_time_ms": processing_time
    }})

    log_ingestion(
        pages_loaded=len(documents),
        chunks_created=len(chunks),
        processing_time_ms=processing_time,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )
    return IngestResponse(
        pages_loaded=len(documents),
        chunks_created=len(chunks),
        processing_time_ms=processing_time
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    global qa_chain, retriever

    if qa_chain is None:
        raise HTTPException(
            status_code=400,
            detail="Vector store not loaded. Please call /ingest first."
        )

    start_time = time.time()

    logger.info("Processing query.", extra={"extra": {"question": request.question}})

    answer, sources = ask_question(qa_chain, retriever, request.question)

    source_documents = [
        SourceDocument(
            source=doc.metadata.get("source", "Unknown"),
            page=doc.metadata.get("page", 0),
            content=doc.page_content[:200]
        )
        for doc in sources[:request.k]
    ]

    processing_time = (time.time() - start_time) * 1000

    logger.info("Query processed.", extra={"extra": {
        "processing_time_ms": processing_time
    }})

    log_query(
        question=request.question,
        answer=answer,
        processing_time_ms=processing_time,
        chunk_size=current_chunk_size,
        chunk_overlap=current_chunk_overlap,
        k=request.k
    )

    return QueryResponse(
        answer=answer,
        sources=source_documents,
        processing_time_ms=processing_time
    )