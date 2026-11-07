import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# shared with rag_eval so the evaluation measures the same model and prompt as the app
EMBEDDING_MODEL = "models/gemini-embedding-001"
ANSWER_MODEL = "models/gemini-3.1-flash-lite"
PROMPT_TEMPLATE = """
    You are a precise question-answering assistant. Answer in the same language as the question.

    Rules:
    - Answer ONLY what is asked, nothing more
    - Be concise and direct
    - Use ONLY information from the context
    - If not in context, say "This information is not available."

    Context: {context}

    Question: {question}

    Answer:
    """

def load_pdfs(pdf_dir="pdfs/"):
    loader = DirectoryLoader(pdf_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"{len(documents)} pages loaded.")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # offset within the page; rag_eval uses it to match evidence spans
    )

    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunks created.")

    return chunks

def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("Vector store created and saved.")

    return vector_store

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store

def create_qa_chain(vector_store, k: int = 7):
    llm = ChatGoogleGenerativeAI(
        model=ANSWER_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # retrieval is not part of the chain: ask_question retrieves once and feeds the
    # same documents to the LLM and to the caller
    chain = prompt | llm | StrOutputParser()

    return chain, retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_question(chain, retriever, question, k=None):
    # k overrides the retriever's default search_kwargs for this call only
    sources = retriever.invoke(question, k=k) if k else retriever.invoke(question)
    answer = chain.invoke({"context": format_docs(sources), "question": question})

    return answer, sources