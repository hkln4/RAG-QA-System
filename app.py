import gradio as gr
import os
import shutil
from rag_pipeline import (
    load_pdfs,
    split_documents,
    create_vector_store,
    create_qa_chain,
    ask_question
)

qa_chain = None
retriever = None

def process_pdfs(pdf_files):
    global qa_chain, retriever

    os.makedirs("pdfs", exist_ok=True)

    for pdf in pdf_files:
        shutil.copy(pdf.name, f"pdfs/{os.path.basename(pdf.name)}")

    documents = load_pdfs()
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    qa_chain, retriever = create_qa_chain(vector_store)

    return f"{len(documents)} pages, {len(chunks)} chunks processed. You can now ask questions."

def answer_question(question):
    global qa_chain, retriever

    if qa_chain is None:
        return "Please upload a PDF first.", ""

    answer, sources = ask_question(qa_chain, retriever, question)

    source_text = ""
    for i, doc in enumerate(sources[:3]):
        source_text += f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')} - Page {doc.metadata.get('page', '?')}\n\n"
        source_text += f"{doc.page_content[:200]}...\n\n"

    return answer, source_text

with gr.Blocks(title="RAG Document Q&A") as app:
    gr.Markdown("# RAG-Based Document Q&A System")
    gr.Markdown("Upload your PDFs and ask questions about their content.")

    with gr.Row():
        pdf_input = gr.File(
            label="Upload PDFs",
            file_types=[".pdf"],
            file_count="multiple"
        )
        upload_btn = gr.Button("Process PDFs", variant="primary")

    upload_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="Ask a question about the uploaded documents..."
        )
        ask_btn = gr.Button("Ask", variant="primary")

    answer_output = gr.Textbox(label="Answer", interactive=False)
    sources_output = gr.Markdown(label="Sources")

    upload_btn.click(process_pdfs, inputs=[pdf_input], outputs=[upload_status])
    ask_btn.click(answer_question, inputs=[question_input], outputs=[answer_output, sources_output])

if __name__ == "__main__":
    app.launch()