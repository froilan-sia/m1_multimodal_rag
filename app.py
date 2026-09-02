#!/usr/bin/env python3
"""
Multimodal RAG Gradio UI
Run with: python app.py
"""

import os
import shutil
from pathlib import Path

import gradio as gr

# Import the core engine
from rag_engine import ingest_pdf, query_rag, clear_database, get_ingested_documents

# ---------- CONFIG ----------
UPLOAD_DIR = Path("./data")
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------- UI FUNCTIONS ----------
def process_upload(file_obj, progress=gr.Progress()):
    """Handle PDF upload and ingestion with progress bar."""
    if file_obj is None:
        return "❌ Please upload a PDF file first."

    pdf_path = UPLOAD_DIR / os.path.basename(file_obj.name)
    shutil.copy(file_obj.name, pdf_path)

    try:
        result = ingest_pdf(str(pdf_path), progress=progress)
        return f"{result}\n\n📄 File saved to: {pdf_path}"
    except Exception as e:
        return f"❌ Error during ingestion: {str(e)}"


def chat_response(message, history):
    """Handle user questions and return responses."""
    if not message or not message.strip():
        return history

    docs = get_ingested_documents()
    if not docs:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ No documents ingested. Please upload and ingest a PDF first."})
        return history

    result = query_rag(message)
    answer = result["answer"]

    if result["sources"]:
        source_text = "\n\n📚 **Sources:** "
        sources_list = []
        for source in result["sources"][:3]:
            if isinstance(source, dict):
                page = source.get("page", "Unknown")
                section = source.get("section", "Section")
                sources_list.append(f"Page {page} ({section})")
            else:
                sources_list.append(str(source))
        answer += source_text + ", ".join(sources_list)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history


def reset_database():
    """Clear the database, chat history, and reset file upload."""
    result = clear_database()
    return result, [], gr.update(value=None)


def get_status():
    """Get current system status."""
    docs = get_ingested_documents()
    if docs:
        return f"✅ {len(docs)} document(s) ingested: {', '.join(docs)}"
    return "ℹ️ No documents ingested. Upload and ingest a PDF to get started."


# ---------- BUILD UI ----------
def create_ui():
    with gr.Blocks(title="Multimodal RAG Assistant") as demo:

        gr.Markdown("""
        # 📄 Zero-Cost Local Multimodal RAG

        A privacy-first assistant that answers questions from complex PDFs containing text, tables, and charts.

        **How it works:**
        1. Upload a PDF and click **Ingest**
        2. Wait for processing (charts will be analyzed by the VLM)
        3. Ask questions about the document content

        🔒 **100% local** — No data ever leaves your machine.
        """)

        # Status Bar
        with gr.Row():
            status_bar = gr.Textbox(value=get_status(), label="📊 Status", interactive=False, scale=3)
            refresh_btn = gr.Button("🔄 Refresh", size="sm", scale=0)
            clear_db_btn = gr.Button("🗑️ Clear Database", size="sm", variant="stop", scale=0)

        # Upload & Chat
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload & Ingest")
                file_upload = gr.File(label="Upload PDF", file_types=[".pdf"], height=100)
                ingest_btn = gr.Button("📥 Ingest PDF", variant="primary", size="lg")
                upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=3)

            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask Questions")
                chatbot = gr.Chatbot(label="Chat", height=400, avatar_images=(None, "🤖"))
                with gr.Row():
                    msg = gr.Textbox(label="Your question", placeholder="e.g., What was the total revenue?", scale=4, container=False)
                    send_btn = gr.Button("Send", variant="primary", scale=0)
                clear_chat_btn = gr.Button("🗑️ Clear Chat", size="sm")

        # Event Handlers
        ingest_btn.click(fn=process_upload, inputs=[file_upload], outputs=[upload_status]).then(
            fn=get_status, inputs=[], outputs=[status_bar])
        msg.submit(fn=chat_response, inputs=[msg, chatbot], outputs=[chatbot]).then(fn=lambda: "", outputs=[msg])
        send_btn.click(fn=chat_response, inputs=[msg, chatbot], outputs=[chatbot]).then(fn=lambda: "", outputs=[msg])
        clear_chat_btn.click(fn=lambda: [], outputs=[chatbot])
        clear_db_btn.click(fn=reset_database, inputs=[], outputs=[upload_status, chatbot, file_upload]).then(
            fn=get_status, inputs=[], outputs=[status_bar])
        refresh_btn.click(fn=get_status, inputs=[], outputs=[status_bar])
        demo.load(fn=get_status, inputs=[], outputs=[status_bar])

    return demo


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    print("🚀 Starting Gradio UI...")
    print("📍 Opening at: http://127.0.0.1:7860")
    print("🔒 All processing is 100% local.")
    
    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css="footer {visibility: hidden}"
    )