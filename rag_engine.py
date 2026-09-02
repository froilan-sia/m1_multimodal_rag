#!/usr/bin/env python3
"""
RAG Engine for Multimodal PDF Processing
Core functions reused across CLI (main.py) and Gradio UI (app.py)
"""

import base64
import os
import uuid
from io import BytesIO
from pathlib import Path

import chromadb
import pymupdf as fitz
import ollama
from PIL import Image

# ---------- CONFIG ----------
TEXT_MODEL = "llama3.2:3b"          # For final RAG answers
VISION_MODEL = "qwen2.5vl:3b"       # For charts
CHROMA_PATH = "./chroma_db"
IMAGE_CACHE = "./images_cache"

# Initialize persistent Chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
child_collection = chroma_client.get_or_create_collection(name="child_chunks")
parent_collection = chroma_client.get_or_create_collection(name="parent_chunks")

Path(IMAGE_CACHE).mkdir(exist_ok=True)


# ---------- HELPER: Encode Image for Ollama ----------
def encode_image_for_ollama(image_bytes: bytes, max_size=800) -> str:
    """Convert PDF image bytes to raw Base64 string."""
    img = Image.open(BytesIO(image_bytes))

    # Convert RGBA/P to RGB to avoid JPEG alpha errors
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')

    # Downscale massive images to save VRAM
    img.thumbnail((max_size, max_size))

    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64


# ---------- PHASE 1: INGESTION ----------
def ingest_pdf(pdf_path: str, progress=None, status=None):
    """
    Parse PDF, extract text, crop images, run VLM, and store in Chroma.

    Args:
        pdf_path: Path to PDF file
        progress: Gradio progress object (optional)
        status: Callback for status updates (optional)
    """
    if status:
        status(f"📄 Opening {os.path.basename(pdf_path)}...")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc[page_num]

        # Update progress
        if progress:
            progress((page_num + 1) / total_pages, desc=f"Processing page {page_num + 1}/{total_pages}")
        if status:
            status(f"📄 Processing page {page_num + 1}/{total_pages}...")

        # 1. Extract main text
        page_text = page.get_text("text").strip()
        if not page_text:
            page_text = "[No extractable text on this page]"

        # 2. Find and process images
        image_list = page.get_images(full=True)
        visual_summaries = []

        for img_idx, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                encoded_img = encode_image_for_ollama(image_bytes)

                prompt = """
                Extract key insights from this chart and return valid JSON.
                Use this schema: {"chart_type": "", "x_axis": [], "y_axis": [], "key_trend": "", "data_points": []}
                If it's not a chart, describe it briefly in text.
                """

                response = ollama.chat(
                    model=VISION_MODEL,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        "images": [encoded_img]
                    }]
                )
                summary = response["message"]["content"]
                visual_summaries.append(f"[Chart on page {page_num+1}]: {summary}")

            except Exception as e:
                if status:
                    status(f"⚠️ Skipped image {img_idx} on page {page_num+1}: {str(e)[:50]}...")
                continue

        # 3. Merge text and summaries
        full_page_content = page_text + "\n" + "\n".join(visual_summaries)
        if not full_page_content.strip():
            continue

        # 4. Split into Parent and Child chunks
        parent_text = full_page_content
        child_chunks = []
        chunk_size = 800
        for i in range(0, len(parent_text), chunk_size):
            child_chunks.append(parent_text[i:i+chunk_size])

        if not child_chunks:
            child_chunks = [parent_text]

        # 5. Store in Chroma (Parent-Child) with page metadata
        parent_id = str(uuid.uuid4())
        metadata = {
            "source": os.path.basename(pdf_path),
            "page": page_num + 1,           # Store page number
            "type": "hybrid"
        }

        parent_collection.add(
            ids=[parent_id],
            documents=[parent_text],
            metadatas=[metadata]
        )

        child_ids = []
        child_metadatas = []
        for idx, chunk in enumerate(child_chunks):
            child_id = f"{parent_id}_child_{idx}"
            child_ids.append(child_id)
            child_metadatas.append({
                **metadata,
                "parent_ref": parent_id
            })

        child_collection.add(
            ids=child_ids,
            documents=child_chunks,
            metadatas=child_metadatas
        )

    doc.close()

    if status:
        status("✅ Ingestion complete!")

    return f"✅ Successfully ingested {os.path.basename(pdf_path)} ({total_pages} pages)"


# ---------- PHASE 2: QUERY ----------
def query_rag(question: str) -> dict:
    """
    Retrieve relevant context and generate an answer.

    Returns:
        dict: {"answer": str, "sources": list[dict]}
    """
    # 1. Retrieve top matching child chunks
    results = child_collection.query(
        query_texts=[question],
        n_results=3
    )

    if not results["ids"] or not results["ids"][0]:
        return {
            "answer": "⚠️ No relevant documents found. Please ingest a PDF first.",
            "sources": []
        }

    # 2. Fetch the full Parent contexts with metadata
    parent_ids = list(set([m["parent_ref"] for m in results["metadatas"][0]]))
    parent_results = parent_collection.get(ids=parent_ids)

    # 3. Build full context from parent documents
    full_context = "\n\n---\n\n".join(parent_results["documents"])

    # 4. Get metadata for source display (list of dicts with page and section)
    source_metadata = []
    if parent_results.get("metadatas"):
        for i, meta in enumerate(parent_results["metadatas"]):
            page = meta.get("page", "Unknown")
            source_metadata.append({"page": page, "section": f"Section {i+1}"})

    # 5. Build prompt
    prompt = f"""
    You are a financial research assistant. Answer the question based strictly on the context below.
    If the context contains chart summaries or tables, use those numbers specifically.
    If you cannot answer from the context, say "I don't have that information."

    Context:
    {full_context}

    Question: {question}
    Answer:
    """

    # 6. Generate answer
    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response["message"]["content"],
        "sources": source_metadata
    }


# ---------- DATABASE CLEAR ----------
def clear_database():
    """Delete all collections to reset the database, then reinitialize them."""
    global child_collection, parent_collection
    
    try:
        chroma_client.delete_collection("child_chunks")
        chroma_client.delete_collection("parent_chunks")
        
        child_collection = chroma_client.get_or_create_collection(name="child_chunks")
        parent_collection = chroma_client.get_or_create_collection(name="parent_chunks")
        
        return "✅ Database cleared and reinitialized successfully!"
        
    except ValueError:
        child_collection = chroma_client.get_or_create_collection(name="child_chunks")
        parent_collection = chroma_client.get_or_create_collection(name="parent_chunks")
        return "ℹ️ Database was already empty. Reinitialized."
        
    except Exception as e:
        return f"⚠️ Could not clear database: {e}"


# ---------- UTILITY: List Ingestion Status ----------
def get_ingested_documents():
    """Return list of all ingested documents."""
    try:
        results = parent_collection.get()
        sources = set()
        for meta in results.get("metadatas", []):
            if "source" in meta:
                sources.add(meta["source"])
        return sorted(list(sources))
    except Exception:
        return []