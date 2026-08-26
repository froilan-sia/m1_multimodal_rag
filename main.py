#!/usr/bin/env python3
"""
Multimodal RAG for Apple Silicon (M1/M2/M3)
Usage:
    python main.py ingest --pdf data/report.pdf
    python main.py ingest --pdf data/new_report.pdf --clear
    python main.py query --question "What was the Q3 revenue?"
"""

import argparse
import base64
import os
import sys
import uuid
from io import BytesIO
from pathlib import Path

import chromadb
import pymupdf as fitz
import ollama
from PIL import Image

# ---------- CONFIG ----------
TEXT_MODEL = "llama3.2:3b"          # For final RAG answers (8GB friendly)
VISION_MODEL = "qwen2.5vl:3b"       # For charts (8GB friendly)
CHROMA_PATH = "./chroma_db"
IMAGE_CACHE = "./images_cache"

# Initialize persistent Chroma client (SQLite, not RAM)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
child_collection = chroma_client.get_or_create_collection(name="child_chunks")
parent_collection = chroma_client.get_or_create_collection(name="parent_chunks")

Path(IMAGE_CACHE).mkdir(exist_ok=True)


# ---------- HELPER: Encode Image for Ollama ----------
def encode_image_for_ollama(image_bytes: bytes, max_size=800) -> str:
    """Convert PDF image bytes to Base64 data URI with size limiting."""
    img = Image.open(BytesIO(image_bytes))
    
    # Convert RGBA/P to RGB to avoid JPEG alpha errors
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')
    
    # Downscale massive images to save VRAM on M1
    img.thumbnail((max_size, max_size))
    
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64


# ---------- PHASE 1: INGESTION ----------
def ingest_pdf(pdf_path: str):
    """Parse PDF, extract text, crop images, run VLM, and store in Chroma."""
    print(f"📄 Processing: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f"  ⏳ Page {page_num + 1}/{len(doc)}")
        
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
                
                # Encode for Ollama
                encoded_img = encode_image_for_ollama(image_bytes)
                
                # Prompt designed for financial charts with structured output
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
                print(f"    ⚠️ Skipped image {img_idx} (Error: {e})")
                continue
        
        # 3. Merge text and summaries
        full_page_content = page_text + "\n" + "\n".join(visual_summaries)
        if not full_page_content.strip():
            continue  # Skip completely empty pages
        
        # 4. Split into Parent (big) and Child (small) for retrieval
        parent_text = full_page_content  # Full page is the "Parent"
        
        # Split into ~200 token chunks for children (roughly 800 chars)
        child_chunks = []
        chunk_size = 800
        for i in range(0, len(parent_text), chunk_size):
            child_chunks.append(parent_text[i:i+chunk_size])
        
        if not child_chunks:
            child_chunks = [parent_text]  # Fallback
        
        # 5. Store in Chroma (Parent-Child)
        parent_id = str(uuid.uuid4())
        metadata = {
            "source": os.path.basename(pdf_path),
            "page": page_num + 1,
            "type": "hybrid"
        }
        
        # Store Parent (full context) - Persisted to disk, not RAM
        parent_collection.add(
            ids=[parent_id],
            documents=[parent_text],
            metadatas=[metadata]
        )
        
        # Store Children (granular search)
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
    print("✅ Ingestion complete!")


# ---------- PHASE 2: QUERY ----------
def query_rag(question: str):
    """Retrieve relevant context using Child chunks, fetch Parent, ask LLM."""
    print(f"❓ Query: {question}")
    
    # 1. Retrieve top matching child chunks
    results = child_collection.query(
        query_texts=[question],
        n_results=3
    )
    
    if not results["ids"] or not results["ids"][0]:
        print("⚠️ No relevant documents found in the database.")
        return
    
    # 2. Fetch the full Parent contexts
    parent_ids = list(set([m["parent_ref"] for m in results["metadatas"][0]]))
    parent_results = parent_collection.get(ids=parent_ids)
    full_context = "\n\n---\n\n".join(parent_results["documents"])
    
    # 3. Build prompt for the text-only LLM
    prompt = f"""
    You are a financial research assistant. Answer the question based strictly on the context below.
    If the context contains chart summaries or tables, use those numbers specifically.
    If you cannot answer from the context, say "I don't have that information."
    
    Context:
    {full_context}
    
    Question: {question}
    Answer:
    """
    
    # 4. Generate answer locally
    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    print("\n🤖 Answer:")
    print(response["message"]["content"])
    print("\n📚 Sources:", ", ".join(parent_ids))


# ---------- DATABASE CLEAR FUNCTION ----------
def clear_database():
    """Delete all collections to reset the database."""
    try:
        chroma_client.delete_collection("child_chunks")
        chroma_client.delete_collection("parent_chunks")
        print("✅ Database cleared successfully!")
    except ValueError:
        print("ℹ️ Database was already empty. Nothing to clear.")
    except Exception as e:
        print(f"⚠️ Could not clear database: {e}")


# ---------- CLI ENTRY POINT ----------
def main():
    parser = argparse.ArgumentParser(description="M1 Multimodal RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Ingest command with --clear flag
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF")
    ingest_parser.add_argument("--pdf", required=True, help="Path to PDF file")
    ingest_parser.add_argument("--clear", action="store_true", help="Clear the database before ingesting")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("--question", required=True, help="Your question")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        if not os.path.exists(args.pdf):
            print(f"❌ File not found: {args.pdf}")
            sys.exit(1)
        
        # Clear the database if the flag is set
        if args.clear:
            clear_database()
            # Re-initialize collections after clearing
            global child_collection, parent_collection
            child_collection = chroma_client.get_or_create_collection(name="child_chunks")
            parent_collection = chroma_client.get_or_create_collection(name="parent_chunks")
        
        ingest_pdf(args.pdf)
    
    elif args.command == "query":
        query_rag(args.question)


if __name__ == "__main__":
    main()
