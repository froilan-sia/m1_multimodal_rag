# Multimodal RAG on Apple Silicon (M1)

A privacy-first, zero-cost search assistant that runs locally on Apple Mac M1/M2/M3. This selective hybrid pipeline parses complex PDFs, extracts text and tables, and uses a Vision Language Model (VLM) exclusively for embedded charts and figures—all without sending any data to the cloud.

> **📚 Series:** This repository contains code for both Part 1 and Part 2 of the tutorial series.
> - **[Read Part 1](https://medium.com/@froilan.sia/zero-cost-multimodal-rag-chromadb-313930800633)** → Command-line RAG with ChromaDB and Ollama
> - **[Read Part 2](https://medium.com/@froilan.sia/from-cli-to-gui-building-a-chat-interface-for-your-rag-pipeline-8b3e757bff5e)** → Gradio UI with real-time progress and source-aware responses

##  Features


**Part 1 (CLI)** 
* Fast text & table extraction using PyMuPDF 
* Selective vision processing with `qwen2.5vl:3b` Parent-Child retrieval with ChromaDB 
* CLI interface for ingestion and querying 

**Part 2 (UI)**  **All of the above, plus:** 
* Drag-and-drop PDF upload 
* Real-time progress bars 
* Chat interface Source citations with page numbers and sections 
* Database management (clear, refresh, status) 

## System Requirements

- Apple Silicon Mac (M1/M2/M3) 
- also works on Linux and Windows
- macOS 12 Monterey or later
- Python 3.9 or later
- [Ollama](https://ollama.com) 0.7.0 or later
- At least 8GB RAM (16GB+ recommended for larger models)

## Setup (Common to Both Parts)

### 1. Clone the Repository

```bash
git clone https://github.com/froilan-sia/m1_multimodal_rag.git
cd m1_multimodal_rag

### 2. Install Ollama & Pull Models
# Install Ollama via Homebrew
brew install ollama

# Verify version (requires >= 0.7.0 for qwen2.5vl models)
ollama --version

# Start the Ollama service (keep this running in a separate terminal tab)
ollama serve

# Pull the recommended models (For 8GB M1 Mac)
ollama pull qwen2.5vl:3b
ollama pull llama3.2:3b

# (For 16GB+ M1/M2/M3, optionally pull larger models)
# ollama pull llama3.2-vision:11b
# ollama pull qwen2.5:7b

# Verify models are installed
ollama list

### 3. Set Up Python Environment
# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import chromadb, fitz, ollama, PIL, gradio; print('✅ All dependencies ready!')"

### 4. Download a Sample PDF
# Download EY Good Group 2024 financial statements
curl -L -o data/sample_financials.pdf \
  "https://www.ey.com/en_pl/technical/ifrs-technical-resources/good-group-international-limited-december-2024"


# Part 1: Command-Line Interface (CLI)
These instructions are for Part 1 of the series. Read the full article here https://medium.com/@froilan.sia/zero-cost-multimodal-rag-chromadb-313930800633

## Ingest a PDF
python main.py ingest --pdf data/sample_financials.pdf

## Ingest a New PDF and Clear the Database
python main.py ingest --pdf data/new_report.pdf --clear

# Part 2: Gradio Web UI
# These instructions are for Part 2 of the series. Read the full article here. https://medium.com/@froilan.sia/from-cli-to-gui-building-a-chat-interface-for-your-rag-pipeline-8b3e757bff5e

## Launch the Gradio UI
python app.py

## Open your browser to http://127.0.0.1:7860.
## In the UI, you can:
## Upload and ingest PDFs with a real-time progress bar
## Chat with your documents
## View source citations with page numbers and sections
## Clear the database or chat history

# Project Structure


m1_multimodal_rag/
├── main.py               # CLI script (Part 1)
├── rag_engine.py         # Core engine with progress callbacks (Part 2)
├── app.py                # Gradio UI application (Part 2)
├── requirements.txt      # Python dependencies
├── README.md             # This is readme file
├── .gitignore            # Git ignore rules
├── data/                 # Place your PDFs here
│   └── .gitkeep
├── chroma_db/            # ChromaDB persistence (auto-created)
└── images_cache/         # Temporary cropped images (auto-created)






