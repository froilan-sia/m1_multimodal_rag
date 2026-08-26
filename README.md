# Multimodal RAG on Apple Silicon (M1)

A privacy-first, zero-cost search assistant that runs locally on Apple Mac M1/M2/M3.

## 📋 Features
- Fast text & table extraction using PyMuPDF
- Selective vision processing using Ollama with qwen2.5vl:3b
- Parent-Child retrieval for accurate context-aware answers
- 100% local & private — no API calls, no data leaving your machine
- Optimized for Apple Silicon with Metal GPU acceleration

## 🚀 Quick Start

### 1. Clone the Repository
\`\`\`bash
git clone https://github.com/YOUR_USERNAME/m1_multimodal_rag.git
cd m1_multimodal_rag
\`\`\`

### 2. Install Ollama & Pull Models
\`\`\`bash
brew install ollama
ollama serve
ollama pull qwen2.5vl:3b
ollama pull llama3.2:3b
\`\`\`

### 3. Set Up Python Environment
\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

### 4. Download a Sample PDF
\`\`\`bash
curl -L -o data/sample_financials.pdf \
  "https://www.ey.com/en_pl/technical/ifrs-technical-resources/good-group-international-limited-december-2024"
\`\`\`

### 5. Ingest the PDF
\`\`\`bash
python main.py ingest --pdf data/sample_financials.pdf
\`\`\`

### 6. Ask a Question
\`\`\`bash
python main.py query --question "What was the total revenue shown in the financial statements?"
\`\`\`

## 📂 Project Structure
\`\`\`
m1_multimodal_rag/
├── main.py               # Unified CLI script
├── requirements.txt      # Python dependencies
├── README.md
├── data/                 # Place your PDFs here
└── chroma_db/            # ChromaDB persistence (auto-created)
\`\`\`

## 📝 License
MIT License
