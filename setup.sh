#!/bin/bash

# --------------------------------------------------------------------
# setup.sh - Complete Git Repository Setup for M1 Multimodal RAG
# --------------------------------------------------------------------

echo "🚀 Setting up Git repository for m1_multimodal_rag..."

# 1. Clean up unnecessary files
echo "🧹 Cleaning up..."
rm -f main.py.bak

# 2. Create .gitignore
cat > .gitignore << 'GITIGNORE'
# Python virtual environment
venv/
env/
.env/
.venv/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.coverage
htmlcov/

# Database and cache
chroma_db/
images_cache/
*.db
*.sqlite
*.sqlite3

# Data files (keep the folder but exclude contents)
data/*.pdf
data/*.pdf.*
!data/.gitkeep

# IDE files
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Logs
*.log
logs/

# Distribution
dist/
build/
*.egg-info/
GITIGNORE

# 3. Create .gitkeep for data folder
touch data/.gitkeep

# 4. Create README.md
cat > README.md << 'README'
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
README

# 5. Create LICENSE
cat > LICENSE << 'LICENSE'
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
LICENSE

# 6. Initialize Git
echo "📦 Initializing Git..."
git init
git add .
git commit -m "Initial commit: Multimodal RAG on Apple Silicon M1"

# 7. Rename branch to main
git branch -M main

# 8. Ask for GitHub username
echo ""
echo "✅ Local Git repository is ready!"
echo ""
read -p "👉 Enter your GitHub username: " github_user

if [ -z "$github_user" ]; then
    echo "⚠️ No username entered. Skipping remote setup."
    echo "Run these commands manually:"
    echo "  git remote add origin https://github.com/YOUR_USERNAME/m1_multimodal_rag.git"
    echo "  git push -u origin main"
else
    repo_name="m1_multimodal_rag"
    echo ""
    echo "🔗 Adding remote: https://github.com/$github_user/$repo_name.git"
    git remote add origin "https://github.com/$github_user/$repo_name.git"
    
    echo ""
    echo "🚀 Pushing to GitHub..."
    echo "⚠️ If this fails, make sure you created the repository on GitHub first!"
    echo "   Go to: https://github.com/new"
    echo "   Repository name: $repo_name"
    echo "   Do NOT initialize with README, .gitignore, or license."
    echo ""
    read -p "Press Enter to push, or Ctrl+C to cancel..."
    
    git push -u origin main
    
    echo ""
    echo "✅ Done! Your repository is live at:"
    echo "   https://github.com/$github_user/$repo_name"
fi

echo ""
echo "🎉 Setup complete!"
