# Production RAG System

A production-oriented Retrieval-Augmented Generation (RAG) system that ingests PDF documents, creates semantic embeddings, performs FAISS-based retrieval, and generates grounded answers using a local Ollama LLM.

## Architecture

PDF document
    ↓
PDF Loader
    ↓
Text Cleaning & Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Index
    ↓
Semantic Retrieval
    ↓
Minimum Score Filtering
    ↓
Context Construction
    ↓
Ollama LLM
    ↓
Citation-validated Answer

## Technology Stack

- Python 3.12.4
- Sentence Transformers
- BAAI/bge-small-en-v1.5
- FAISS
- Ollama
- llama3.1
- pytest
- python-dotenv

## Project Structure

```text
production-rag-system/
├── .env
├── .gitignore
├── evaluation_questions.json
├── evaluation_results.json
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── research_paper.pdf
│   └── processed/
│       ├── chunks.json
│       ├── embeddings.json
│       ├── evaluation_questions.json
│       ├── faiss.index
│       └── pages.json
├── src/
│   ├── build_index.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── generate_embeddings.py
│   ├── pdf_loader.py
│   ├── rag.py
│   └── retriever.py
└── tests/
    ├── evaluation.py
    ├── retrieval_evaluation.py
    ├── test_ollama.py
    └── test_rag.py