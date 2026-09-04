# Production RAG System



A production-oriented Retrieval-Augmented Generation (RAG) system that ingests PDF documents, performs semantic search using vector embeddings and FAISS, and generates grounded, citation-backed answers using a local Ollama LLM.

The system is designed with reliability, citation validation, relevance filtering, testing, and CI automation in mind.

## Features

* PDF document ingestion
* Text extraction, cleaning, and chunking
* Semantic embeddings using `BAAI/bge-small-en-v1.5`
* FAISS vector similarity search
* Minimum relevance-score filtering
* Grounded answer generation using local Ollama
* Citation-aware responses
* Citation validation
* Conversation/history support
* Logging and error handling
* Evaluation framework
* Automated pytest test suite
* GitHub Actions CI
* Safe handling of unanswerable questions
* No external OpenAI API dependency

## Architecture

```text
PDF Document
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
Citation Validation
     ↓
Grounded Answer
```

## Technology Stack

| Technology             | Purpose                   |
| ---------------------- | ------------------------- |
| Python 3.12            | Application development   |
| Sentence Transformers  | Semantic embeddings       |
| BAAI/bge-small-en-v1.5 | Embedding model           |
| FAISS                  | Vector similarity search  |
| Ollama                 | Local LLM inference       |
| llama3.1               | Generation model          |
| pytest                 | Automated testing         |
| python-dotenv          | Environment configuration |
| GitHub Actions         | Continuous integration    |

## Retrieval Pipeline

The system converts the source PDF into searchable semantic chunks.

1. The PDF is loaded and converted into text.
2. Text is cleaned and divided into manageable chunks.
3. Each chunk is converted into a 384-dimensional embedding.
4. Embeddings are stored in a FAISS index.
5. A user question is embedded using the same model.
6. FAISS retrieves the most semantically relevant chunks.
7. Results below the configured relevance threshold are filtered.
8. The remaining context is passed to the local Ollama model.
9. The generated response is validated for document citations.

## Reliability Features

### Minimum Score Filtering

The retriever uses a configurable minimum similarity score to avoid returning irrelevant documents.

The current production configuration uses:

```text
RAG_MIN_SCORE=0.60
```

This helps the system distinguish between questions that can be answered from the indexed documents and questions that are outside the available knowledge base.

### Unanswerable Questions

When relevant information cannot be found in the provided documents, the system avoids inventing an answer and returns a safe response indicating that the information was not found in the documents.

### Citation Validation

Generated answers are checked to ensure that citations follow the expected document-reference format.

### Local LLM

The system uses Ollama locally rather than requiring an external OpenAI API.

Default configuration:

```text
OLLAMA_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
```

## Project Structure

```text
production-rag-system/
├── .env
├── .gitignore
├── README.md
├── evaluation_questions.json
├── evaluation_results.json
├── requirements.txt
├── pytest.ini
├── .github/
│   └── workflows/
│       └── tests.yml
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
    ├── ollama_check.py
    ├── retrieval_evaluation.py
    └── test_rag.py
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Preethi3026/production-rag-system.git
cd production-rag-system
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```text
OLLAMA_MODEL=llama3.1
RAG_MIN_SCORE=0.60
OLLAMA_BASE_URL=http://localhost:11434
```

### 5. Install and start Ollama

Make sure Ollama is installed and the required model is available locally.

Verify the model:

```powershell
ollama list
```

If `llama3.1` is not available:

```powershell
ollama pull llama3.1
```

## Running the Application

Start the RAG CLI with:

```powershell
python -m src.rag
```

The application will load the embedding model, FAISS index, document records, and Ollama configuration.

You can then enter questions interactively.

Example:

```text
Ask a question (or type 'exit'): What is Microsoft Azure?
```

The system retrieves relevant document context and generates a citation-backed answer.

To exit:

```text
exit
```

## Testing

Run the complete local test suite:

```powershell
python -m pytest -q
```

The Ollama-dependent tests require a running local Ollama server.

To run the tests that do not require Ollama:

```powershell
python -m pytest -q -m "not ollama"
```

The Ollama-specific local check is available separately as:

```text
tests/ollama_check.py
```

## Continuous Integration

The repository includes a GitHub Actions workflow:

```text
.github/workflows/tests.yml
```

Every push to `main` and pull request targeting `main` automatically runs the CI-safe pytest suite.

The CI environment explicitly uses:

```text
RAG_MIN_SCORE=0.60
```

Ollama-dependent tests are excluded from CI because the application uses a local Ollama service.

## Evaluation

The project includes an evaluation framework for testing retrieval and answer behavior.

Evaluation questions are stored in:

```text
evaluation_questions.json
```

and processed evaluation results are stored in:

```text
evaluation_results.json
```

The evaluation includes both answerable and unanswerable questions to verify that the system retrieves relevant context and avoids unsupported answers.

## Example Behaviors

### Answerable Question

```text
Question:
What percentage of Fortune 500 companies use Azure?

Response:
According to [Document 1], 95% of Fortune 500 companies use Azure.
```

### Unanswerable Question

```text
Question:
What is the capital of Japan?

Response:
I couldn't find the answer in the provided documents.
```

The second behavior demonstrates the system's document-grounding mechanism rather than relying on the model's general world knowledge.

## Production-Oriented Design

The project focuses on several principles important for production RAG systems:

* Deterministic document preprocessing
* Reusable vector index
* Semantic retrieval
* Configurable relevance filtering
* Grounded generation
* Citation validation
* Explicit handling of unanswerable queries
* Environment-based configuration
* Logging and error handling
* Automated testing
* Continuous integration

## License

This project is intended for educational, portfolio, and demonstration purposes.
