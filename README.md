# AI Document Q&A (RAG) System

A modular Retrieval-Augmented Generation (RAG) system for document question answering using Google Gemini and local ChromaDB.

## Architecture Overview

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   app.py    │────▶│   RAGPipeline   │────▶│  KnowledgeBase   │
│   (CLI)     │     │   (facade)      │     │  + VectorStore   │
└─────────────┘     └────────┬────────┘     └──────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌────────────┐
       │LLMService│   │ Chunker  │   │LoaderFactory│
       │ (Gemini) │   │          │   │ → PDF/TXT  │
       └──────────┘   └──────────┘   └────────────┘
```

- **RAGPipeline**: Facade for `ingest()` and `ask()` operations.
- **KnowledgeBase**: Manages documents via loaders, chunker, vector store, and SQLite registry.
- **VectorStoreManager**: ChromaDB with persistent storage (`./vector_db`).
- **DocumentRegistry**: SQLite metadata store (`data/documents.db`).
- **LoaderFactory**: Extensible mapping of file extensions and URLs to loaders (PDF, TXT, MD, URL).

## Setup

### 1. Create virtual environment (recommended)

```bash
cd doc_qa
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # Linux/macOS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
```

Edit `.env` and add your Google Gemini API key:

```
GEMINI_API_KEY=your_actual_key_here
```

Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## How to Run

### Streamlit UI (recommended)

```bash
cd doc_qa
streamlit run streamlit_app.py
```

- Upload PDF, TXT, or MD files, or paste a URL for external web pages, then click **Ingest**
- Ask questions in the chat; responses stream in real time with source citations
- Manage documents (view, delete) from the sidebar

### CLI

Run from the `doc_qa` directory:

### Ingest a document or URL

```bash
python app.py ingest path/to/document.pdf
python app.py ingest notes.txt
python app.py ingest https://example.com/article
```

### Ask a question

```bash
python app.py ask "What is the main topic of the document?"
```

### List documents

```bash
python app.py list
```

### Delete a document

```bash
python app.py delete <doc_id>
```

Use the ID shown by `list` when deleting.

## Extensibility

To add support for new formats (e.g. DOCX), register a loader:

```python
from pipeline.loader_factory import LoaderFactory
from my_loaders import DOCXLoader

LoaderFactory.register(".docx", DOCXLoader)
```

Then implement a loader that extends `BaseLoader` and implements `load(file_path) -> str`.

## Project Structure

```
doc_qa/
├── app.py                 # CLI entry point
├── streamlit_app.py       # Streamlit UI (file upload, streaming chat)
├── requirements.txt
├── .env.example
├── data/
│   └── documents.db       # SQLite metadata (created on first run)
├── vector_db/             # ChromaDB persistence (created on first run)
├── pipeline/
│   ├── rag_pipeline.py    # RAG facade
│   ├── knowledge_base.py
│   ├── vector_store.py
│   ├── registry.py
│   ├── chunker.py
│   ├── loader_factory.py
│   ├── llm_service.py
│   └── loaders/
│       ├── base_loader.py
│       ├── pdf_loader.py
│       ├── text_loader.py
│       ├── markdown_loader.py
│       └── url_loader.py   # Web pages as external knowledge
└── README.md
```
