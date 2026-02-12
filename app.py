"""CLI for the AI Document Q&A (RAG) system."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from pipeline.chunker import Chunker
from pipeline.knowledge_base import KnowledgeBase
from pipeline.loader_factory import is_url
from pipeline.llm_service import LLMService
from pipeline.rag_pipeline import RAGPipeline
from pipeline.registry import DocumentRegistry
from pipeline.vector_store import VectorStoreManager

# Paths relative to project root (doc_qa/)
BASE_DIR = Path(__file__).resolve().parent

# Load .env from doc_qa folder (so it works regardless of cwd)
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
DB_PATH = str(DATA_DIR / "documents.db")


def _init_components():
    """Initialize pipeline components."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    llm = LLMService(api_key=api_key)
    vector_store = VectorStoreManager(persist_directory=str(VECTOR_DB_DIR))
    registry = DocumentRegistry(db_path=DB_PATH)
    chunker = Chunker()
    knowledge_base = KnowledgeBase(
        registry=registry,
        vector_store=vector_store,
        chunker=chunker,
    )
    pipeline = RAGPipeline(knowledge_base=knowledge_base, llm_service=llm)
    return pipeline


def cmd_ingest(pipeline: RAGPipeline, source: str) -> None:
    """Ingest a document (file path or URL)."""
    if not is_url(source):
        path = Path(source)
        if not path.exists():
            print(f"Error: File not found: {source}")
            sys.exit(1)
    try:
        doc_ids = pipeline.ingest([source])
        name = Path(source).name if not is_url(source) else source[:50]
        print(f"Ingested: {name} (id: {doc_ids[0]})")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during ingestion: {e}")
        sys.exit(1)


def cmd_ask(pipeline: RAGPipeline, question: str) -> None:
    """Answer a question."""
    try:
        answer = pipeline.ask(question)
        print(answer)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_list(pipeline: RAGPipeline) -> None:
    """List all documents."""
    docs = pipeline.knowledge_base.list_documents()
    if not docs:
        print("No documents in the knowledge base.")
        return
    for d in docs:
        print(f"  {d['id']}  {d['file_name']}")


def cmd_delete(pipeline: RAGPipeline, doc_id: str) -> None:
    """Delete a document."""
    removed = pipeline.knowledge_base.delete_document(doc_id)
    if removed:
        print(f"Deleted document: {doc_id}")
    else:
        print(f"Document not found: {doc_id}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python app.py ingest <file|url>")
        print("  python app.py ask \"<question>\"")
        print("  python app.py list")
        print("  python app.py delete <doc_id>")
        sys.exit(1)

    pipeline = _init_components()
    cmd = sys.argv[1].lower()

    if cmd == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python app.py ingest <file|url>")
            sys.exit(1)
        cmd_ingest(pipeline, sys.argv[2])
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print("Usage: python app.py ask \"<question>\"")
            sys.exit(1)
        cmd_ask(pipeline, sys.argv[2])
    elif cmd == "list":
        cmd_list(pipeline)
    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("Usage: python app.py delete <doc_id>")
            sys.exit(1)
        cmd_delete(pipeline, sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
