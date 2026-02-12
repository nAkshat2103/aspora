"""Document registry using SQLite for metadata storage."""

import os
import sqlite3
import uuid
from pathlib import Path
from urllib.parse import urlparse


def _display_name_for_source(source: str) -> str:
    """Derive display name from file path or URL."""
    source = source.strip()
    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        domain = parsed.netloc or "page"
        path = (parsed.path or "").strip("/")
        if path:
            name = path.split("/")[-1][:30] or domain
        else:
            name = domain.replace("www.", "")
        return name[:50]
    return os.path.basename(source)


class DocumentRegistry:
    """SQLite-backed registry for document metadata."""

    def __init__(self, db_path: str = "data/documents.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the documents table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def register(self, source: str, display_name: str | None = None) -> str:
        """Register a document (file path or URL) and return its unique ID."""
        doc_id = str(uuid.uuid4())
        file_name = display_name or _display_name_for_source(source)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO documents (id, file_path, file_name) VALUES (?, ?, ?)",
                (doc_id, source, file_name),
            )
            conn.commit()
        return doc_id

    def list_docs(self) -> list[dict]:
        """List all registered documents."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT id, file_path, file_name, created_at FROM documents ORDER BY created_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]

    def remove(self, doc_id: str) -> bool:
        """Remove a document from the registry. Returns True if found and removed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get(self, doc_id: str) -> dict | None:
        """Get a document by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
