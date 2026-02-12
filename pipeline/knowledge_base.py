"""Knowledge base for document storage and retrieval."""

from pathlib import Path

from .chunker import Chunker
from .loader_factory import LoaderFactory, is_url
from .loaders.url_loader import URLLoader
from .registry import DocumentRegistry
from .vector_store import VectorStoreManager


class KnowledgeBase:
    """Manages document ingestion, storage, and retrieval."""

    def __init__(
        self,
        registry: DocumentRegistry,
        vector_store: VectorStoreManager,
        chunker: Chunker | None = None,
    ):
        self.registry = registry
        self.vector_store = vector_store
        self.chunker = chunker or Chunker()

    def add_documents(self, files: list[str]) -> list[str]:
        """Add documents (files or URLs) to the knowledge base. Returns list of doc_ids."""
        doc_ids = []
        for item in files:
            if is_url(item):
                doc_ids.extend(self._add_url(item))
            else:
                doc_ids.append(self._add_file(item))
        return doc_ids

    def _add_file(self, file_path: str) -> str:
        """Add a local file to the knowledge base."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        loader = LoaderFactory.get_loader(file_path)
        if not loader:
            raise ValueError(f"No loader registered for: {path.suffix}")

        doc_id = self.registry.register(file_path, path.name)
        # Store human-readable document name for citations.
        base_meta = {
            "doc_id": doc_id,
            "file_name": path.name,
            "document_name": path.name,
        }

        if hasattr(loader, "load_pages"):
            all_chunks: list[str] = []
            all_metadatas: list[dict] = []
            for page_label, page_text in loader.load_pages(file_path):
                page_chunks = self.chunker.chunk(page_text)
                for chunk in page_chunks:
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        **base_meta,
                        "chunk_index": len(all_chunks) - 1,
                        # Keep both page and page_number for compatibility.
                        "page": int(page_label),
                        "page_number": int(page_label),
                    })
            if all_chunks:
                self.vector_store.add(all_chunks, all_metadatas)
        else:
            text = loader.load(file_path)
            chunks = self.chunker.chunk(text)
            # Treat non-paged documents as single-page for citation purposes.
            metadatas = [
                {
                    **base_meta,
                    "chunk_index": i,
                    "page_number": 1,
                }
                for i in range(len(chunks))
            ]
            self.vector_store.add(chunks, metadatas)

        return doc_id

    def _add_url(self, url: str) -> list[str]:
        """Add a URL (web page) to the knowledge base. Returns [doc_id]."""
        loader = LoaderFactory.get_loader(url)
        if not loader or not isinstance(loader, URLLoader):
            raise ValueError("URL loader not available")

        display_name = URLLoader.get_display_name(url)
        doc_id = self.registry.register(url, display_name)
        base_meta = {
            "doc_id": doc_id,
            "file_name": display_name,
            "url": url,
            "document_name": display_name,
        }

        text = loader.load(url)
        chunks = self.chunker.chunk(text)
        # Web pages are treated as single-page for citation purposes.
        metadatas = [
            {
                **base_meta,
                "chunk_index": i,
                "page_number": 1,
            }
            for i in range(len(chunks))
        ]
        self.vector_store.add(chunks, metadatas)
        return [doc_id]

    def retrieve(self, query: str, k: int = 16) -> list[dict]:
        """Retrieve relevant chunks from the vector store, including neighbors for context."""
        base_results = self.vector_store.search(query, k=k)
        if not base_results:
            return []

        expanded: list[dict] = []
        seen_keys: set[tuple[str, int | None]] = set()

        for r in base_results:
            meta = r.get("metadata", {}) or {}
            doc_id = meta.get("doc_id")
            chunk_index = meta.get("chunk_index")

            # Use (doc_id, chunk_index) when available; otherwise fall back to document text.
            key: tuple[str, int | None]
            if doc_id is not None and chunk_index is not None:
                key = (str(doc_id), int(chunk_index))
            else:
                key = (r["document"], None)

            if key not in seen_keys:
                seen_keys.add(key)
                expanded.append(r)

            # If we don't have proper metadata, we can't find neighbors.
            if doc_id is None or chunk_index is None:
                continue

            neighbor_indices = [
                chunk_index - 2,
                chunk_index - 1,
                chunk_index + 1,
                chunk_index + 2,
            ]
            neighbors = self.vector_store.get_chunks_by_indices(str(doc_id), neighbor_indices)

            for n in neighbors:
                n_meta = n.get("metadata", {}) or {}
                n_chunk_index = n_meta.get("chunk_index")
                n_key: tuple[str, int | None]
                if n_chunk_index is not None:
                    n_key = (str(doc_id), int(n_chunk_index))
                else:
                    n_key = (n["document"], None)

                if n_key not in seen_keys:
                    seen_keys.add(n_key)
                    expanded.append(n)

        return expanded

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base."""
        removed = self.registry.remove(doc_id)
        if removed:
            self.vector_store.delete(doc_id)
        return removed

    def list_documents(self) -> list[dict]:
        """List all documents in the knowledge base."""
        return self.registry.list_docs()
