

from collections.abc import Iterator
from .knowledge_base import KnowledgeBase
from .llm_service import LLMService


class RAGPipeline:
    """
    Facade class for the RAG pipeline:
    - Ingest documents
    - Retrieve relevant chunks
    - Build structured context
    - Generate grounded answers
    """

    def __init__(self, knowledge_base: KnowledgeBase, llm_service: LLMService):
        self.knowledge_base = knowledge_base
        self.llm_service = llm_service

   

    def ingest(self, files: list[str]) -> list[str]:
        """Ingest documents into the knowledge base."""
        return self.knowledge_base.add_documents(files)

   
    def ask(
        self,
        question: str,
        k: int = 16,
        chat_history: list[dict] | None = None,
    ) -> str:
        """
        Answer a question using retrieved context.
        Retrieval is based ONLY on the current question.
        """

        results = self.knowledge_base.retrieve(question, k=k)

        if not results:
            return "No relevant documents found. Please ingest documents first."

        context = self._build_context(results[:k])

        return self.llm_service.generate(question, context, chat_history)

 
    def ask_stream(
        self,
        question: str,
        k: int = 16,
        chat_history: list[dict] | None = None,
    ) -> Iterator[str]:
        """
        Stream answer tokens as generated.
        Preserves all formatting and spacing exactly.
        """

        results = self.knowledge_base.retrieve(question, k=k)

        if not results:
            yield "No relevant documents found. Please ingest documents first."
            return

        context = self._build_context(results[:k])

        for token in self.llm_service.generate_stream(
            question, context, chat_history
        ):
            yield token

    def _build_context(self, results: list[dict]) -> str:
        """
        Build structured context blocks from retrieved results.
        No spacing normalization is applied.
        """

        context_blocks: list[str] = []

        for r in results:
            meta = r.get("metadata", {}) or {}

            document_name = (
                meta.get("document_name")
                or meta.get("file_name")
                or "Unknown document"
            )

            page_number = meta.get("page_number") or meta.get("page")
            page_start = meta.get("page_start")
            page_end = meta.get("page_end")

            if page_start is not None and page_end is not None:
                page_line = f"Pages: {page_start}-{page_end}"
            elif page_number is not None:
                page_line = f"Page: {page_number}"
            else:
                page_line = "Page: Unknown"

            # IMPORTANT: do not modify retrieved text
            text = r.get("document", "")

            block = (
                f"Document: {document_name}\n"
                f"{page_line}\n"
                f'Text: "{text}"'
            )

            context_blocks.append(block)

        return "\n\n".join(context_blocks)
