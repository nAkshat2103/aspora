# """RAG pipeline facade for ingestion and Q&A."""

# from collections.abc import Iterator
# import re

# from .knowledge_base import KnowledgeBase
# from .llm_service import LLMService


# def normalize_text_spacing(text: str) -> str:
#     """
#     Deterministic text normalization to fix spacing issues.

#     Handles:
#     - Concatenated words (lowercase-uppercase boundaries)
#     - Missing spaces around punctuation
#     - Alphabetic-numeric boundaries
#     - Multiple spaces collapse
#     """
#     if not text:
#         return text

#     # 1) Fix lowercase-uppercase boundaries (e.g., "systemDesigned" → "system Designed")
#     # But preserve common acronyms and proper nouns (e.g., "AI", "PDF", "BERT")
#     text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

#     # 2) Fix alphabetic-numeric boundaries (e.g., "system2024" → "system 2024")
#     text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
#     text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)

#     # 3) Ensure proper spacing around punctuation
#     text = re.sub(r"([,.;:!?])([A-Za-z0-9])", r"\1 \2", text)
#     text = re.sub(r"([A-Za-z0-9])([,.;:!?])", r"\1\2", text)

#     # 4) Fix spacing around bullets and special characters
#     text = re.sub(r"•(\S)", r"• \1", text)
#     text = re.sub(r"(\S)•", r"\1 •", text)

#     # 5) Collapse multiple spaces/tabs into single space
#     text = re.sub(r"[ \t]+", " ", text)

#     # 6) Preserve paragraph breaks (double newlines) but normalize single newlines
#     paragraphs = text.split("\n\n")
#     normalized_paragraphs = []
#     for para in paragraphs:
#         # Within a paragraph, collapse newlines to spaces
#         para = para.replace("\n", " ")
#         # Clean up any extra spaces
#         para = re.sub(r" +", " ", para).strip()
#         if para:
#             normalized_paragraphs.append(para)
#     text = "\n\n".join(normalized_paragraphs)

#     return text.strip()


# class RAGPipeline:
#     """Facade class for the RAG pipeline: ingest documents and answer questions."""

#     def __init__(self, knowledge_base: KnowledgeBase, llm_service: LLMService):
#         self.knowledge_base = knowledge_base
#         self.llm_service = llm_service

#     def ingest(self, files: list[str]) -> list[str]:
#         """Ingest documents into the knowledge base. Returns list of doc_ids."""
#         return self.knowledge_base.add_documents(files)

#     def ask(
#         self,
#         question: str,
#         k: int = 16,
#         chat_history: list[dict] | None = None,
#     ) -> str:
#         """Answer a question using retrieved context and optional chat history for multi-turn."""
#         # Per requirements, retrieval should be based ONLY on the current question.
#         results = self.knowledge_base.retrieve(question, k=k)
#         if not results:
#             return "No relevant documents found. Please ingest documents first."
#         # Build context from all retrieved chunks (up to k), including citation metadata.
#         context_blocks: list[str] = []
#         for r in results[:k]:
#             meta = (r.get("metadata") or {}) if isinstance(r, dict) else {}
#             document_name = (
#                 meta.get("document_name")
#                 or meta.get("file_name")
#                 or "Unknown document"
#             )

#             # Support both single page_number and potential page_start/page_end.
#             page_number = meta.get("page_number") or meta.get("page")
#             page_start = meta.get("page_start")
#             page_end = meta.get("page_end")

#             if page_start is not None and page_end is not None:
#                 page_line = f"Pages: {page_start}-{page_end}"
#             elif page_number is not None:
#                 page_line = f"Page: {page_number}"
#             else:
#                 page_line = "Page: Unknown"

#             text = r.get("document", "") if isinstance(r, dict) else ""
#             # Normalize retrieved chunk text BEFORE passing to LLM
#             text = normalize_text_spacing(text)
#             block = (
#                 f"Document: {document_name}\n"
#                 f"{page_line}\n"
#                 f'Text: "{text}"'
#             )
#             context_blocks.append(block)

#         context = "\n\n".join(context_blocks)
#         answer = self.llm_service.generate(question, context, chat_history)
#         # Normalize generated answer before returning
#         return normalize_text_spacing(answer)

#     def ask_stream(
#         self,
#         question: str,
#         k: int = 16,
#         chat_history: list[dict] | None = None,
#     ) -> Iterator[str]:
#         """Stream answer tokens as they are generated."""
#         # Per requirements, retrieval should be based ONLY on the current question.
#         results = self.knowledge_base.retrieve(question, k=k)
#         if not results:
#             yield "No relevant documents found. Please ingest documents first."
#             return
#         # Build context from all retrieved chunks (up to k), including citation metadata.
#         context_blocks: list[str] = []
#         for r in results[:k]:
#             meta = (r.get("metadata") or {}) if isinstance(r, dict) else {}
#             document_name = (
#                 meta.get("document_name")
#                 or meta.get("file_name")
#                 or "Unknown document"
#             )

#             page_number = meta.get("page_number") or meta.get("page")
#             page_start = meta.get("page_start")
#             page_end = meta.get("page_end")

#             if page_start is not None and page_end is not None:
#                 page_line = f"Pages: {page_start}-{page_end}"
#             elif page_number is not None:
#                 page_line = f"Page: {page_number}"
#             else:
#                 page_line = "Page: Unknown"

#             text = r.get("document", "") if isinstance(r, dict) else ""
#             # Normalize retrieved chunk text BEFORE passing to LLM
#             text = normalize_text_spacing(text)
#             block = (
#                 f"Document: {document_name}\n"
#                 f"{page_line}\n"
#                 f'Text: "{text}"'
#             )
#             context_blocks.append(block)

#         context = "\n\n".join(context_blocks)
#         # Collect streamed tokens, normalize, then yield
#         accumulated = ""
#         for token in self.llm_service.generate_stream(
#             question, context, chat_history
#         ):
#             accumulated += token
#         # Normalize the complete answer before yielding
#         normalized = normalize_text_spacing(accumulated)
#         yield normalized




"""RAG pipeline facade for ingestion and Q&A."""

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

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, files: list[str]) -> list[str]:
        """Ingest documents into the knowledge base."""
        return self.knowledge_base.add_documents(files)

    # ------------------------------------------------------------------
    # Non-streaming Q&A
    # ------------------------------------------------------------------

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

        # IMPORTANT:
        # Do NOT normalize or mutate retrieved text.
        # Pass raw retrieved chunks directly to LLM.
        return self.llm_service.generate(question, context, chat_history)

    # ------------------------------------------------------------------
    # Streaming Q&A
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
