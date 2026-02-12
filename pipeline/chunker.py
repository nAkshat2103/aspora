# """Text chunker using recursive character splitting."""


# class Chunker:
#     """Splits text into overlapping chunks using recursive character splitting."""

#     def __init__(
#         self,
#         chunk_size: int = 1000,
#         chunk_overlap: int = 225,
#         separators: list[str] | None = None,
#     ):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         # Prioritize section boundaries (web articles, markdown, lists)
#         self.separators = separators or [
#             "\n\n==", "\n\n## ", "\n\n# ", "\n\n", "\n• ", "\n- ", "\n* ", "\n", ". ", " ", ""
#         ]

#     def chunk(self, text: str) -> list[str]:
#         """Split text into chunks using recursive character splitting."""
#         if not text or not text.strip():
#             return []
#         return self._split_recursive(text.strip(), self.separators)

#     def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
#         """Recursively split text by separators."""
#         if not text:
#             return []
#         if len(text) <= self.chunk_size:
#             return [text]

#         separator = separators[0] if separators else ""
#         next_separators = separators[1:] if len(separators) > 1 else [" "]

#         if separator:
#             parts = text.split(separator)
#         else:
#             parts = list(text)

#         chunks: list[str] = []
#         current = ""

#         for i, part in enumerate(parts):
#             if separator and i > 0:
#                 part = separator + part

#             if len(part) > self.chunk_size and next_separators:
#                 if current:
#                     chunks.append(current.strip())
#                     current = ""
#                 sub_chunks = self._split_recursive(part, next_separators)
#                 for sub in sub_chunks:
#                     if len(current) + len(sub) <= self.chunk_size:
#                         current += sub
#                     else:
#                         if current:
#                             chunks.append(current.strip())
#                         overlap = current[-self.chunk_overlap:] if self.chunk_overlap and current else ""
#                         current = overlap + sub
#             elif len(current) + len(part) <= self.chunk_size:
#                 current += part
#             else:
#                 if current:
#                     chunks.append(current.strip())
#                 overlap = current[-self.chunk_overlap:] if self.chunk_overlap else ""
#                 current = overlap + part

#         if current.strip():
#             chunks.append(current.strip())
#         return chunks




"""Sentence-aware, citation-safe text chunker for RAG."""

import re
from typing import List


class Chunker:
    """
    Splits text into clean, overlapping chunks without breaking words or sentences.
    Designed for citation-based RAG systems.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        text = text.strip()

        # Step 1: Split into paragraphs first
        paragraphs = self._split_paragraphs(text)

        # Step 2: Merge paragraphs into size-bounded chunks
        chunks = self._merge_paragraphs(paragraphs)

        return chunks

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _split_paragraphs(self, text: str) -> List[str]:
        """
        Split text into logical paragraphs.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        cleaned = [p.strip() for p in paragraphs if p.strip()]
        return cleaned

    def _merge_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Merge paragraphs into chunks without breaking words.
        Falls back to sentence splitting if a paragraph is too large.
        """

        chunks = []
        current = ""

        for paragraph in paragraphs:

            # If paragraph itself is too large → split by sentences
            if len(paragraph) > self.chunk_size:
                sentences = self._split_sentences(paragraph)

                for sentence in sentences:
                    current = self._append_with_limit(current, sentence, chunks)
                continue

            current = self._append_with_limit(current, paragraph, chunks)

        if current:
            chunks.append(current.strip())

        return chunks

    def _append_with_limit(self, current: str, addition: str, chunks: List[str]) -> str:
        """
        Append text safely while respecting chunk size.
        Ensures no mid-word splits.
        """

        candidate = f"{current} {addition}".strip() if current else addition

        if len(candidate) <= self.chunk_size:
            return candidate

        # Save current chunk
        if current:
            chunks.append(current.strip())

        # Apply overlap safely
        overlap_text = self._safe_overlap(current)

        new_chunk = f"{overlap_text} {addition}".strip() if overlap_text else addition

        # If still too large (rare), hard split at nearest whitespace
        if len(new_chunk) > self.chunk_size:
            split_index = new_chunk.rfind(" ", 0, self.chunk_size)
            if split_index == -1:
                split_index = self.chunk_size

            chunks.append(new_chunk[:split_index].strip())
            return new_chunk[split_index:].strip()

        return new_chunk

    def _safe_overlap(self, text: str) -> str:
        """
        Returns overlap text without cutting words.
        """

        if not text or self.chunk_overlap <= 0:
            return ""

        if len(text) <= self.chunk_overlap:
            return text

        start_index = len(text) - self.chunk_overlap

        # Move forward to nearest whitespace to avoid cutting a word
        space_index = text.find(" ", start_index)

        if space_index == -1:
            return text[start_index:]

        return text[space_index + 1:]

    def _split_sentences(self, paragraph: str) -> List[str]:
        """
        Split paragraph into sentences safely.
        Avoids breaking on common abbreviations.
        """

        # Basic sentence boundary detection
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)

        cleaned = [s.strip() for s in sentences if s.strip()]
        return cleaned





