# """PDF document loader."""

# from pathlib import Path
# import re

# from pypdf import PdfReader

# from .base_loader import BaseLoader


# class PDFLoader(BaseLoader):
#     """Loader for PDF documents."""

#     def load(self, file_path: str) -> str:
#         """Extract text from a PDF file."""
#         pages = self.load_pages(file_path)
#         return "\n\n".join(text for _, text in pages)

#     def load_pages(self, file_path: str) -> list[tuple[int, str]]:
#         """
#         Extract text per page for citation support.

#         Returns a list of (page_number, text) using simple 1-based page numbers.
#         """
#         path = Path(file_path)
#         if not path.exists():
#             raise FileNotFoundError(f"PDF file not found: {file_path}")

#         reader = PdfReader(file_path)
#         result: list[tuple[int, str]] = []

#         for idx, page in enumerate(reader.pages, start=1):
#             page_text = page.extract_text()
#             if not page_text:
#                 continue

#             # Normalize whitespace so words don't get jammed together.
#             # 1) Standardize newlines
#             page_text = page_text.replace("\r", "\n")
#             # 2) Convert multiple blank lines to double newlines (paragraph breaks)
#             page_text = re.sub(r"\n\s*\n+", "\n\n", page_text)
#             # 3) Within paragraphs, collapse internal newlines and excessive spaces to single spaces
#             paragraphs = page_text.split("\n\n")
#             normalized_paragraphs = [" ".join(p.split()) for p in paragraphs]
#             normalized_text = "\n\n".join(normalized_paragraphs)

#             # 4) Fix common PDF extraction spacing issues
#             # Fix lowercase-uppercase boundaries (e.g., "systemDesigned" → "system Designed")
#             normalized_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", normalized_text)
#             # Fix alphabetic-numeric boundaries
#             normalized_text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", normalized_text)
#             normalized_text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", normalized_text)
#             # Ensure punctuation spacing
#             normalized_text = re.sub(r"([,.;:!?])([A-Za-z0-9])", r"\1 \2", normalized_text)
#             # Fix bullet spacing
#             normalized_text = re.sub(r"•(\S)", r"• \1", normalized_text)
#             normalized_text = re.sub(r"(\S)•", r"\1 •", normalized_text)

#             # Use simple 1-based page numbers for reliability.
#             result.append((idx, normalized_text.strip()))
#         return result






# """PDF document loader."""

# from pathlib import Path
# import re
# from typing import List, Tuple

# from pypdf import PdfReader

# from .base_loader import BaseLoader


# class PDFLoader(BaseLoader):
#     """Loader for PDF documents with safe whitespace handling."""

#     def load(self, file_path: str) -> str:
#         """Extract full text from a PDF file."""
#         pages = self.load_pages(file_path)
#         return "\n\n".join(text for _, text in pages)

#     def load_pages(self, file_path: str) -> List[Tuple[int, str]]:
#         """
#         Extract text per page for citation support.

#         Returns:
#             List of (page_number, text) using simple 1-based page numbers.
#         """
#         path = Path(file_path)
#         if not path.exists():
#             raise FileNotFoundError(f"PDF file not found: {file_path}")

#         reader = PdfReader(file_path)
#         results: List[Tuple[int, str]] = []

#         for idx, page in enumerate(reader.pages, start=1):
#             raw_text = page.extract_text()

#             if not raw_text:
#                 continue

#             # ---- Minimal & Safe Normalization ----

#             # Normalize carriage returns
#             text = raw_text.replace("\r", "\n")

#             # Collapse excessive spaces (but keep real spacing)
#             text = re.sub(r"[ \t]+", " ", text)

#             # Normalize excessive blank lines (preserve paragraph breaks)
#             text = re.sub(r"\n{3,}", "\n\n", text)

#             # ---- Optional Light Spacing Repairs ----
#             # Only fix very common extraction boundary issues

#             # Fix lowercase-uppercase joins (e.g., systemDesign -> system Design)
#             text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

#             # Fix letter-number joins (e.g., page12 -> page 12)
#             text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
#             text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)

#             # Ensure punctuation is followed by space
#             text = re.sub(r"([,.;:!?])(?=[A-Za-z0-9])", r"\1 ", text)

#             # Fix bullet spacing
#             text = re.sub(r"•\s*", "• ", text)

#             cleaned_text = text.strip()

#             if cleaned_text:
#                 results.append((idx, cleaned_text))

#         return results

"""PDF document loader using pdfplumber for correct spacing reconstruction."""

from pathlib import Path
from typing import List, Tuple

import pdfplumber

from .base_loader import BaseLoader


class PDFLoader(BaseLoader):
    """
    Loader for PDF documents.

    Uses pdfplumber instead of pypdf because:
    - It reconstructs spaces using glyph positioning
    - It handles academic PDFs much better
    - It avoids word-concatenation issues
    """

    def load(self, file_path: str) -> str:
        """Extract full text from a PDF file."""
        pages = self.load_pages(file_path)
        return "\n\n".join(text for _, text in pages)

    def load_pages(self, file_path: str) -> List[Tuple[int, str]]:
        """
        Extract text per page for citation support.

        Returns:
            List of (page_number, text)
        """

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        results: List[Tuple[int, str]] = []

        with pdfplumber.open(file_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):

                # IMPORTANT:
                # x_tolerance helps determine when a space should exist
                # These values work well for most research PDFs
                text = page.extract_text(
                    x_tolerance=2,
                    y_tolerance=3,
                )

                if not text:
                    continue

                # Minimal safe cleanup ONLY
                text = text.replace("\r", "\n")

                # Normalize excessive blank lines
                while "\n\n\n" in text:
                    text = text.replace("\n\n\n", "\n\n")

                cleaned_text = text.strip()

                if cleaned_text:
                    results.append((idx, cleaned_text))

        return results
