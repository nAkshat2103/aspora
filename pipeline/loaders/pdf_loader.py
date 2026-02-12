

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
