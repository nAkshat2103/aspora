"""Markdown document loader."""

from pathlib import Path

from .base_loader import BaseLoader


class MarkdownLoader(BaseLoader):
    """Loader for Markdown files."""

    def load(self, file_path: str) -> str:
        """Read content from a Markdown file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
