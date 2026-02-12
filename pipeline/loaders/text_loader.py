"""Plain text document loader."""

from pathlib import Path

from .base_loader import BaseLoader


class TextLoader(BaseLoader):
    """Loader for plain text files."""

    def load(self, file_path: str) -> str:
        """Read text from a plain text file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
