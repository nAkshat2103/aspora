"""Loader factory that maps file extensions and URLs to loader classes."""

from pathlib import Path

from .loaders.base_loader import BaseLoader
from .loaders.markdown_loader import MarkdownLoader
from .loaders.pdf_loader import PDFLoader
from .loaders.text_loader import TextLoader
from .loaders.url_loader import URLLoader


def is_url(s: str) -> bool:
    """Check if string is a valid URL."""
    s = s.strip()
    return s.startswith(("http://", "https://"))


class LoaderFactory:
    """Maps file extensions to appropriate loader classes."""

    _EXTENSION_MAP: dict[str, type[BaseLoader]] = {
        ".pdf": PDFLoader,
        ".txt": TextLoader,
        ".text": TextLoader,
        ".md": MarkdownLoader,
        ".md": MarkdownLoader,
    }

    @classmethod
    def get_loader(cls, file_path: str) -> BaseLoader | None:
        """Get the appropriate loader for a file or URL.

        Args:
            file_path: Path to the file or URL.

        Returns:
            Loader instance or None if no loader is registered.
        """
        if is_url(file_path):
            return URLLoader()
        ext = Path(file_path).suffix.lower()
        loader_class = cls._EXTENSION_MAP.get(ext)
        if loader_class:
            return loader_class()
        return None

    @classmethod
    def register(cls, extension: str, loader_class: type[BaseLoader]) -> None:
        """Register a new loader for an extension. Enables extensibility for formats like DOCX."""
        cls._EXTENSION_MAP[extension.lower()] = loader_class
