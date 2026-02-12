"""Base loader abstract class for document loaders."""

from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: str) -> str:
        """Load and return the text content of a document.

        Args:
            file_path: Path to the document file.

        Returns:
            Extracted text content as a string.
        """
        pass
