# src/core/interfaces/data_loader.py
"""
Abstract Base Class for data loading components.
"""

from abc import ABC, abstractmethod
from typing import List, Any
from pathlib import Path

# Import the Document model we defined
from src.core.models.document import Document

class DataLoader(ABC):
    """Interface for loading documents from various sources."""

    @abstractmethod
    def load(self, source: Any) -> List[Document]:
        """
        Loads documents from a given source.

        Args:
            source: The data source (e.g., file path, directory path, URL).
                    The specific type depends on the implementation.

        Returns:
            A list of Document objects.
        """
        pass

    # Optional: Add a method to check if the loader supports a given source type
    # @abstractmethod
    # def supports(self, source: Any) -> bool:
    #     """Checks if this loader can handle the given source."""
    #     pass
