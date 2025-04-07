# src/core/interfaces/embedding_model.py
"""
Abstract Base Class for text embedding model components.
"""

from abc import ABC, abstractmethod
from typing import List

class EmbeddingModel(ABC):
    """Interface for embedding text into vectors."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text documents.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors (list of floats), one for each input text.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query text.

        Often uses the same underlying model as embed_documents, but
        some models might have different modes for queries vs. documents.

        Args:
            text: The query string to embed.

        Returns:
            The embedding vector (list of floats) for the query.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Returns the dimension of the embedding vectors produced by this model.
        """
        pass
