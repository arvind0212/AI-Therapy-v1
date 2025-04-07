# src/core/interfaces/vector_store.py
"""
Abstract Base Class for vector store components.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

# Import the DocumentChunk model
from src.core.models.document import DocumentChunk

class VectorStore(ABC):
    """Interface for storing and querying document chunks and their embeddings."""

    @abstractmethod
    def add(self, chunks: List[DocumentChunk], **kwargs: Any) -> List[str]:
        """
        Adds document chunks (including their embeddings) to the store.

        Args:
            chunks: A list of DocumentChunk objects to add.
                    Assumes embeddings are already populated if needed by the store.
            **kwargs: Additional provider-specific parameters.

        Returns:
            A list of IDs for the added chunks.
        """
        pass

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[DocumentChunk]:
        """
        Queries the vector store for chunks similar to the query embedding.

        Args:
            query_embedding: The embedding vector of the query.
            top_k: The number of most similar chunks to retrieve.
            filters: Optional dictionary for metadata filtering (structure depends on implementation).
                     Example: {'source': 'some_document.pdf'}
            **kwargs: Additional provider-specific parameters.

        Returns:
            A list of the top_k most similar DocumentChunk objects, potentially including
            similarity scores depending on the implementation (could be added to metadata).
            Chunks returned should ideally include their content and metadata.
        """
        pass

    # Optional: Add methods for deleting, updating, or ensuring collection/table exists
    # @abstractmethod
    # def delete(self, chunk_ids: List[str], **kwargs: Any) -> bool:
    #     """Deletes chunks by their IDs."""
    #     pass

    # @abstractmethod
    # def ensure_collection(self, **kwargs: Any):
    #     """Ensures the necessary collection/table exists in the vector store."""
    #     pass
