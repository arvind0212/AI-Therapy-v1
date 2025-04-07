# src/infrastructure/embedding/sentence_transformer_embedder.py
"""
Concrete implementation of the EmbeddingModel interface using sentence-transformers.
"""

import logging
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

# Import interface and settings
from src.core.interfaces.embedding_model import EmbeddingModel
from src.core.config import settings

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedder(EmbeddingModel):
    """Uses sentence-transformers library to embed text."""

    def __init__(self, model_name: str = settings.embedding_model_name, device: str = 'cpu'):
        """
        Initializes the SentenceTransformerEmbedder.

        Args:
            model_name: The name of the sentence-transformer model to load
                        (e.g., 'all-MiniLM-L6-v2'). Defaults to the one in settings.
            device: The device to run the model on ('cpu', 'cuda', etc.).
        """
        self.model_name = model_name
        self.device = device
        logger.info(f"Initializing SentenceTransformerEmbedder with model: {self.model_name} on device: {self.device}")
        try:
            # Load the model from Hugging Face Hub or cache
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Get the embedding dimension from the loaded model
            self._dimension = self.model.get_sentence_embedding_dimension()
            if self._dimension != settings.embedding_dimension:
                 logger.warning(f"Model dimension {self._dimension} differs from settings dimension {settings.embedding_dimension}. Using model dimension.")
            logger.info(f"SentenceTransformer model '{self.model_name}' loaded successfully. Dimension: {self._dimension}")
        except Exception as e:
            logger.exception(f"Failed to load sentence-transformer model '{self.model_name}': {e}")
            raise # Re-raise the exception to indicate failure

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of text documents."""
        if not texts:
            logger.warning("embed_documents called with empty list.")
            return []
        logger.info(f"Embedding {len(texts)} documents...")
        try:
            # The encode method returns a numpy array or tensor. Convert to list of lists.
            embeddings_np = self.model.encode(texts, convert_to_numpy=True, device=self.device, show_progress_bar=True)
            embeddings_list = embeddings_np.tolist()
            logger.info(f"Successfully embedded {len(texts)} documents.")
            return embeddings_list
        except Exception as e:
            logger.exception(f"Error embedding documents with model {self.model_name}: {e}")
            return [] # Return empty list on failure

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query text."""
        if not text:
             logger.warning("embed_query called with empty text.")
             return []
        logger.info("Embedding query...")
        try:
            embedding_np = self.model.encode(text, convert_to_numpy=True, device=self.device)
            embedding_list = embedding_np.tolist()
            logger.info("Successfully embedded query.")
            return embedding_list
        except Exception as e:
            logger.exception(f"Error embedding query with model {self.model_name}: {e}")
            return [] # Return empty list on failure

    @property
    def dimension(self) -> int:
        """Returns the dimension of the embedding vectors."""
        return self._dimension
