# src/services/ingestion_service.py
"""
Service layer for orchestrating the document ingestion pipeline.
"""

import logging
from typing import List, Optional, Union
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import interfaces and models
from src.core.interfaces.data_loader import DataLoader
from src.core.interfaces.embedding_model import EmbeddingModel
from src.core.interfaces.vector_store import VectorStore
from src.core.models.document import Document, DocumentChunk

logger = logging.getLogger(__name__)

class IngestionService:
    """Orchestrates loading, chunking, embedding, and storing documents."""

    def __init__(
        self,
        data_loader: DataLoader,
        text_splitter: RecursiveCharacterTextSplitter, # Inject a specific splitter
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        batch_size: int = 100 # Batch size for embedding and adding to vector store
    ):
        """
        Initializes the IngestionService.

        Args:
            data_loader: An instance of a DataLoader implementation.
            text_splitter: An instance of a text splitter (e.g., RecursiveCharacterTextSplitter).
            embedding_model: An instance of an EmbeddingModel implementation.
            vector_store: An instance of a VectorStore implementation.
            batch_size: Number of chunks to process in each batch.
        """
        self.data_loader = data_loader
        self.text_splitter = text_splitter
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.batch_size = batch_size
        logger.info("IngestionService initialized.")

    def run_ingestion(self, source: Union[str, Path]) -> bool:
        """
        Runs the end-to-end ingestion pipeline for a given source.

        Args:
            source: The source to load data from (file or directory path).

        Returns:
            True if ingestion completed successfully for at least one chunk, False otherwise.
        """
        logger.info(f"Starting ingestion process for source: {source}")
        total_chunks_added = 0

        try:
            # 1. Load Documents
            logger.info("Step 1: Loading documents...")
            documents = self.data_loader.load(source)
            if not documents:
                logger.warning("No documents loaded. Ingestion finished.")
                return False
            logger.info(f"Loaded {len(documents)} documents.")

            # 2. Chunk Documents
            logger.info("Step 2: Chunking documents...")
            all_chunks: List[DocumentChunk] = []
            for doc in documents:
                texts = self.text_splitter.split_text(doc.content)
                logger.debug(f"Split document {doc.id} (source: {doc.metadata.get('source', 'N/A')}) into {len(texts)} chunks.")
                for i, text_chunk in enumerate(texts):
                    chunk_metadata = doc.metadata.copy() # Start with original doc metadata
                    # Add chunk-specific metadata if desired (e.g., chunk index)
                    chunk_metadata["chunk_index"] = i
                    all_chunks.append(DocumentChunk(
                        document_id=doc.id,
                        content=text_chunk,
                        metadata=chunk_metadata
                        # Embedding will be added later
                    ))
            if not all_chunks:
                logger.warning("No chunks created from documents. Ingestion finished.")
                return False
            logger.info(f"Created a total of {len(all_chunks)} chunks.")

            # 3. Embed and Store Chunks (in batches)
            logger.info(f"Step 3: Embedding and storing chunks in batches of {self.batch_size}...")
            for i in range(0, len(all_chunks), self.batch_size):
                batch_chunks = all_chunks[i : i + self.batch_size]
                logger.info(f"Processing batch {i // self.batch_size + 1} ({len(batch_chunks)} chunks)...")

                # Embed batch
                texts_to_embed = [chunk.content for chunk in batch_chunks]
                embeddings = self.embedding_model.embed_documents(texts_to_embed)

                if not embeddings or len(embeddings) != len(batch_chunks):
                    logger.error(f"Embedding failed or returned incorrect number of vectors for batch {i // self.batch_size + 1}. Skipping batch.")
                    continue # Skip this batch

                # Add embeddings to chunks
                for chunk, embedding in zip(batch_chunks, embeddings):
                    chunk.embedding = embedding

                # Add batch to vector store
                added_ids = self.vector_store.add(batch_chunks)
                if added_ids:
                    logger.info(f"Successfully added {len(added_ids)} chunks to vector store for batch {i // self.batch_size + 1}.")
                    total_chunks_added += len(added_ids)
                else:
                     logger.error(f"Failed to add chunks to vector store for batch {i // self.batch_size + 1}.")
                     # Decide if we should stop or continue on batch failure

            logger.info(f"Ingestion process finished. Total chunks added: {total_chunks_added}")
            return total_chunks_added > 0

        except Exception as e:
            logger.exception(f"An error occurred during the ingestion process: {e}")
            return False
