# src/services/query_service.py
"""
Service layer for handling RAG queries.
"""

import logging
from typing import List, Optional, Dict, Any

# Import interfaces and models
from src.core.interfaces.embedding_model import EmbeddingModel
from src.core.interfaces.vector_store import VectorStore
from src.core.interfaces.llm import LLM
from src.core.models.document import DocumentChunk

logger = logging.getLogger(__name__)

# --- Default Prompt Template ---
# This can be made more configurable later (e.g., loaded from a file or settings)
DEFAULT_PROMPT_TEMPLATE = """
Based on the following context documents, please answer the user's query.
Provide a concise and helpful answer grounded in the provided context.
If the context doesn't contain the answer, state that you couldn't find relevant information in the documents.

Context Documents:
---
{context}
---

User Query: {query}

Answer:
"""

class QueryService:
    """Orchestrates the RAG query process."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        llm: LLM,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        top_k_retrieval: int = 5 # Number of chunks to retrieve
    ):
        """
        Initializes the QueryService.

        Args:
            embedding_model: An instance of an EmbeddingModel implementation.
            vector_store: An instance of a VectorStore implementation.
            llm: An instance of an LLM implementation.
            prompt_template: The template string used to construct the final prompt.
            top_k_retrieval: The number of relevant document chunks to retrieve.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.prompt_template = prompt_template
        self.top_k_retrieval = top_k_retrieval
        logger.info("QueryService initialized.")

    def query(self, user_query: str) -> str:
        """
        Processes a user query using the RAG pipeline.

        Args:
            user_query: The query string from the user.

        Returns:
            The generated response string from the LLM.
        """
        logger.info(f"Received query: '{user_query}'")

        try:
            # 1. Embed the query
            logger.info("Step 1: Embedding query...")
            query_embedding = self.embedding_model.embed_query(user_query)
            if not query_embedding:
                logger.error("Failed to embed query.")
                return "[Error: Failed to embed query]"
            logger.info("Query embedded successfully.")

            # 2. Retrieve relevant chunks
            logger.info(f"Step 2: Retrieving top {self.top_k_retrieval} relevant chunks...")
            # TODO: Add metadata filtering capabilities if needed based on user query or context
            retrieved_chunks = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=self.top_k_retrieval,
                filters=None # Placeholder for potential future filters
            )
            if not retrieved_chunks:
                logger.warning("No relevant chunks found in vector store.")
                # Optionally, still ask the LLM without context, or return a specific message
                # For now, we'll proceed without context.
                context_str = "No relevant context documents found."
            else:
                logger.info(f"Retrieved {len(retrieved_chunks)} chunks.")
                # Format context for the prompt
                context_str = "\n---\n".join([
                    f"Source: {chunk.metadata.get('source', 'N/A')}, Page: {chunk.metadata.get('page', 'N/A')}\n{chunk.content}"
                    for chunk in retrieved_chunks
                ])
            logger.debug(f"Context String: {context_str[:200]}...") # Log truncated context

            # 3. Construct the prompt
            logger.info("Step 3: Constructing prompt...")
            final_prompt = self.prompt_template.format(
                context=context_str,
                query=user_query
            )
            logger.debug(f"Final Prompt: {final_prompt[:200]}...") # Log truncated prompt

            # 4. Generate response using LLM
            logger.info("Step 4: Generating response from LLM...")
            response = self.llm.generate(prompt=final_prompt)
            logger.info("Response generated successfully.")

            return response

        except Exception as e:
            logger.exception(f"An error occurred during the query process: {e}")
            return f"[Error processing query: {e}]"
