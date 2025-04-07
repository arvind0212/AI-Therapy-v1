# src/infrastructure/vector_stores/postgres_vector_store.py
"""
Concrete implementation of the VectorStore interface using PostgreSQL with pgvector.
"""

import logging
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, text, Column, String, Index
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.exc import SQLAlchemyError
from pgvector.sqlalchemy import Vector  # Import Vector type

# Import interface and models
from src.core.interfaces.vector_store import VectorStore
from src.core.models.document import DocumentChunk
from src.core.config import settings # Import the application settings

logger = logging.getLogger(__name__)

# Define the SQLAlchemy base
Base = declarative_base()

# Define the ORM model for our document chunks
class ChunkModel(Base):
    __tablename__ = 'document_chunks'

    id = Column(PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    document_id = Column(String, nullable=False, index=True)
    content = Column(String, nullable=False)
    chunk_metadata = Column(JSONB, nullable=False, server_default='{}') # Renamed from metadata
    # Define the embedding column using the Vector type from pgvector
    # The dimension needs to match the embedding model being used
    embedding = Column(Vector(settings.embedding_dimension), nullable=True)

    # Add an index for the embedding column using a suitable pgvector index type
    # HNSW is often recommended for high performance, but requires pgvector >= 0.5.0
    # IVFFlat is another option. Using cosine distance (vector_cosine_ops) is common.
    # Note: Index creation might take time on large datasets.
    # Consider creating indexes separately via migrations if needed.
    __table_args__ = (
        Index(
            'ix_document_chunks_embedding',
            embedding,
            postgresql_using='hnsw', # or 'ivfflat'
            postgresql_with={'m': 16, 'ef_construction': 64}, # HNSW parameters
            # postgresql_with={'lists': 100}, # IVFFlat parameter example
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
        Index('ix_document_chunks_chunk_metadata', chunk_metadata, postgresql_using='gin'), # Index for renamed metadata queries
    )

class PostgresVectorStore(VectorStore):
    """PostgreSQL/pgvector implementation of the VectorStore interface."""

    def __init__(self, db_url: str = str(settings.database_url), embedding_dim: int = settings.embedding_dimension):
        """
        Initializes the PostgresVectorStore.

        Args:
            db_url: The PostgreSQL connection URL.
            embedding_dim: The dimension of the vectors to be stored.
        """
        logger.info(f"Initializing PostgresVectorStore with DB URL: {db_url[:db_url.find('@') + 1]}... and dimension: {embedding_dim}")
        try:
            self.engine = create_engine(db_url, pool_pre_ping=True)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self._embedding_dim = embedding_dim # Store dimension if needed later
            self._create_table_and_extension()
            logger.info("PostgresVectorStore initialized successfully.")
        except SQLAlchemyError as e:
            logger.exception(f"Failed to initialize PostgresVectorStore: {e}")
            raise

    def _create_table_and_extension(self):
        """Ensures the 'vector' extension is enabled and the table exists."""
        logger.info("Ensuring 'vector' extension and 'document_chunks' table exist...")
        try:
            with self.engine.connect() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                connection.commit() # Commit extension creation explicitly
            # Create tables defined in Base's metadata
            Base.metadata.create_all(bind=self.engine)
            logger.info("'vector' extension checked/enabled and 'document_chunks' table checked/created.")
        except SQLAlchemyError as e:
            logger.exception(f"Error during table/extension creation: {e}")
            # Decide if this should be a fatal error or just a warning
            raise # Re-raise for now

    def add(self, chunks: List[DocumentChunk], **kwargs: Any) -> List[str]:
        """Adds document chunks to the PostgreSQL database."""
        if not chunks:
            logger.warning("Add called with empty list of chunks.")
            return []

        chunk_models = []
        added_ids = []
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(f"Chunk {chunk.id} has no embedding. Skipping.")
                continue
            if len(chunk.embedding) != self._embedding_dim:
                 logger.error(f"Chunk {chunk.id} embedding dimension {len(chunk.embedding)} != store dimension {self._embedding_dim}. Skipping.")
                 continue

            chunk_models.append(ChunkModel(
                id=chunk.id, # Use provided ID
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_metadata=chunk.metadata or {}, # Use renamed field, ensure metadata is not None
                embedding=chunk.embedding
            ))
            added_ids.append(chunk.id)

        if not chunk_models:
             logger.warning("No valid chunks with embeddings found to add.")
             return []

        session: Session = self.SessionLocal()
        try:
            logger.info(f"Adding {len(chunk_models)} chunks to the database...")
            # Consider using bulk_save_objects for potentially better performance
            session.add_all(chunk_models)
            session.commit()
            logger.info(f"Successfully added {len(chunk_models)} chunks.")
            return added_ids
        except SQLAlchemyError as e:
            logger.exception(f"Database error occurred during add: {e}")
            session.rollback()
            return [] # Return empty list on failure
        finally:
            session.close()

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[DocumentChunk]:
        """Queries the vector store for chunks similar to the query embedding."""
        if not query_embedding:
            logger.warning("Query called with empty query embedding.")
            return []
        if len(query_embedding) != self._embedding_dim:
            logger.error(f"Query embedding dimension {len(query_embedding)} != store dimension {self._embedding_dim}.")
            return []

        session: Session = self.SessionLocal()
        try:
            logger.info(f"Querying for top {top_k} chunks.")
            stmt = select(ChunkModel)

            # --- Metadata Filtering (Example: simple key-value equality) ---
            # This part needs refinement based on actual filtering needs.
            # Using JSONB operators like ->> for text comparison.
            # Ensure the 'ix_document_chunks_metadata' GIN index supports your queries.
            if filters:
                logger.debug(f"Applying filters: {filters}")
                for key, value in filters.items():
                    # Example: Filter where chunk_metadata->>'source' == 'some_value'
                    # Adjust the operator and value type as needed.
                    stmt = stmt.where(ChunkModel.chunk_metadata[key].astext == str(value)) # Use renamed field
            # --- End Metadata Filtering ---

            # Order by cosine distance (smaller is better) and limit
            # <=> is the cosine distance operator in pgvector
            stmt = stmt.order_by(ChunkModel.embedding.cosine_distance(query_embedding)).limit(top_k)

            results = session.execute(stmt).scalars().all()
            logger.info(f"Found {len(results)} chunks matching query.")

            # Convert SQLAlchemy models back to Pydantic models
            document_chunks = [
                DocumentChunk(
                    id=str(result.id), # Convert UUID back to string if needed by model
                    document_id=result.document_id,
                    content=result.content,
                    metadata=result.chunk_metadata, # Use renamed field
                    embedding=result.embedding # Keep embedding if needed downstream
                ) for result in results
            ]
            return document_chunks

        except SQLAlchemyError as e:
            logger.exception(f"Database error occurred during query: {e}")
            return [] # Return empty list on failure
        finally:
            session.close()

# Note: Need to import 'select' from sqlalchemy
from sqlalchemy import select
