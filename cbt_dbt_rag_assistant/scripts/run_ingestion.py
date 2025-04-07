#!/usr/bin/env python
# scripts/run_ingestion.py
"""
Command-line script to run the document ingestion pipeline.
"""

import logging
import logging.config
import yaml
import typer
from pathlib import Path
from typing_extensions import Annotated
import sys

# --- Add project root to sys.path ---
# This allows importing from 'src' when running the script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End sys.path modification ---

# --- Pre-calculate paths needed for logging ---
# Avoid importing 'settings' here to prevent circular dependencies during logging setup
_project_root_for_log = Path(__file__).resolve().parent.parent
_log_config_path = _project_root_for_log / 'config' / 'logging.yaml'
_log_dir = _project_root_for_log / 'logs'
_log_file_path = _log_dir / 'app.log'
# --- End path pre-calculation ---

# --- Logging Setup ---
# This block attempts to configure logging from the YAML file.
# If it fails or the file doesn't exist, it falls back to basic configuration.
log_config_loaded = False
try:
    if _log_config_path.exists():
        _log_dir.mkdir(parents=True, exist_ok=True) # Ensure log dir exists
        with open(_log_config_path, 'rt') as f:
            log_config = yaml.safe_load(f.read())
        if isinstance(log_config, dict):
            # Set the log file path using the pre-calculated path
            if 'file' in log_config.get('handlers', {}):
                log_config['handlers']['file']['filename'] = str(_log_file_path)
            logging.config.dictConfig(log_config)
            log_config_loaded = True
            print(f"Logging configured successfully from {_log_config_path}") # Use print as logger might not be ready
        else:
            print(f"Warning: YAML config file at {_log_config_path} did not load as a dictionary.") # Use print
    else:
        print(f"Warning: Logging config file not found at {_log_config_path}.") # Use print

except Exception as e:
    print(f"Error loading logging config from {_log_config_path}: {e}.") # Use print

if not log_config_loaded:
    print("Using basic logging configuration.") # Use print
    logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Get logger instance *after* configuration attempt
logger = logging.getLogger(__name__)
if log_config_loaded:
     logger.info(f"Logging configured successfully from {_log_config_path}")
elif _log_config_path.exists():
     logger.error(f"Logging config file {_log_config_path} was invalid or caused an error. Using basic config.")
else:
     logger.warning(f"Logging config file not found at {_log_config_path}. Using basic config.")
# --- End Logging Setup ---


# Import configuration, services, and infrastructure components
# These imports happen *after* logging is configured
from src.core.config import settings
from src.services.ingestion_service import IngestionService
from src.infrastructure.data_loaders.file_loader import FileLoader
from src.infrastructure.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.infrastructure.vector_stores.postgres_vector_store import PostgresVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Create a Typer application
app = typer.Typer(help="CLI for running the RAG assistant ingestion pipeline.")

@app.command()
def main(
    source_path: Annotated[Path, typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Path to the source file or directory containing documents to ingest."
    )],
    chunk_size: Annotated[int, typer.Option(help="Target size for text chunks.")] = 1000,
    chunk_overlap: Annotated[int, typer.Option(help="Overlap size between consecutive chunks.")] = 200,
    batch_size: Annotated[int, typer.Option(help="Number of chunks to process in each batch.")] = 100,
):
    """
    Runs the ingestion pipeline: Load -> Chunk -> Embed -> Store.
    """
    # Note: Initial print statements happen before logger might be fully configured by YAML
    print(f"Starting ingestion for: {source_path}")
    logger.info(f"Ingestion script started for source: {source_path}")
    logger.info(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}, Batch size: {batch_size}")

    try:
        # --- Initialize Components ---
        print("Initializing components...")
        logger.info("Initializing infrastructure components...")

        data_loader = FileLoader()
        # Ensure embedding dimension from settings is used for consistency check
        embedding_model = SentenceTransformerEmbedder(
            model_name=settings.embedding_model_name
            # device can be configured if needed, e.g., based on env var or availability
        )
        # Pass the correct dimension to the vector store constructor
        vector_store = PostgresVectorStore(
             db_url=str(settings.database_url),
             embedding_dim=embedding_model.dimension # Use actual model dimension
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        ingestion_service = IngestionService(
            data_loader=data_loader,
            text_splitter=text_splitter,
            embedding_model=embedding_model,
            vector_store=vector_store,
            batch_size=batch_size
        )
        print("Components initialized.")
        logger.info("Components initialized successfully.")

        # --- Run Ingestion ---
        print("Running ingestion pipeline...")
        success = ingestion_service.run_ingestion(source=source_path)

        if success:
            print("✅ Ingestion completed successfully.")
            logger.info("Ingestion pipeline completed successfully.")
            raise typer.Exit(code=0) # Success
        else:
            # Distinguish between failure and just no documents found
            # Use source_path here, which is the argument to main()
            # Re-load documents to check if the source was actually empty vs. an ingestion error
            docs_check = data_loader.load(source_path)
            if not docs_check:
                 print("ℹ️ No documents found or loaded from the source. Ingestion finished.")
                 logger.warning("No documents found or loaded from the source. Ingestion finished.")
                 # Exit with success code 0 as it's not an error if the source is empty
                 raise typer.Exit(code=0)
            else:
                 print("❌ Ingestion failed or no chunks were added. Check logs for details.")
                 logger.error("Ingestion pipeline failed or no chunks were added.")
                 raise typer.Exit(code=1) # Failure

    except typer.Exit as e:
        # Re-raise typer.Exit exceptions to respect the exit code
        raise e
    except Exception as e:
        logger.exception(f"An unexpected error occurred during ingestion: {e}")
        print(f"❌ An unexpected error occurred: {e}")
        raise typer.Exit(code=1) # Failure

if __name__ == "__main__":
    app()
