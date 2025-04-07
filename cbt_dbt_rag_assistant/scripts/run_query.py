#!/usr/bin/env python
# scripts/run_query.py
"""
Command-line script to run a query through the RAG pipeline.
"""

import logging
import logging.config
import yaml
import typer
from pathlib import Path
from typing_extensions import Annotated
import sys
import time # To measure query time

# --- Add project root to sys.path ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# --- End sys.path modification ---

# --- Pre-calculate paths needed for logging ---
_project_root_for_log = Path(__file__).resolve().parent.parent
_log_config_path = _project_root_for_log / 'config' / 'logging.yaml'
_log_dir = _project_root_for_log / 'logs'
_log_file_path = _log_dir / 'app.log'
# --- End path pre-calculation ---

# --- Logging Setup ---
# Using the same robust setup as run_ingestion.py
log_config_loaded = False
try:
    if _log_config_path.exists():
        _log_dir.mkdir(parents=True, exist_ok=True)
        with open(_log_config_path, 'rt') as f:
            log_config = yaml.safe_load(f.read())
        if isinstance(log_config, dict):
            if 'file' in log_config.get('handlers', {}):
                log_config['handlers']['file']['filename'] = str(_log_file_path)
            logging.config.dictConfig(log_config)
            log_config_loaded = True
            print(f"Logging configured successfully from {_log_config_path}")
        else:
            print(f"Warning: YAML config file at {_log_config_path} did not load as a dictionary.")
    else:
        print(f"Warning: Logging config file not found at {_log_config_path}.")

except Exception as e:
    print(f"Error loading logging config from {_log_config_path}: {e}.")

if not log_config_loaded:
    print("Using basic logging configuration.")
    logging.basicConfig(level="INFO", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
if log_config_loaded:
     logger.info(f"Logging configured successfully from {_log_config_path}")
elif _log_config_path.exists():
     logger.error(f"Logging config file {_log_config_path} was invalid or caused an error. Using basic config.")
else:
     logger.warning(f"Logging config file not found at {_log_config_path}. Using basic config.")
# --- End Logging Setup ---


# Import configuration, services, and infrastructure components
from src.core.config import settings
from src.services.query_service import QueryService
from src.infrastructure.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.infrastructure.vector_stores.postgres_vector_store import PostgresVectorStore
from src.infrastructure.llm_providers.ollama_provider import OllamaProvider


# Create a Typer application
app = typer.Typer(help="CLI for running queries through the RAG assistant.")

@app.command()
def main(
    query: Annotated[str, typer.Argument(help="The user query to process.")],
    top_k: Annotated[int, typer.Option(help="Number of relevant chunks to retrieve.")] = 5,
):
    """
    Runs the RAG query pipeline for a given user query.
    """
    print(f"\nProcessing query: '{query}'")
    logger.info(f"Query script started for query: '{query}'")
    logger.info(f"Top K retrieval: {top_k}")

    start_time = time.time()

    try:
        # --- Initialize Components ---
        print("Initializing components...")
        logger.info("Initializing infrastructure components...")

        # Ensure embedding dimension from settings is used for consistency check
        embedding_model = SentenceTransformerEmbedder(
            model_name=settings.embedding_model_name
        )
        # Pass the correct dimension to the vector store constructor
        vector_store = PostgresVectorStore(
             db_url=str(settings.database_url),
             embedding_dim=embedding_model.dimension
        )
        llm = OllamaProvider(
            host=str(settings.ollama_api_base),
            default_model=settings.default_llm_model
        )

        query_service = QueryService(
            embedding_model=embedding_model,
            vector_store=vector_store,
            llm=llm,
            top_k_retrieval=top_k
        )
        print("Components initialized.")
        logger.info("Components initialized successfully.")

        # --- Run Query ---
        print("Running query pipeline...")
        response = query_service.query(user_query=query)

        end_time = time.time()
        duration = end_time - start_time

        print("\n--- Response ---")
        print(response)
        print("----------------")
        print(f"(Query processed in {duration:.2f} seconds)")
        logger.info(f"Query processed successfully in {duration:.2f} seconds.")
        raise typer.Exit(code=0)

    except typer.Exit as e:
        # Re-raise typer.Exit exceptions to respect the exit code
        raise e
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.exception(f"An unexpected error occurred during query processing: {e}")
        print(f"\n❌ An unexpected error occurred after {duration:.2f} seconds: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
