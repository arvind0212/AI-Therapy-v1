# CBT/DBT RAG Assistant

This project implements a Retrieval-Augmented Generation (RAG) assistant
focused on Cognitive Behavioral Therapy (CBT) and Dialectical Behavior Therapy (DBT)
documents. It allows ingesting documents (PDF, TXT) into a vector store and querying them using a local LLM via Ollama.

## Features

*   Modular architecture with clear separation of concerns (core interfaces, infrastructure implementations, services).
*   Document ingestion pipeline (Load -> Chunk -> Embed -> Store).
    *   Supports PDF and TXT files.
    *   Uses Sentence Transformers for embeddings.
    *   Stores data in PostgreSQL with the pgvector extension.
*   RAG query pipeline (Embed Query -> Retrieve Chunks -> Construct Prompt -> Generate Response).
    *   Uses Ollama for local LLM interaction.
*   Docker Compose setup for easy local deployment of PostgreSQL/pgvector and Ollama.
*   Command-line scripts for ingestion (`run_ingestion.py`) and querying (`run_query.py`).
*   Basic Streamlit web UI (`src/ui/app.py`) for interactive querying.

## Architecture Overview

The project follows a clean architecture pattern:

*   **`src/core`**: Defines core business logic, interfaces (ABCs), and data models (Pydantic).
*   **`src/infrastructure`**: Contains concrete implementations of the core interfaces, interacting with external systems like databases (PostgreSQL), embedding models (Sentence Transformers), and LLMs (Ollama).
*   **`src/services`**: Orchestrates the application logic by combining core components through their interfaces (Dependency Injection).
*   **`src/ui`**: Contains the Streamlit user interface code.
*   **`scripts`**: Holds command-line utilities for tasks like ingestion and querying.
*   **`config`**: Manages application configuration (settings loading, logging config).
*   **`data`**: Stores raw source documents and potentially processed data.

## Setup

**Prerequisites:**

*   [Docker](https://www.docker.com/get-started/) and Docker Compose
*   [Python](https://www.python.org/) (>=3.10 recommended)
*   [Poetry](https://python-poetry.org/docs/#installation) for dependency management

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd cbt_dbt_rag_assistant
    ```
2.  **Create Environment File:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Review and modify `.env` if necessary (e.g., database credentials if not using defaults, different Ollama port).
3.  **Start Backend Services:**
    *   Ensure Docker Desktop (or your Docker engine) is running.
    *   Run Docker Compose to start PostgreSQL/pgvector and Ollama:
        ```bash
        docker-compose up -d
        ```
    *   Verify containers are running: `docker ps` (You should see `rag_postgres` and `rag_ollama`).
4.  **Install Dependencies:**
    *   Use Poetry to install Python dependencies into a virtual environment:
        ```bash
        poetry install
        ```
5.  **Pull Ollama Model:**
    *   Download the default LLM model specified in `.env` (e.g., `gemma:7b`) into the Ollama container:
        ```bash
        docker exec -it rag_ollama ollama pull gemma:7b
        ```
        (Replace `gemma:7b` if you changed `DEFAULT_LLM_MODEL` in `.env`).

## Usage

Ensure the Docker containers are running (`docker-compose up -d`). All commands should be run from the project root directory (`cbt_dbt_rag_assistant/`).

**1. Ingest Documents:**

*   Place your `.txt` or `.pdf` documents into the `data/raw/cbt/` or `data/raw/dbt/` subdirectories (or any structure within `data/raw/`).
*   Run the ingestion script, pointing it to the source directory:
    ```bash
    poetry run python scripts/run_ingestion.py data/raw
    ```
    *   You can adjust chunking parameters using options:
        ```bash
        poetry run python scripts/run_ingestion.py data/raw --chunk-size 1500 --chunk-overlap 300
        ```

**2. Query via Command Line:**

*   Use the query script to ask questions about the ingested documents:
    ```bash
    poetry run python scripts/run_query.py "Your question here?"
    ```
    *   You can specify the number of retrieved chunks (`top_k`):
        ```bash
        poetry run python scripts/run_query.py "Your question here?" --top-k 3
        ```

**3. Use the Streamlit Web UI:**

*   Launch the Streamlit application:
    ```bash
    poetry run streamlit run src/ui/app.py
    ```
*   Open the local URL provided (usually `http://localhost:8501`) in your web browser.
*   Use the chat interface to ask questions.

## Configuration

*   **Environment Variables:** Sensitive or deployment-specific settings are managed in the `.env` file (see `.env.example` for details). This includes database connection URL, Ollama API endpoint, and default model names.
*   **Logging:** Logging behavior is configured in `config/logging.yaml`. Currently, there's an issue loading this file reliably in scripts, so it falls back to basic console logging (see Known Issues).

## Known Issues

*   **Logging Configuration:** The loading of the `config/logging.yaml` file via `logging.config.dictConfig` fails intermittently or consistently in script execution (`run_ingestion.py`, `run_query.py`), causing the scripts to fall back to basic logging. The root cause needs further investigation. The Streamlit app might handle logging differently.
*   **Streamlit/Torch Watcher Error:** A traceback related to `torch.classes` may appear in the console when starting the Streamlit app. This seems related to Streamlit's file watcher and doesn't appear to affect the app's core functionality currently.

## Next Steps

*   Implement FastAPI backend (Phase 3).
*   Add comprehensive unit and integration tests (Phase 4).
*   Refine chunking, prompt engineering, and model choices (Phase 4).
*   Implement metadata filtering and conversation history (Phase 4).
*   Improve error handling and user feedback (Phase 4).
*   Resolve the logging configuration issue (Phase 4).
