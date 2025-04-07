import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import PostgresDsn, HttpUrl, Field
from dotenv import load_dotenv

# Determine the base directory of the project
# This assumes settings.py is in the config/ directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file in the project root
dotenv_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=dotenv_path)

class Settings(BaseSettings):
    """Application settings."""

    # Pydantic-settings configuration
    # Tells pydantic-settings to load from the .env file
    model_config = SettingsConfigDict(
        env_file=str(dotenv_path),
        env_file_encoding='utf-8',
        case_sensitive=False,  # Environment variables are typically case-insensitive
        extra='ignore' # Ignore extra fields not defined in the model
    )

    # Database configuration
    database_url: PostgresDsn = Field(..., validation_alias='DATABASE_URL')

    # Ollama configuration
    ollama_api_base: HttpUrl = Field(..., validation_alias='OLLAMA_API_BASE')
    default_llm_model: str = Field(..., validation_alias='DEFAULT_LLM_MODEL')

    # Embedding model configuration
    embedding_model_name: str = Field(..., validation_alias='EMBEDDING_MODEL_NAME')
    embedding_dimension: int = Field(..., validation_alias='EMBEDDING_DIMENSION')

    # Logging configuration
    log_level: str = Field("INFO", validation_alias='LOG_LEVEL')

    # Project paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / 'data'
    raw_data_dir: Path = data_dir / 'raw'
    processed_data_dir: Path = data_dir / 'processed'
    log_dir: Path = BASE_DIR / 'logs'
    config_dir: Path = BASE_DIR / 'config'

    # Ensure log directory exists
    def __init__(self, **values):
        super().__init__(**values)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# Instantiate settings
# This instance will be imported by other modules
settings = Settings()

# Example usage (optional, for testing)
if __name__ == "__main__":
    print("Loaded Settings:")
    print(f"  Database URL: {settings.database_url}")
    print(f"  Ollama Base URL: {settings.ollama_api_base}")
    print(f"  Default LLM: {settings.default_llm_model}")
    print(f"  Embedding Model: {settings.embedding_model_name}")
    print(f"  Embedding Dimension: {settings.embedding_dimension}")
    print(f"  Log Level: {settings.log_level}")
    print(f"  Base Directory: {settings.base_dir}")
    print(f"  Log Directory: {settings.log_dir}")
