# src/infrastructure/llm_providers/ollama_provider.py
"""
Concrete implementation of the LLM interface using the Ollama API client.
"""

import logging
from typing import List, Optional, Dict, Any
import ollama

# Import interface and settings
from src.core.interfaces.llm import LLM
from src.core.config import settings

logger = logging.getLogger(__name__)

class OllamaProvider(LLM):
    """Uses the ollama library to interact with a local Ollama instance."""

    def __init__(
        self,
        host: str = str(settings.ollama_api_base), # Get host URL from settings
        default_model: str = settings.default_llm_model,
        default_options: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the OllamaProvider.

        Args:
            host: The base URL of the Ollama API.
            default_model: The default model to use if none is specified in generate.
            default_options: Default options for the Ollama API calls (e.g., temperature).
        """
        self.default_model = default_model
        self.default_options = default_options or {}
        logger.info(f"Initializing OllamaProvider with host: {host}, default model: {self.default_model}")
        try:
            # Initialize the Ollama client
            # The host parameter connects it to the specified Ollama instance
            self.client = ollama.Client(host=host)
            # Optional: Check connection or list models to ensure connectivity
            # self.client.list()
            logger.info("Ollama client initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize Ollama client for host {host}: {e}")
            raise

    def generate(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs: Any # Allow passthrough of other ollama specific args
    ) -> str:
        """
        Generates a response from the Ollama LLM.

        Args:
            prompt: The input prompt for the LLM.
            history: List of previous conversation turns in Ollama format
                     (e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]).
            model: The specific Ollama model to use (overrides default).
            options: Ollama-specific options (e.g., temperature, top_p). Merged with defaults.
            **kwargs: Additional arguments passed directly to ollama.chat or ollama.generate.

        Returns:
            The generated text response.
        """
        target_model = model or self.default_model
        request_options = {**self.default_options, **(options or {})}

        logger.info(f"Generating response using Ollama model: {target_model}")
        logger.debug(f"Prompt: {prompt[:100]}...") # Log truncated prompt
        logger.debug(f"History: {history}")
        logger.debug(f"Options: {request_options}")
        logger.debug(f"Kwargs: {kwargs}")

        try:
            if history:
                # Use chat endpoint if history is provided
                messages = history + [{"role": "user", "content": prompt}]
                response = self.client.chat(
                    model=target_model,
                    messages=messages,
                    options=request_options,
                    **kwargs
                )
                # Extract content from the assistant's message
                generated_text = response.get('message', {}).get('content', '')
            else:
                # Use generate endpoint for single prompts (simpler)
                response = self.client.generate(
                    model=target_model,
                    prompt=prompt,
                    options=request_options,
                    **kwargs
                )
                generated_text = response.get('response', '')

            logger.info(f"Received response from {target_model}.")
            logger.debug(f"Generated Text: {generated_text[:100]}...") # Log truncated response
            return generated_text.strip()

        except Exception as e:
            logger.exception(f"Error during Ollama API call to model {target_model}: {e}")
            # Return an error message or raise a custom exception
            return f"[Error generating response: {e}]"
