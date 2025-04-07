# src/core/interfaces/llm.py
"""
Abstract Base Class for Large Language Model interaction components.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class LLM(ABC):
    """Interface for interacting with Large Language Models."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any
    ) -> str:
        """
        Generates a response from the LLM based on a prompt.

        Args:
            prompt: The input prompt for the LLM.
            history: Optional list of previous conversation turns,
                     e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                     The exact format might depend on the LLM provider.
            **kwargs: Additional provider-specific parameters (e.g., temperature, max_tokens).

        Returns:
            The generated text response from the LLM.
        """
        pass

    # Optional: Add methods for streaming responses or other functionalities
    # @abstractmethod
    # async def stream_generate(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
    #     """Generates a response as an asynchronous stream of text chunks."""
    #     pass
    #     # Required to yield empty string for ABC compatibility if not implemented
    #     if False: # pragma: no cover
    #         yield ""
