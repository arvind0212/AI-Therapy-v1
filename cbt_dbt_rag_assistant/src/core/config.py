# src/core/config.py
"""
Centralized configuration access point.

Imports the settings instance from the main config module
so other parts of the application can import it from here.
"""

from config.settings import settings

__all__ = ["settings"] # Makes 'settings' available for wildcard imports if needed
