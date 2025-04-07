# src/core/models/__init__.py
# Makes models easily importable, e.g., from src.core.models import Document, DocumentChunk

from .document import Document, DocumentChunk

__all__ = ["Document", "DocumentChunk"]
