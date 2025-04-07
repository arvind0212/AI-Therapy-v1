# src/core/models/document.py
"""
Core data models for representing documents and their chunks.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid

class Document(BaseModel):
    """Represents a single loaded document before chunking."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict) # e.g., {'source': 'file.pdf', 'page': 1}

class DocumentChunk(BaseModel):
    """Represents a chunk of a document, potentially with its embedding."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str # ID of the parent Document
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict) # Inherited + chunk-specific info
    embedding: Optional[List[float]] = None # Embedding vector

    # Allow extra fields for flexibility if needed, though usually avoided
    # class Config:
    #     extra = 'allow'
