# src/infrastructure/data_loaders/file_loader.py
"""
Concrete implementation of the DataLoader interface for loading
text from local files (.txt, .pdf).
"""

import logging
from pathlib import Path
from typing import List, Any, Union
import PyPDF2 # Corrected import name

# Import interface and models
from src.core.interfaces.data_loader import DataLoader
from src.core.models.document import Document

logger = logging.getLogger(__name__)

class FileLoader(DataLoader):
    """Loads documents from local text or PDF files."""

    SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Loads documents from a file path or a directory path.

        Args:
            source: A string or Path object representing a file or directory.

        Returns:
            A list of Document objects loaded from the source.
        """
        source_path = Path(source)
        documents = []

        if not source_path.exists():
            logger.error(f"Source path does not exist: {source_path}")
            return []

        if source_path.is_file():
            if source_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                documents.extend(self._load_single_file(source_path))
            else:
                logger.warning(f"Unsupported file type skipped: {source_path}")
        elif source_path.is_dir():
            logger.info(f"Loading documents from directory: {source_path}")
            for item in source_path.rglob('*'): # Use rglob for recursive search
                if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    documents.extend(self._load_single_file(item))
                # else: # Optional: log skipped files/dirs
                #     if item.is_file():
                #         logger.debug(f"Skipping unsupported file in directory: {item}")
        else:
            logger.error(f"Source path is neither a file nor a directory: {source_path}")

        logger.info(f"Loaded {len(documents)} documents from source: {source_path}")
        return documents

    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Loads content from a single supported file."""
        docs = []
        file_extension = file_path.suffix.lower()
        source_str = str(file_path.resolve()) # Use resolved path as source identifier
        logger.debug(f"Loading file: {file_path}")

        try:
            if file_extension == ".txt":
                try:
                    content = file_path.read_text(encoding='utf-8')
                    metadata = {"source": source_str}
                    docs.append(Document(content=content, metadata=metadata))
                except UnicodeDecodeError:
                    try:
                        # Try a different encoding if utf-8 fails
                        content = file_path.read_text(encoding='latin-1')
                        metadata = {"source": source_str}
                        docs.append(Document(content=content, metadata=metadata))
                        logger.warning(f"File {file_path} read with latin-1 encoding after utf-8 failed.")
                    except Exception as e_enc:
                        logger.error(f"Failed to read text file {file_path} with multiple encodings: {e_enc}")
                except Exception as e:
                     logger.error(f"Failed to read text file {file_path}: {e}")


            elif file_extension == ".pdf":
                try:
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        num_pages = len(reader.pages)
                        for page_num in range(num_pages):
                            page = reader.pages[page_num]
                            content = page.extract_text()
                            if content: # Only add pages with extracted text
                                metadata = {
                                    "source": source_str,
                                    "page": page_num + 1, # 1-based page number
                                    "total_pages": num_pages
                                }
                                docs.append(Document(content=content, metadata=metadata))
                            else:
                                logger.warning(f"No text extracted from page {page_num + 1} of {file_path}")
                except Exception as e:
                    logger.error(f"Failed to read PDF file {file_path}: {e}")

        except Exception as e:
            logger.exception(f"Unexpected error loading file {file_path}: {e}") # Log full traceback

        return docs
