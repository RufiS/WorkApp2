"""File processing functionality for document ingestion"""

import os
import time
import logging
import tempfile
from typing import List, Dict, Any, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)

from core.config import performance_config
from utils.loaders.pdf_hyperlink_loader import PDFHyperlinkLoader
from utils.common.error_handler import CommonErrorHandler, with_error_context
from utils.logging.error_logging import log_error, log_warning
from utils.error_handling.enhanced_decorators import with_advanced_retry, with_timing

logger = logging.getLogger(__name__)


class FileProcessor:
    """Handles file loading, validation, and chunking operations"""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initialize the file processor

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Verify and adjust parameters
        self.chunk_size, self.chunk_overlap = self._verify_chunk_parameters()

        logger.info(f"File processor initialized with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def _verify_chunk_parameters(self) -> Tuple[int, int]:
        """
        Verify and potentially adjust chunk size and overlap parameters

        Returns:
            Tuple of (chunk_size, chunk_overlap)
        """
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap

        # Ensure chunk size is reasonable
        if chunk_size < 100:
            logger.warning(f"Chunk size {chunk_size} is too small, adjusting to 100")
            chunk_size = 100
        elif chunk_size > 8000:
            logger.warning(f"Chunk size {chunk_size} is too large, adjusting to 8000")
            chunk_size = 8000

        # Ensure chunk overlap is reasonable
        if chunk_overlap < 0:
            logger.warning(f"Negative chunk overlap {chunk_overlap} is invalid, adjusting to 0")
            chunk_overlap = 0
        elif chunk_overlap >= chunk_size:
            logger.warning(
                f"Chunk overlap {chunk_overlap} is >= chunk size {chunk_size}, adjusting to {chunk_size // 4}"
            )
            chunk_overlap = chunk_size // 4
        elif chunk_overlap > chunk_size // 2:
            logger.warning(
                f"Chunk overlap {chunk_overlap} is > 50% of chunk size, which may be inefficient"
            )

        # Update instance variables if adjusted
        if chunk_size != self.chunk_size or chunk_overlap != self.chunk_overlap:
            logger.info(f"Adjusted chunk parameters: size={chunk_size}, overlap={chunk_overlap}")
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        return chunk_size, chunk_overlap

    @with_error_context("file loader selection")
    def get_file_loader(self, file_path: str):
        """
        Get the appropriate loader for a file based on its extension

        Args:
            file_path: Path to the file

        Returns:
            A document loader instance

        Raises:
            ValueError: If the file type is not supported
            FileNotFoundError: If the file does not exist
            PermissionError: If the file cannot be accessed due to permissions
        """
        # Check if file exists and is accessible
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not os.path.isfile(file_path):
            raise ValueError(f"Not a file: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Permission denied: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()

        # Support PDF, TXT, and DOCX files
        if file_ext == ".txt":
            return TextLoader(file_path)
        elif file_ext == ".pdf":
            if performance_config.extract_pdf_hyperlinks:
                return PDFHyperlinkLoader(file_path)
            else:
                return PyPDFLoader(file_path)
        elif file_ext in [".docx", ".doc"]:
            return UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {file_ext}. Only PDF, TXT, and DOCX files are supported."
            )

    @with_timing(threshold=1.0)
    @with_advanced_retry(max_attempts=3, backoff_factor=2)
    def process_file_upload(self, file) -> List[Dict[str, Any]]:
        """
        Process a file object (from streamlit file uploader)

        Args:
            file: File object from streamlit file uploader

        Returns:
            List of document chunks with metadata
        """
        temp_file_path = None
        try:
            # Validate file
            if not file or not hasattr(file, "getvalue") or len(file.getvalue()) == 0:
                error_msg = f"Empty or invalid file: {getattr(file, 'name', 'unknown')}"
                log_error(error_msg, include_traceback=False)
                raise ValueError(error_msg)

            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(file.name)[1]
            ) as temp_file:
                temp_file.write(file.getvalue())
                temp_file_path = temp_file.name

            # Process the temporary file
            chunks = self.load_and_chunk_document(temp_file_path)

            # Check if chunks are empty and log it
            if not chunks:
                error_msg = f"No content extracted from file {file.name}. File may be empty, corrupted, or in an unsupported format."
                logger.warning(error_msg)
                log_warning(error_msg, include_traceback=False)

            return chunks

        except Exception as e:
            error_msg = f"Error processing file {getattr(file, 'name', 'unknown')}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            log_error(error_msg, include_traceback=True)
            raise RuntimeError(error_msg) from e

        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_file_path}: {str(e)}")

    @with_timing(threshold=1.0)
    @with_error_context("document loading and chunking")
    def load_and_chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a document and split it into chunks

        Args:
            file_path: Path to the document file

        Returns:
            List of document chunks with metadata
        """
        # Validate file
        self._validate_file(file_path)

        try:
            # Get appropriate loader
            loader = self.get_file_loader(file_path)

            # Load document
            documents = self._load_document(loader, file_path)

            # Check if documents were loaded
            if not documents:
                warning_msg = f"No content loaded from {os.path.basename(file_path)}"
                logger.warning(warning_msg)
                log_warning(warning_msg)
                return []

            # Create and configure text splitter
            text_splitter = self._create_text_splitter()

            # Log chunking parameters
            logger.info(
                f"Chunking document {os.path.basename(file_path)} with size={self.chunk_size}, overlap={self.chunk_overlap}"
            )

            # Split documents into chunks
            chunks = self._split_documents(text_splitter, documents, file_path)

            # Format chunks with metadata
            formatted_chunks = self._format_chunks(chunks, file_path)

            logger.info(f"Successfully processed {file_path}: {len(formatted_chunks)} chunks")
            return formatted_chunks

        except Exception as e:
            error_msg = f"Error processing document {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            log_error(error_msg)
            raise

    def _validate_file(self, file_path: str) -> None:
        """Validate file existence and accessibility"""
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            log_error(error_msg, include_traceback=False)
            raise FileNotFoundError(error_msg)

        if not os.access(file_path, os.R_OK):
            error_msg = f"Permission denied: {file_path}"
            log_error(error_msg, include_traceback=False)
            raise PermissionError(error_msg)

        if os.path.getsize(file_path) == 0:
            error_msg = f"Empty file: {file_path}"
            log_error(error_msg, include_traceback=False)
            raise ValueError(error_msg)

    def _load_document(self, loader, file_path: str):
        """Load document using the provided loader"""
        try:
            documents = loader.load()
            return documents
        except Exception as e:
            error_msg = f"Error loading document {file_path}: {str(e)}"
            log_error(error_msg)
            raise RuntimeError(f"Failed to load document: {str(e)}") from e

    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create and configure text splitter"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=False,
        )

    def _split_documents(self, text_splitter, documents, file_path: str):
        """Split documents into chunks"""
        try:
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split document into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            error_msg = f"Error chunking document {file_path}: {str(e)}"
            log_error(error_msg)
            raise RuntimeError(f"Failed to chunk document: {str(e)}") from e

    def _format_chunks(self, chunks, file_path: str) -> List[Dict[str, Any]]:
        """Format chunks with metadata and handle edge cases"""
        formatted_chunks = []
        total_chunk_size = 0
        small_chunks_count = 0
        large_chunks_count = 0
        empty_chunks_count = 0

        for i, chunk in enumerate(chunks):
            # Check for empty or whitespace-only chunks
            if not chunk.page_content or chunk.page_content.isspace():
                logger.warning(
                    f"Empty or whitespace-only chunk detected in {os.path.basename(file_path)}, chunk {i}"
                )
                log_warning(
                    f"Empty or whitespace-only chunk detected in {os.path.basename(file_path)}, chunk {i}"
                )
                empty_chunks_count += 1
                continue

            # Log chunk size
            chunk_size = len(chunk.page_content)
            total_chunk_size += chunk_size

            # Check for abnormal chunk sizes
            if chunk_size < 50:
                small_chunks_count += 1
            elif chunk_size > self.chunk_size * 1.5:
                large_chunks_count += 1

            # Create chunk with enhanced metadata
            formatted_chunks.append({
                "id": f"{os.path.basename(file_path)}-{i}",
                "text": chunk.page_content,
                "metadata": {
                    "source": file_path,
                    "page": chunk.metadata.get("page", None),
                    "chunk_index": i,
                    "chunk_size": chunk_size,
                    "creation_time": time.time(),
                    "chunk_params": {
                        "size": self.chunk_size,
                        "overlap": self.chunk_overlap,
                    },
                },
            })

        # Log warning if no valid chunks were found
        if not formatted_chunks:
            error_msg = f"No valid chunks extracted from {os.path.basename(file_path)}. File may be empty or contain only non-text content."
            logger.error(error_msg)
            log_error(error_msg, include_traceback=False)

        # Log chunking summary
        avg_chunk_size = total_chunk_size / len(formatted_chunks) if formatted_chunks else 0
        logger.info(f"Chunking summary for {os.path.basename(file_path)}:")
        logger.info(f"  - Total chunks: {len(formatted_chunks)}")
        logger.info(f"  - Average chunk size: {avg_chunk_size:.2f} chars")
        logger.info(f"  - Empty chunks skipped: {empty_chunks_count}")
        logger.info(f"  - Small chunks (<50 chars): {small_chunks_count}")
        logger.info(f"  - Large chunks (>{self.chunk_size * 1.5} chars): {large_chunks_count}")

        return formatted_chunks
