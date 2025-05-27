import os
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader

from utils.pdf_hyperlink_loader import PDFHyperlinkLoader
from utils.config_unified import performance_config

logger = logging.getLogger(__name__)

def get_file_loader(file_path: str):
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
        raise ValueError(f"Unsupported file type: {file_ext}. Only PDF, TXT, and DOCX files are supported.")

def load_and_chunk_document(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """
    Load a document and split it into chunks with caching

    Args:
        file_path: Path to the document file
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks with metadata

    Raises:
        FileNotFoundError: If the file does not exist
        PermissionError: If the file cannot be accessed due to permissions
        ValueError: If the file type is not supported
        IOError: If there's an I/O error during file reading
        RuntimeError: If document loading or chunking fails
        Exception: For any other unexpected errors
    """
    # Check if file exists
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Check if file is readable
    if not os.access(file_path, os.R_OK):
        error_msg = f"Permission denied: {file_path}"
        logger.error(error_msg)
        raise PermissionError(error_msg)

    # Check if file is empty
    if os.path.getsize(file_path) == 0:
        error_msg = f"Empty file: {file_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Compute document hash for metadata
        doc_hash = compute_document_hash(file_path)

        # Get appropriate loader
        try:
            loader = get_file_loader(file_path)
        except ValueError as e:
            error_msg = f"Unsupported file type: {file_path}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        # Load document
        try:
            documents = loader.load()
        except IOError as e:
            error_msg = f"I/O error loading document {file_path}: {str(e)}"
            logger.error(error_msg)
            raise IOError(error_msg) from e
        except UnicodeDecodeError as e:
            error_msg = f"Unicode decode error in document {file_path}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Error loading document {file_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Check if documents were loaded
        if not documents:
            warning_msg = f"No content loaded from {os.path.basename(file_path)}"
            logger.warning(warning_msg)
            return []

        # Create text splitter with optimized settings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=False
        )

        # Log chunking parameters
        logger.info(f"Chunking document {os.path.basename(file_path)} with size={chunk_size}, overlap={chunk_overlap}")

        # Split documents into chunks
        try:
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split document into {len(chunks)} chunks")
        except Exception as e:
            error_msg = f"Error chunking document {file_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Format chunks with metadata
        formatted_chunks = []
        total_chunk_size = 0
        small_chunks_count = 0
        large_chunks_count = 0
        empty_chunks_count = 0

        for i, chunk in enumerate(chunks):
            # Check for empty or whitespace-only chunks
            if not chunk.page_content or chunk.page_content.isspace():
                logger.warning(f"Empty or whitespace-only chunk detected in {os.path.basename(file_path)}, chunk {i}")
                empty_chunks_count += 1
                continue

            # Log chunk size
            chunk_size = len(chunk.page_content)
            total_chunk_size += chunk_size

            # Check for abnormal chunk sizes
            if chunk_size < 50:
                warning_msg = f"Abnormally small chunk detected in {os.path.basename(file_path)}, chunk {i}: {chunk_size} chars"
                logger.warning(warning_msg)
                small_chunks_count += 1
            elif chunk_size > chunk_size * 1.5:
                warning_msg = f"Abnormally large chunk detected in {os.path.basename(file_path)}, chunk {i}: {chunk_size} chars"
                logger.warning(warning_msg)
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
                    "doc_hash": doc_hash,
                    "creation_time": time.time(),
                    "chunk_params": {
                        "size": chunk_size,
                        "overlap": chunk_overlap
                    }
                }
            })

        # Log warning if no valid chunks were found
        if not formatted_chunks:
            error_msg = f"No valid chunks extracted from {os.path.basename(file_path)}. File may be empty or contain only non-text content."
            logger.error(error_msg)

        # Log chunking metrics
        total_chunk_size = sum(len(chunk.get('text', '')) for chunk in formatted_chunks)
        empty_chunks_count = sum(1 for chunk in formatted_chunks if not chunk.get('text', '').strip())
        small_chunks_count = sum(1 for chunk in formatted_chunks if 0 < len(chunk.get('text', '').strip()) < 50)
        large_chunks_count = sum(1 for chunk in formatted_chunks if len(chunk.get('text', '')) > chunk_size * 1.5)

        # Handle small chunks by merging them with adjacent chunks
        if small_chunks_count > 0:
            formatted_chunks = handle_small_chunks(formatted_chunks, min_size=50, file_path=file_path)
            logger.info(f"Handled {small_chunks_count} small chunks, resulting in {len(formatted_chunks)} chunks")

        # Update metrics
        logger.info(f"Chunking summary for {os.path.basename(file_path)}:")
        logger.info(f"  - Total chunks: {len(formatted_chunks)}")
        logger.info(f"  - Average chunk size: {total_chunk_size / len(formatted_chunks) if formatted_chunks else 0:.2f} chars")
        logger.info(f"  - Empty chunks skipped: {empty_chunks_count}")
        logger.info(f"  - Small chunks (<50 chars): {small_chunks_count}")
        logger.info(f"  - Large chunks (>{chunk_size * 1.5} chars): {large_chunks_count}")

        return formatted_chunks
    except Exception as e:
        error_msg = f"Error processing document {file_path}: {str(e)}"
        logger.error(error_msg)
        raise

def handle_small_chunks(chunks: List[Dict[str, Any]], min_size: int = 50, file_path: str = None) -> List[Dict[str, Any]]:
    """
    Handle abnormally small chunks by merging them with adjacent chunks or removing them

    Args:
        chunks: List of document chunks
        min_size: Minimum acceptable chunk size in characters
        file_path: Path to the document file for logging purposes

    Returns:
        List of processed chunks with small chunks handled
    """
    if not chunks:
        return []

    # Early return if only one chunk and it's too small
    if len(chunks) == 1 and len(chunks[0].get('text', '')) < min_size:
        logger.warning(f"Document contains only one small chunk ({len(chunks[0].get('text', ''))} chars)")
        return chunks  # Return as is, can't merge with anything

    result = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        current_text = current.get('text', '')

        # If current chunk is too small
        if len(current_text) < min_size:
            file_name = os.path.basename(file_path) if file_path else "unknown"
            logger.warning(f"Abnormally small chunk detected in {file_name}, chunk {i}: {len(current_text)} chars")

            # Try to merge with next chunk if available
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                next_text = next_chunk.get('text', '')

                # Create merged chunk
                merged = current.copy()
                merged['text'] = current_text + " " + next_text
                merged['merged'] = True
                merged['original_indices'] = [i, i + 1]

                result.append(merged)
                i += 2  # Skip the next chunk since we merged it
                logger.info(f"Merged small chunk {i-1} with chunk {i}")
            else:
                # If this is the last chunk, try to merge with previous
                if result:  # If we have previous chunks
                    prev = result.pop()  # Remove the last chunk
                    prev_text = prev.get('text', '')

                    # Create merged chunk
                    merged = prev.copy()
                    merged['text'] = prev_text + " " + current_text
                    merged['merged'] = True
                    merged['original_indices'] = [i-1, i] if 'original_indices' not in prev else prev['original_indices'] + [i]

                    result.append(merged)
                    logger.info(f"Merged small chunk {i} with previous chunk")
                else:
                    # If no previous chunks, just add it despite being small
                    result.append(current)
                i += 1
        else:
            # Current chunk is large enough, add it as is
            result.append(current)
            i += 1

    return result

def compute_document_hash(file_path: str) -> str:
    """
    Compute a hash for a document file

    Args:
        file_path: Path to the document file

    Returns:
        Hash string for the document
    """
    try:
        # Check if file exists and is readable
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"Permission denied: {file_path}")

        # Use a buffer to handle large files efficiently
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                hasher.update(byte_block)

        # Get the hexadecimal digest
        hash_value = hasher.hexdigest()
        logger.debug(f"Computed hash for {os.path.basename(file_path)}: {hash_value}")
        return hash_value
    except (FileNotFoundError, PermissionError) as e:
        # Re-raise these specific exceptions
        logger.error(f"Error accessing file for hashing: {str(e)}")
        raise
    except IOError as e:
        logger.warning(f"I/O error computing hash for {file_path}: {str(e)}")
        # Return a fallback hash based on filename and modification time
        return compute_fallback_hash(file_path)
    except Exception as e:
        logger.warning(f"Error computing hash for {file_path}: {str(e)}")
        # Return a fallback hash based on filename and modification time
        return compute_fallback_hash(file_path)

def compute_fallback_hash(file_path: str) -> str:
    """
    Compute a fallback hash based on filename and modification time
    when the standard hashing method fails

    Args:
        file_path: Path to the document file

    Returns:
        Fallback hash string
    """
    try:
        # Get file metadata
        mtime = os.path.getmtime(file_path)
        file_size = os.path.getsize(file_path)

        # Create a string with file info
        fallback = f"{file_path}:{mtime}:{file_size}"

        # Hash the string
        hash_value = hashlib.md5(fallback.encode()).hexdigest()
        logger.info(f"Using fallback hash for {os.path.basename(file_path)}: {hash_value}")
        return hash_value
    except Exception as e:
        # Last resort fallback
        logger.error(f"Error computing fallback hash: {str(e)}")
        return hashlib.md5(file_path.encode()).hexdigest()
