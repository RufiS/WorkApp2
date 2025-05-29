# WorkApp2 Progress: Last completed Ret-2 (Potential Index Dimension Mismatch)
# Next pending Ret-3 (Inconsistent Error Handling in keyword_fallback_search())
# PART 1: Fixed hardcoded file paths by consistently using resolve_path
# Doc-5 Fix: Enhanced path resolution with better logging and consistent use of resolve_path
# throughout the codebase to ensure cross-platform compatibility and proper path handling.
# PART 2: Fixed potential index dimension mismatch
# Ret-2 Fix: Enhanced dimension mismatch handling with better error messages, migration instructions,
# batch processing for large indices, backup creation before rebuilding, and improved error recovery.

from dataclasses import dataclass, field

import os
import time
import json
import logging
import hashlib
import shutil
import threading
import tempfile
from typing import List, Dict, Tuple, Any, Optional, Union, Set

import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from sentence_transformers import SentenceTransformer

from utils.config_unified import retrieval_config, performance_config, app_config, resolve_path
from utils.pdf_hyperlink_loader import PDFHyperlinkLoader
from utils.index_management.index_operations import get_saved_chunk_params
from utils.index_management.index_manager_unified import index_manager
from utils.error_logging import query_logger
from utils.error_handling.decorators import with_retry, with_error_handling, RetryableError
from utils.error_handling.enhanced_decorators import with_advanced_retry, with_timing
from utils.error_logging import log_error, log_warning

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ChunkCacheEntry:
    """Cache entry for document chunks"""
    chunks: List[Dict[str, Any]]
    timestamp: float = field(default_factory=time.time)
    ttl: float = 3600 * 24  # Time to live in seconds (default: 24 hours)
    
    def is_expired(self) -> bool:
        """Check if the cache entry is expired"""
        return time.time() - self.timestamp > self.ttl

class DocumentProcessor:
    """Handles document loading, chunking, and indexing with optimized caching"""
    
    def __init__(self, embedding_model_name: str = None):
        """
        Initialize the document processor
        
        Args:
            embedding_model_name: Name of the embedding model to use (defaults to config value)
        """
        self.embedding_model_name = embedding_model_name or retrieval_config.embedding_model
        self.chunk_size = retrieval_config.chunk_size
        self.chunk_overlap = retrieval_config.chunk_overlap
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Verify and adjust chunk parameters if needed
        self.chunk_size, self.chunk_overlap = self._verify_chunk_parameters()
        
        # Initialize FAISS index
        self.index = None
        self.texts = []
        self.chunks = []  # Alias for self.texts for backward compatibility
        self.processed_files = set()
        
        # Initialize cache
        self.chunk_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.enable_cache = performance_config.enable_chunk_cache
        self.cache_size = performance_config.chunk_cache_size
        
        # Initialize metrics
        self.total_documents = 0
        self.total_chunks = 0
        self.embedding_times = []
        self.max_embedding_times = 100
        
        # Store chunk parameters
        self.saved_chunk_params = (self.chunk_size, self.chunk_overlap)
        
        # Check for GPU availability
        self.gpu_available = self._check_gpu_availability()
        if self.gpu_available:
            logger.info("GPU is available for embeddings")
        else:
            logger.info("GPU is not available, using CPU for embeddings")
        
        logger.info(f"Document processor initialized with model {self.embedding_model_name}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for embeddings"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
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
            logger.warning(f"Chunk overlap {chunk_overlap} is >= chunk size {chunk_size}, adjusting to {chunk_size // 4}")
            chunk_overlap = chunk_size // 4
        elif chunk_overlap > chunk_size // 2:
            logger.warning(f"Chunk overlap {chunk_overlap} is > 50% of chunk size, which may be inefficient")
        
        # Check if parameters were adjusted
        if chunk_size != self.chunk_size or chunk_overlap != self.chunk_overlap:
            logger.info(f"Adjusted chunk parameters: size={chunk_size}, overlap={chunk_overlap}")
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        return chunk_size, chunk_overlap
    def _get_cache_key(self, file_path: str, chunk_size: int, chunk_overlap: int) -> str:
        """
        Generate a cache key for document chunks
        
        Args:
            file_path: Path to the document file
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Cache key string
        """
        # Get file modification time
        mtime = os.path.getmtime(file_path)
        
        # Create a dictionary of chunking parameters
        params_dict = {
            "file_path": file_path,
            "mtime": mtime,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        
        # Convert to JSON and hash
        params_json = json.dumps(params_dict, sort_keys=True)
        return hashlib.md5(params_json.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, chunks: List[Dict[str, Any]], ttl: Optional[float] = None) -> None:
        """
        Add chunks to the cache
        
        Args:
            key: Cache key
            chunks: Chunks to cache
            ttl: Time to live in seconds (None for default)
        """
        if not self.enable_cache:
            return
            
        # Create cache entry
        entry = ChunkCacheEntry(chunks=chunks, ttl=ttl or 3600 * 24)
        
        # Add to cache
        self.chunk_cache[key] = entry
        
        # Trim cache if needed
        if len(self.chunk_cache) > self.cache_size:
            # Remove oldest entry
            try:
                if self.chunk_cache:
                    oldest_key = min(self.chunk_cache.keys(), key=lambda k: self.chunk_cache[k].timestamp)
                    del self.chunk_cache[oldest_key]
                    logger.debug(f"Removed oldest cache entry with key {oldest_key}")
            except (ValueError, KeyError) as e:
                # This would happen if the cache became empty during processing
                # or if the key was already removed by another thread
                logger.warning(f"Error removing oldest cache entry: {str(e)}")
                # Fallback: clear a random entry if cache is still too large
                if len(self.chunk_cache) > self.cache_size and self.chunk_cache:
                    try:
                        random_key = next(iter(self.chunk_cache.keys()))
                        del self.chunk_cache[random_key]
                        logger.debug(f"Removed random cache entry with key {random_key} as fallback")
                    except (StopIteration, KeyError) as inner_e:
                        # Last resort: clear the entire cache if we can't remove a single entry
                        logger.warning(f"Cache management failed: {str(inner_e)}, clearing entire cache")
                        # Create a new cache with just the current entry to prevent memory leaks
                        self.chunk_cache = {}
                        self.chunk_cache[key] = entry  # Keep only the current entry
                        logger.info(f"Cache reset with single entry for key {key}")
    
    def _get_from_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get chunks from the cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached chunks or None if not found
        """
        if not self.enable_cache:
            return None
            
        # Check if key exists in cache
        if key in self.chunk_cache:
            entry = self.chunk_cache[key]
            
            # Check if entry is expired
            if entry.is_expired():
                # Remove expired entry
                del self.chunk_cache[key]
                self.cache_misses += 1
                return None
            
            # Update timestamp to keep entry fresh
            entry.timestamp = time.time()
            self.cache_hits += 1
            return entry.chunks
        
        self.cache_misses += 1
        return None
    
    def has_index(self, index_dir: Optional[str] = None) -> bool:
        """
        Check if an index exists either in memory or on disk
        
        Args:
            index_dir: Directory to check for index files (defaults to config value)
            
        Returns:
            True if an index exists, False otherwise
        """
        # Check if index is loaded in memory
        if self.index is not None and len(self.texts) > 0:
            logger.info("Index is already loaded in memory")
            return True
        
        # Use configured index path if not specified
        index_dir = index_dir or retrieval_config.index_path
        
        # Resolve the path to ensure consistency
        index_dir = resolve_path(index_dir, create_dir=True)
        
        # Check if index exists on disk
        index_file = os.path.join(index_dir, "index.faiss")
        texts_file = os.path.join(index_dir, "texts.npy")
        logger.debug(f"Resolved index directory for checking: {index_dir}")
        
        # Check if files exist
        files_exist = os.path.exists(index_file) and os.path.exists(texts_file)
        
        if files_exist:
            logger.info(f"Index files found at {index_file} and {texts_file}")
        else:
            logger.warning(f"Index files not found at {index_file} and/or {texts_file}")
        
        return files_exist
        
    def create_empty_index(self) -> None:
        """
        Create a new empty FAISS index
        """
        logger.info("Creating new empty FAISS index")
        
        # Initialize empty index with correct dimensions
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Initialize empty texts list
        self.texts = []
        self.chunks = []  # Alias for backward compatibility
        
        # Initialize empty processed files set
        self.processed_files = set()
        
        # Reset metrics
        self.total_documents = 0
        self.total_chunks = 0
        
        logger.info(f"Created empty index with dimension {self.embedding_dim}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get processor metrics
        
        Returns:
            Dictionary with processor metrics
        """
        metrics = {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.chunk_cache),
            "cache_max_size": self.cache_size,
            "gpu_available": self.gpu_available
        }
        
        # Calculate cache hit rate
        total_cache_accesses = self.cache_hits + self.cache_misses
        if total_cache_accesses > 0:
            metrics["cache_hit_rate"] = self.cache_hits / total_cache_accesses
        else:
            metrics["cache_hit_rate"] = 0.0
        
        # Calculate average embedding time
        if self.embedding_times:
            metrics["avg_embedding_time"] = sum(self.embedding_times) / len(self.embedding_times)
            metrics["min_embedding_time"] = min(self.embedding_times)
            metrics["max_embedding_time"] = max(self.embedding_times)
        else:
            metrics["avg_embedding_time"] = 0.0
            metrics["min_embedding_time"] = 0.0
            metrics["max_embedding_time"] = 0.0
        
        return metrics
    
    def _get_file_loader(self, file_path: str):
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
    
    @with_timing(threshold=1.0)
    @with_advanced_retry(max_attempts=3, backoff_factor=2)
    def process_file(self, file) -> List[Dict[str, Any]]:
        """
        Process a file object (from streamlit file uploader)
        
        Args:
            file: File object from streamlit file uploader
            
        Returns:
            List of document chunks with metadata
            
        Raises:
            ValueError: If file is empty or invalid
            IOError: If there's an I/O error during file processing
            RuntimeError: If document processing fails
            Exception: For any other unexpected errors
        """
        temp_file_path = None
        try:
            # Check if file is empty
            if not file or not hasattr(file, 'getvalue') or len(file.getvalue()) == 0:
                error_msg = f"Empty or invalid file: {getattr(file, 'name', 'unknown')}"
                log_error(error_msg, include_traceback=False)
                raise ValueError(error_msg)
                
            # Create a temporary file to save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp_file:
                temp_file.write(file.getvalue())
                temp_file_path = temp_file.name
            
            # Process the temporary file
            chunks = self.load_and_chunk_document(temp_file_path)
            
            # Check if chunks are empty and log it
            if not chunks:
                error_msg = f"No content extracted from file {file.name}. File may be empty, corrupted, or in an unsupported format."
                logger.warning(error_msg)
                # Log to central error log
                log_warning(error_msg, include_traceback=False)
            
            return chunks
            
        except FileNotFoundError as e:
            error_msg = f"File not found error processing {getattr(file, 'name', 'unknown')}: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg)
            raise ValueError(error_msg) from e
            
        except PermissionError as e:
            error_msg = f"Permission denied for file {getattr(file, 'name', 'unknown')}: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg)
            raise ValueError(error_msg) from e
            
        except ValueError as e:
            error_msg = f"Invalid file or format for {getattr(file, 'name', 'unknown')}: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg)
            raise ValueError(error_msg) from e
            
        except IOError as e:
            error_msg = f"I/O error processing file {getattr(file, 'name', 'unknown')}: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg)
            raise IOError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error processing file {getattr(file, 'name', 'unknown')}: {str(e)}"
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
    @with_advanced_retry(max_attempts=3, backoff_factor=2)
    def load_and_chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a document and split it into chunks with caching
        
        Args:
            file_path: Path to the document file
            
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
            log_error(error_msg, include_traceback=False)
            raise FileNotFoundError(error_msg)
        
        # Check if file is readable
        if not os.access(file_path, os.R_OK):
            error_msg = f"Permission denied: {file_path}"
            log_error(error_msg, include_traceback=False)
            raise PermissionError(error_msg)
        
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            error_msg = f"Empty file: {file_path}"
            log_error(error_msg, include_traceback=False)
            raise ValueError(error_msg)
        
        # Check cache first
        cache_key = self._get_cache_key(file_path, self.chunk_size, self.chunk_overlap)
        cached_chunks = self._get_from_cache(cache_key)
        if cached_chunks is not None:
            logger.info(f"Cache hit for document {os.path.basename(file_path)}")
            return cached_chunks
        
        try:
            # Compute document hash for metadata
            doc_hash = self._compute_document_hash(file_path)
            
            # Get appropriate loader
            try:
                loader = self._get_file_loader(file_path)
            except ValueError as e:
                error_msg = f"Unsupported file type: {file_path}"
                log_error(error_msg, include_traceback=False)
                raise ValueError(error_msg) from e
            
            # Load document
            try:
                documents = loader.load()
            except IOError as e:
                error_msg = f"I/O error loading document {file_path}: {str(e)}"
                log_error(error_msg)
                raise IOError(f"I/O error loading document: {str(e)}") from e
            except UnicodeDecodeError as e:
                error_msg = f"Unicode decode error in document {file_path}: {str(e)}"
                log_error(error_msg)
                raise ValueError(f"Document encoding error: {str(e)}") from e
            except Exception as e:
                error_msg = f"Error loading document {file_path}: {str(e)}"
                log_error(error_msg)
                raise RuntimeError(f"Failed to load document: {str(e)}") from e
            
            # Check if documents were loaded
            if not documents:
                warning_msg = f"No content loaded from {os.path.basename(file_path)}"
                logger.warning(warning_msg)
                log_warning(warning_msg)
                return []
            
            # Create text splitter with optimized settings
            # Verify chunk parameters again in case they were changed externally
            chunk_size, chunk_overlap = self._verify_chunk_parameters()
            
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
                log_error(error_msg)
                raise RuntimeError(f"Failed to chunk document: {str(e)}") from e
            
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
                    # Log to central error log
                    log_warning(f"Empty or whitespace-only chunk detected in {os.path.basename(file_path)}, chunk {i}")
                    empty_chunks_count += 1
                    continue
                
                # Log chunk size
                chunk_size = len(chunk.page_content)
                total_chunk_size += chunk_size
                
                # Check for abnormal chunk sizes
                if chunk_size < 50:
                    warning_msg = f"Abnormally small chunk detected in {os.path.basename(file_path)}, chunk {i}: {chunk_size} chars"
                    logger.warning(warning_msg)
                    # Log to central error log
                    log_warning(warning_msg)
                    small_chunks_count += 1
                elif chunk_size > self.chunk_size * 1.5:
                    warning_msg = f"Abnormally large chunk detected in {os.path.basename(file_path)}, chunk {i}: {chunk_size} chars"
                    logger.warning(warning_msg)
                    # Log to central error log
                    log_warning(warning_msg)
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
                            "size": self.chunk_size,
                            "overlap": self.chunk_overlap
                        }
                    }
                })
            
            # Log warning if no valid chunks were found
            if not formatted_chunks:
                error_msg = f"No valid chunks extracted from {os.path.basename(file_path)}. File may be empty or contain only non-text content."
                logger.error(error_msg)
                # Log to central error log
                log_error(error_msg, include_traceback=False)
            
            # Log chunking metrics
            total_chunk_size = sum(len(chunk.get('text', '')) for chunk in formatted_chunks)
            empty_chunks_count = sum(1 for chunk in formatted_chunks if not chunk.get('text', '').strip())
            small_chunks_count = sum(1 for chunk in formatted_chunks if 0 < len(chunk.get('text', '').strip()) < 50)
            large_chunks_count = sum(1 for chunk in formatted_chunks if len(chunk.get('text', '')) > self.chunk_size * 1.5)
            
            # Handle small chunks by merging them with adjacent chunks
            if small_chunks_count > 0:
                formatted_chunks = self._handle_small_chunks(formatted_chunks, min_size=50, file_path=file_path)
                logger.info(f"Handled {small_chunks_count} small chunks, resulting in {len(formatted_chunks)} chunks")
            
            # Update metrics
            self.total_documents += 1
            self.total_chunks += len(formatted_chunks)
            
            # Add to cache
            self._add_to_cache(cache_key, formatted_chunks)
            
            # Update metadata.json with document hash
            self._update_metadata_with_hash(file_path, doc_hash, len(formatted_chunks))
            
            # Log chunking summary
            avg_chunk_size = total_chunk_size / len(formatted_chunks) if formatted_chunks else 0
            logger.info(f"Chunking summary for {os.path.basename(file_path)}:")
            logger.info(f"  - Total chunks: {len(formatted_chunks)}")
            logger.info(f"  - Average chunk size: {avg_chunk_size:.2f} chars")
            logger.info(f"  - Empty chunks skipped: {empty_chunks_count}")
            logger.info(f"  - Small chunks (<50 chars): {small_chunks_count}")
            logger.info(f"  - Large chunks (>{self.chunk_size * 1.5} chars): {large_chunks_count}")
            
            return formatted_chunks
        except Exception as e:
            error_msg = f"Error processing document {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Log to central error log with full stack trace
            log_error(error_msg)
            raise
    
    def _handle_small_chunks(self, chunks: List[Dict[str, Any]], min_size: int = 50, file_path: str = None) -> List[Dict[str, Any]]:
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
                log_warning(f"Abnormally small chunk detected in {file_name}, chunk {i}: {len(current_text)} chars")
                
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
    
    @with_timing(threshold=1.0)
    def batch_embed_chunks(self, chunks: Sequence[Union[Dict[str, Any], str]], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Embed document chunks in batches
        
        Args:
            chunks: List of document chunks (either dictionaries with 'text' key or strings)
            batch_size: Size of batches for embedding (None for default)
            
        Returns:
            NumPy array of embeddings
        """
        # Sanity check: Ensure chunks is not None and is a list
        if chunks is None:
            logger.error("Cannot embed None chunks")
            raise ValueError("chunks parameter cannot be None")
            
        if not isinstance(chunks, list):
            logger.error(f"chunks must be a list, got {type(chunks)}")
            raise TypeError(f"chunks must be a list, got {type(chunks)}")
            
        if len(chunks) == 0:
            logger.warning("Empty chunks list provided for embedding")
            return np.array([])
        
        # Use configured batch size if not specified
        batch_size = batch_size or performance_config.embedding_batch_size
        
        # Sanity check: Ensure batch_size is positive
        if batch_size <= 0:
            logger.warning(f"Invalid batch size {batch_size}, using default of 32")
            batch_size = 32
        
        # Extract texts from chunks
        texts = []
        invalid_chunks = 0
        for chunk in chunks:
            if isinstance(chunk, dict) and "text" in chunk:
                if not chunk["text"] or not isinstance(chunk["text"], str):
                    logger.warning(f"Empty or non-string text in chunk: {chunk}")
                    texts.append("")
                    invalid_chunks += 1
                else:
                    texts.append(chunk["text"])
            elif isinstance(chunk, str):
                texts.append(chunk)
            else:
                logger.warning(f"Skipping invalid chunk format: {type(chunk)}")
                # Add empty string as placeholder to maintain index alignment
                texts.append("")
                invalid_chunks += 1
        
        # Sanity check: Log warning if too many invalid chunks
        if invalid_chunks > 0:
            logger.warning(f"Found {invalid_chunks} invalid chunks out of {len(chunks)} total chunks")
            if invalid_chunks / len(chunks) > 0.5:
                logger.error(f"More than 50% of chunks are invalid ({invalid_chunks}/{len(chunks)})")
        
        # Embed in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Time the embedding process
            start_time = time.time()
            try:
                batch_embeddings = self.embedding_model.encode(batch_texts)
                embedding_time = time.time() - start_time
                
                # Sanity check: Verify embedding dimensions
                if batch_embeddings.shape[1] != self.embedding_dim:
                    logger.error(f"Embedding dimension mismatch: got {batch_embeddings.shape[1]}, expected {self.embedding_dim}")
                    raise ValueError(f"Embedding dimension mismatch: got {batch_embeddings.shape[1]}, expected {self.embedding_dim}")
                
                # Update embedding times
                self.embedding_times.append(embedding_time)
                if len(self.embedding_times) > self.max_embedding_times:
                    self.embedding_times = self.embedding_times[-self.max_embedding_times:]
                
                all_embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
                # If this is the first batch and it fails, re-raise the exception
                if i == 0:
                    raise
                # Otherwise, log the error and continue with remaining batches
                logger.warning(f"Continuing with remaining batches after error in batch {i//batch_size + 1}")
        
        # Sanity check: Ensure we have embeddings
        if not all_embeddings:
            logger.error("No embeddings were generated")
            return np.array([])
        
        # Concatenate all embeddings
        result = np.vstack(all_embeddings)
        
        # Final sanity check: Verify shape of result
        if result.shape[0] != len(chunks):
            logger.error(f"Embedding count mismatch: got {result.shape[0]}, expected {len(chunks)}")
        if result.shape[1] != self.embedding_dim:
            logger.error(f"Final embedding dimension mismatch: got {result.shape[1]}, expected {self.embedding_dim}")
            
        return result
    
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index from embeddings
        
        Args:
            embeddings: NumPy array of embeddings
            
        Returns:
            FAISS index
            
        Raises:
            ValueError: If embedding dimensions don't match index dimensions
            AssertionError: If embeddings array is empty or invalid
        """
        # Assert that embeddings array is valid
        if embeddings is None or not isinstance(embeddings, np.ndarray):
            error_msg = f"Invalid embeddings array: {type(embeddings)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if len(embeddings) == 0:
            error_msg = "Empty embeddings array"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Assert that embedding dimensions match index dimensions
        if embeddings.shape[1] != self.embedding_dim:
            error_msg = f"Embedding dimensions ({embeddings.shape[1]}) don't match index dimensions ({self.embedding_dim})"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Create index
        index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Apply FAISS optimizations if enabled
        if performance_config.enable_faiss_optimization:
            # Use IVF index for faster search with large datasets
            if len(embeddings) > 1000:
                nlist = min(int(np.sqrt(len(embeddings))), 100)  # Rule of thumb for nlist
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)
                index.train(embeddings)
            
            # Use GPU if available and enabled
            if self.gpu_available and performance_config.use_gpu_for_faiss:
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.info("Using GPU for FAISS index")
                except Exception as e:
                    logger.warning(f"Failed to use GPU for FAISS: {str(e)}")
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Verify that index dimensions match embedding dimensions
        assert index.d == self.embedding_dim, f"Index dimensions ({index.d}) don't match embedding dimensions ({self.embedding_dim})"
        
        return index
    
    def _build_faiss_index(self, texts: List[str]) -> Tuple[faiss.Index, List[str]]:
        """
        Build a FAISS index from text chunks
        
        Args:
            texts: List of text chunks
            
        Returns:
            Tuple of (FAISS index, list of texts)
        """
        # Embed texts
        embeddings = self.embedding_model.encode(texts)
        
        # Build index
        index = self.build_index(embeddings)
        
        return index, texts
    
    def process_documents(self, file_paths: List[str], dry_run: bool = False) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """
        Process multiple documents and build a search index
        
        Args:
            file_paths: List of paths to document files
            dry_run: If True, preview changes without writing to disk
            
        Returns:
            Tuple of (FAISS index, list of chunks)
        """
        all_chunks = []
        processed_files_count = 0
        skipped_files_count = 0
        failed_files_count = 0
        
        if dry_run:
            logger.info("[DRY RUN] Running in dry run mode - no changes will be written to disk")
        
        # Process each document
        for file_path in file_paths:
            try:
                # Skip already processed files
                if file_path in self.processed_files:
                    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Skipping already processed file: {file_path}")
                    skipped_files_count += 1
                    continue
                
                # Load and chunk document
                chunks = self.load_and_chunk_document(file_path)
                all_chunks.extend(chunks)
                processed_files_count += 1
                
                # Mark file as processed (only if not in dry run mode)
                if not dry_run:
                    self.processed_files.add(file_path)
                    
                logger.info(f"{'[DRY RUN] ' if dry_run else ''}Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"{'[DRY RUN] ' if dry_run else ''}Error processing document {file_path}: {str(e)}")
                failed_files_count += 1
        
        # Perform semantic deduplication if there are enough chunks
        if len(all_chunks) > 1:
            original_chunk_count = len(all_chunks)
            all_chunks = self._semantic_deduplication(all_chunks)
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}After semantic deduplication: {len(all_chunks)} chunks (removed {original_chunk_count - len(all_chunks)} duplicates)")
        
        # Update chunks list (only if not in dry run mode)
        if not dry_run:
            self.texts = all_chunks
            self.chunks = self.texts  # Keep both references in sync
        
        # Embed chunks
        embeddings = self.batch_embed_chunks(all_chunks)
        
        # Build index (only if not in dry run mode)
        if not dry_run:
            self.index = self.build_index(embeddings)
        else:
            # Create a temporary index for preview purposes
            temp_index = self.build_index(embeddings)
            logger.info(f"[DRY RUN] Would add {len(all_chunks)} chunks to index")
            logger.info(f"[DRY RUN] Would process {processed_files_count} new files, skipped {skipped_files_count} already processed files, failed {failed_files_count} files")
            
            # Log what would happen in a real run
            logger.info("[DRY RUN] Summary of changes that would be made:")
            logger.info(f"[DRY RUN] - New documents processed: {processed_files_count}")
            logger.info(f"[DRY RUN] - Total chunks that would be added: {len(all_chunks)}")
            logger.info(f"[DRY RUN] - Index dimension would be: {self.embedding_dim}")
            
            # Return the temporary index without modifying self.index
            return temp_index, all_chunks
        
        return self.index, self.chunks
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using the query
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks with scores
            
        Raises:
            ValueError: If no index has been built
        """
        # Sanity check: Validate inputs
        if not query or not isinstance(query, str):
            logger.error(f"Invalid query: {query}")
            raise ValueError(f"Query must be a non-empty string, got {type(query)}")
            
        if not isinstance(top_k, int) or top_k <= 0:
            logger.error(f"Invalid top_k value: {top_k}")
            raise ValueError(f"top_k must be a positive integer, got {top_k}")
        
        # Log query and parameters for instrumentation
        start_time = time.time()
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"Searching for query: '{query_preview}' with top_k={top_k}")
        
        # Initialize metrics
        search_type = "vector"
        fallback_used = False
        # Try to load index if it exists on disk but not in memory
        if (self.index is None or not self.chunks) and self.has_index():
            try:
                self.load_index(resolve_path(retrieval_config.index_path))
                logger.info("Index loaded on demand during search")
            except Exception as e:
                logger.error(f"Failed to load index during search: {str(e)}")
                raise ValueError(f"Failed to load index: {str(e)}")
        
        # Check again after attempted load
        if self.index is None or not self.chunks:
            # Resolve the index path
            resolved_index_path = resolve_path(retrieval_config.index_path)
            logger.debug(f"Resolved index path for on-demand loading: {resolved_index_path}")
            try:
                self.load_index(resolved_index_path)
                logger.info(f"Index loaded from {resolved_index_path}")
            except Exception as e:
                logger.error(f"Failed to load index from {resolved_index_path}: {str(e)}")
                raise ValueError("No index has been built. Process documents first.")
        
        # Sanity check: Verify chunks list
        if not isinstance(self.chunks, list):
            logger.error(f"self.chunks is not a list: {type(self.chunks)}")
            raise TypeError(f"self.chunks must be a list, got {type(self.chunks)}")
            
        if len(self.chunks) == 0:
            logger.warning("Empty chunks list, search will return no results")
            return []
        
        # Embed query
        try:
            query_embedding = self.embedding_model.encode([query])
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise ValueError(f"Failed to embed query: {str(e)}")
        
        # Sanity check: Verify query embedding dimensions
        if query_embedding.shape[1] != self.embedding_dim:
            logger.error(f"Query embedding dimension mismatch: got {query_embedding.shape[1]}, expected {self.embedding_dim}")
            raise ValueError(f"Query embedding dimension mismatch: got {query_embedding.shape[1]}, expected {self.embedding_dim}")
        
        # Check if index is valid and has the 'd' attribute
        if not hasattr(self.index, 'd') or self.index.d is None:
            logger.error("Index is not properly initialized or is corrupted")
            # Try to rebuild the index with current embedding dimension
            logger.info(f"Creating new index with dimension {self.embedding_dim}")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            if len(self.chunks) > 0:
                # Re-embed chunks and add to index
                embeddings = self.batch_embed_chunks(self.chunks)
                self.index.add(embeddings)
                logger.info(f"Rebuilt index with {len(self.chunks)} chunks")
            else:
                logger.warning("No chunks available to rebuild index")
                raise ValueError("Index is corrupted and no chunks are available to rebuild it")
        
        # Check if embedding dimensions match index dimensions
        if query_embedding.shape[1] != self.index.d:
            logger.error(f"Query embedding dimension ({query_embedding.shape[1]}) doesn't match index dimension ({self.index.d})")
            
            # Create a detailed notification with clear information about the mismatch
            notification = {
                "type": "dimension_mismatch",
                "message": f"Embedding dimensions mismatch detected: query={query_embedding.shape[1]}, index={self.index.d}.",
                "details": {
                    "query_dimension": int(query_embedding.shape[1]),
                    "index_dimension": int(self.index.d),
                    "embedding_model": self.embedding_model_name,
                    "possible_causes": [
                        "The embedding model was changed since the index was built",
                        "The index was created with a different embedding model",
                        "The index file was corrupted or modified externally"
                    ],
                    "recommended_actions": [
                        "Rebuild the index with the current embedding model",
                        "Use the same embedding model that was used to build the index",
                        "Check for configuration changes in embedding_model setting"
                    ]
                }
            }
            
            # Log the detailed notification
            logger.warning(f"Dimension mismatch notification: {notification}")
            
            # Log to central error log with comprehensive information
            from utils.error_logging import log_error, log_warning
            log_warning(
                f"Embedding dimension mismatch: query={query_embedding.shape[1]}, index={self.index.d}",
                error_type="DIMENSION_MISMATCH",
                source="DocumentProcessor.search",
                additional_data={
                    "query": query,
                    "query_dimension": int(query_embedding.shape[1]),
                    "index_dimension": int(self.index.d),
                    "embedding_model": self.embedding_model_name,
                    "action": "rebuild_index",
                    "notification": notification
                }
            )
            
            # Check if we should attempt to rebuild or just notify
            rebuild_on_mismatch = retrieval_config.rebuild_index_on_dimension_mismatch
            
            if not rebuild_on_mismatch:
                # If automatic rebuilding is disabled, raise a clear error with migration instructions
                error_msg = (
                    f"Embedding dimension mismatch: query={query_embedding.shape[1]}, index={self.index.d}. " 
                    f"The index was built with a different embedding model than the current one ({self.embedding_model_name}). " 
                    f"Please rebuild the index manually using the current embedding model or configure 'rebuild_index_on_dimension_mismatch' to True."
                )
                logger.error(error_msg)
                
                # Add migration instructions to the error message
                migration_instructions = (
                    "\n\nMigration Instructions:\n" 
                    "1. Backup your current index files\n" 
                    "2. Set 'rebuild_index_on_dimension_mismatch' to True in your configuration\n" 
                    "3. Run a search query to trigger automatic rebuilding\n" 
                    "4. Alternatively, manually rebuild the index using the 'rebuild_index' command"
                )
                raise ValueError(error_msg + migration_instructions)
            
            # Attempt to rebuild the index with current embedding dimension
            logger.warning("Attempting to rebuild index with current embedding model dimensions")
            try:
                # Create a new index with the correct dimensions
                new_index = faiss.IndexFlatL2(self.embedding_dim)
                
                # Check if we have chunks to re-embed
                if not self.chunks or len(self.chunks) == 0:
                    # Try to load chunks from disk if available
                    try:
                        index_dir = resolve_path(retrieval_config.index_path)
                        texts_file = os.path.join(index_dir, "texts.npy")
                        if os.path.exists(texts_file):
                            logger.info(f"Loading chunks from {texts_file} for index migration")
                            self.chunks = np.load(texts_file, allow_pickle=True).tolist()
                            logger.info(f"Loaded {len(self.chunks)} chunks from disk for migration")
                        else:
                            raise ValueError("No chunks available in memory or on disk to rebuild the index")
                    except Exception as load_error:
                        logger.error(f"Failed to load chunks from disk: {str(load_error)}")
                        raise ValueError(f"No chunks available to rebuild the index and failed to load from disk: {str(load_error)}")
                
                # Re-embed all chunks with the current model
                if self.chunks and len(self.chunks) > 0:
                    logger.info(f"Re-embedding {len(self.chunks)} chunks with model {self.embedding_model_name}")
                    
                    # Create a backup of the index before rebuilding if configured
                    if retrieval_config.backup_before_rebuild:
                        try:
                            index_dir = resolve_path(retrieval_config.index_path)
                            backup_dir = os.path.join(index_dir, f"backup_{int(time.time())}")
                            os.makedirs(backup_dir, exist_ok=True)
                            
                            # Copy index files to backup directory
                            for filename in ["index.faiss", "texts.npy", "metadata.json"]:
                                src_file = os.path.join(index_dir, filename)
                                if os.path.exists(src_file):
                                    dst_file = os.path.join(backup_dir, filename)
                                    shutil.copy2(src_file, dst_file)
                                    logger.info(f"Backed up {src_file} to {dst_file}")
                            
                            logger.info(f"Created index backup at {backup_dir}")
                        except Exception as backup_error:
                            logger.warning(f"Failed to create index backup: {str(backup_error)}")
                    
                    # Re-embed chunks in batches to avoid memory issues
                    batch_size = retrieval_config.embedding_batch_size or 32
                    total_chunks = len(self.chunks)
                    new_embeddings = None
                    
                    for i in range(0, total_chunks, batch_size):
                        batch_end = min(i + batch_size, total_chunks)
                        logger.info(f"Re-embedding batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}: chunks {i} to {batch_end-1}")
                        
                        batch_chunks = self.chunks[i:batch_end]
                        batch_embeddings = self.batch_embed_chunks(batch_chunks)
                        
                        if new_embeddings is None:
                            new_embeddings = batch_embeddings
                        else:
                            new_embeddings = np.vstack((new_embeddings, batch_embeddings))
                    
                    # Add embeddings to the new index
                    new_index.add(new_embeddings)
                    self.index = new_index
                    logger.info(f"Successfully rebuilt index with dimension {self.embedding_dim}")
                    
                    # Save the rebuilt index to disk if configured
                    if retrieval_config.save_rebuilt_index:
                        try:
                            # Resolve the index path
                            resolved_index_path = resolve_path(retrieval_config.index_path, create_dir=True)
                            logger.info(f"Saving rebuilt index to disk at {resolved_index_path}")
                            self.save_index(resolved_index_path)
                            logger.info("Successfully saved rebuilt index")
                        except Exception as save_error:
                            logger.error(f"Failed to save rebuilt index: {str(save_error)}")
                            log_error(
                                f"Failed to save rebuilt index: {str(save_error)}",
                                include_traceback=True,
                                error_type="INDEX_SAVE_FAILURE",
                                source="DocumentProcessor.search"
                            )
                    
                    # Continue with the search using the rebuilt index
                    # Continue with the search using the rebuilt index
                    safe_top_k = min(top_k, len(self.chunks))  # Ensure top_k doesn't exceed available chunks
                    scores, indices = self.index.search(query_embedding, safe_top_k)
                    
                    # Notify about successful migration
                    log_warning(
                        f"Successfully migrated index from dimension {self.index.d} to {self.embedding_dim}",
                        error_type="DIMENSION_MIGRATION_SUCCESS",
                        source="DocumentProcessor.search",
                        additional_data={
                            "original_dimension": int(self.index.d),
                            "new_dimension": int(self.embedding_dim),
                            "embedding_model": self.embedding_model_name,
                            "chunks_count": len(self.chunks)
                        }
                    )
                else:
                    raise ValueError("No chunks available to rebuild the index")
            except Exception as rebuild_error:
                logger.error(f"Failed to rebuild index: {str(rebuild_error)}")
                # Log detailed error information
                log_error(
                    f"Index rebuild failed: {str(rebuild_error)}",
                    include_traceback=True,
                    error_type="INDEX_REBUILD_FAILURE",
                    source="DocumentProcessor.search",
                    additional_data={
                        "query": query,
                        "original_error": str(rebuild_error),
                        "notification": notification
                    }
                )
                
                # Provide a more detailed error message with troubleshooting steps
                troubleshooting_steps = (
                    "\n\nTroubleshooting Steps:\n"
                    "1. Check if the index directory exists and is writable\n"
                    "2. Verify that the chunks data is not corrupted\n"
                    "3. Ensure sufficient disk space and memory\n"
                    "4. Try rebuilding the index manually with the 'rebuild_index' command\n"
                    "5. Check the logs for more detailed error information"
                )
                raise ValueError(f"Embedding dimensions mismatch and index rebuild failed: {str(rebuild_error)}. {troubleshooting_steps}")
        
        # Search index
        safe_top_k = min(top_k, len(self.chunks))  # Ensure top_k doesn't exceed available chunks
        if safe_top_k < top_k:
            logger.info(f"Adjusted top_k from {top_k} to {safe_top_k} based on available chunks")
            
        try:
            scores, indices = self.index.search(query_embedding, safe_top_k)
        except Exception as e:
            logger.error(f"Error during FAISS search: {str(e)}")
            raise ValueError(f"Search operation failed: {str(e)}")
        
        # Sanity check: Verify search results
        if indices.shape[0] == 0 or scores.shape[0] == 0:
            logger.warning("Search returned empty results")
            return []
            
        if indices.shape[1] != safe_top_k or scores.shape[1] != safe_top_k:
            logger.warning(f"Search returned unexpected dimensions: indices {indices.shape}, scores {scores.shape}, expected ({1}, {safe_top_k})")
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunks):  # Ensure index is valid
                try:
                    chunk = self.chunks[idx].copy() if isinstance(self.chunks[idx], dict) else {"text": self.chunks[idx]}
                    chunk["score"] = float(scores[0][i])  # Convert numpy float to Python float
                    # Ensure metadata exists
                    if "metadata" not in chunk:
                        chunk["metadata"] = {"source": "unknown"}
                    results.append(chunk)
                except (IndexError, KeyError, AttributeError) as e:
                    logger.warning(f"Error processing search result at index {idx}: {str(e)}")
                    # Create a minimal valid result
                    results.append({
                        "text": f"Error retrieving chunk {idx}",
                        "score": float(scores[0][i]),
                        "metadata": {"source": "unknown", "error": str(e)}
                    })
            else:
                logger.warning(f"Invalid index {idx} returned by search, valid range is 0-{len(self.chunks)-1}")
        
        # Sanity check: Verify we have results
        if len(results) == 0 and len(indices[0]) > 0:
            logger.error("Search returned indices but no valid results could be created")
        
        # Check if we need to use keyword fallback for zero-hit cases
        if not results and retrieval_config.enable_keyword_fallback:
            logger.info(f"Vector search returned no results, trying keyword fallback for query: '{query_preview}'")
            try:
                # Import enhanced retrieval for keyword fallback
                from utils.enhanced_retrieval import EnhancedRetrieval
                
                # Initialize enhanced retrieval with BM25 index
                enhanced_retriever = EnhancedRetrieval()
                
                # Build BM25 index from chunks if needed
                if not hasattr(enhanced_retriever, 'bm25_index') or enhanced_retriever.bm25_index is None:
                    enhanced_retriever.build_bm25_index(self.chunks)
                
                # Perform keyword fallback search
                fallback_results = enhanced_retriever.keyword_fallback_search(query, top_k=top_k)
                
                if fallback_results:
                    logger.info(f"Keyword fallback search returned {len(fallback_results)} results")
                    results = fallback_results
                    search_type = "keyword_fallback"
                    fallback_used = True
                else:
                    logger.warning(f"Keyword fallback search also returned no results for query: '{query_preview}'")
            except Exception as e:
                logger.error(f"Error in keyword fallback search: {str(e)}")
                # Continue with empty results if fallback fails
        
        # Log search metrics
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.4f}s, returned {len(results)} results")
        
        # Log query metrics if enabled
        if performance_config.log_query_metrics:
            query_logger.log_query(
                query=query,
                latency=search_time,
                hit_count=len(results),
                metadata={
                    "search_type": search_type,
                    "top_k": top_k,
                    "fallback_used": fallback_used
                }
            )
            
            # Log zero-hit queries separately if enabled
            if len(results) == 0 and performance_config.log_zero_hit_queries:
                query_logger.log_zero_hit_query(query, {"search_type": search_type, "top_k": top_k})
        
        return results
    
    def clear_index(self, index_dir: Optional[str] = None) -> None:
        """
        Clear the index and all cached data
        
        Args:
            index_dir: Directory containing the index files to clear (defaults to config value)
        """
        # Clear index and texts
        self.index = None
        self.texts = []
        self.chunks = []  # Keep alias in sync
        self.processed_files = set()
        
        # Clear cache
        self.chunk_cache = {}
        
        # Reset metrics
        self.total_chunks = 0
        
        # Use configured index path if not specified
        index_dir = index_dir or retrieval_config.index_path
        
        # Resolve the path to ensure consistency
        index_dir = resolve_path(index_dir, create_dir=True)
        
        # Clear index files if they exist
        logger.debug(f"Resolved index directory for clearing: {index_dir}")
        index_file = os.path.join(index_dir, "index.faiss")
        texts_file = os.path.join(index_dir, "texts.npy")
        metadata_file = os.path.join(index_dir, "metadata.json")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(index_dir, exist_ok=True)
            
            # Remove files if they exist
            for file_path in [index_file, texts_file, metadata_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed index file: {file_path}")
        except Exception as e:
            logger.error(f"Error clearing index files: {str(e)}")
            log_error(f"Error clearing index files: {str(e)}", include_traceback=True)
            raise
        
        logger.info("Index cleared successfully")
    
    def save_index(self, index_dir: Optional[str] = None, dry_run: bool = False) -> None:
        """
        Save the FAISS index and chunks to disk using atomic file operations
        
        Args:
            index_dir: Directory to save the index and chunks (defaults to config value)
            dry_run: If True, skip saving to disk (preview only)
            
        Raises:
            ValueError: If no index has been built
            IOError: If there's an I/O error during file writing
            OSError: If there's an OS error during atomic operations
        """
        if self.index is None or not self.texts:
            raise ValueError("No index has been built. Process documents first.")
        
        # Use configured index path if not specified
        index_dir = index_dir or retrieval_config.index_path
        
        # Resolve the path to ensure consistency
        index_dir = resolve_path(index_dir, create_dir=True)
        
        # If in dry-run mode, log what would happen but don't save
        if dry_run:
            logger.info(f"[DRY RUN] Would save index with {len(self.chunks)} chunks to {index_dir}")
            logger.debug(f"Resolved index directory for saving: {index_dir}")
            return
        
        # Create directory if it doesn't exist
        logger.debug(f"Resolved index directory for saving: {index_dir}")
        os.makedirs(index_dir, exist_ok=True)
        
        # Create a lock file to indicate that an update is in progress
        logger.debug(f"Resolved index directory for saving: {index_dir}")
        lock_path = os.path.join(index_dir, "index.lock")
        with open(lock_path, "w") as lock_file:
            lock_file.write(f"Update in progress: {time.time()}")
        
        logger.debug(f"Resolved index directory for saving: {index_dir}")
        # Define paths for files and their temporary versions
        index_path = os.path.join(index_dir, "index.faiss")
        temp_index_path = os.path.join(index_dir, "index.faiss.tmp")
        texts_path = os.path.join(index_dir, "texts.npy")
        temp_texts_path = os.path.join(index_dir, "texts.npy.tmp")
        metadata_path = os.path.join(index_dir, "metadata.json")
        temp_metadata_path = os.path.join(index_dir, "metadata.json.tmp")
        
        # Check if index is on GPU before converting
        if self.gpu_available and hasattr(self.index, 'getDevice') and self.index.getDevice() >= 0:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
            
        # Defensive check before saving
        if not self.texts or len(self.texts) == 0:
            logger.error("Attempting to save index with empty texts list!")
        else:
            logger.info(f"Preparing to save {len(self.texts)} chunks to {texts_path}")

        # Write index to temporary file
        try:
            faiss.write_index(cpu_index, temp_index_path)
        except IOError as e:
            raise IOError(f"Failed to write index to temporary file: {str(e)}") from e
        
        # Save chunks to temporary file
        try:
            np.save(temp_texts_path, np.array(self.texts, dtype=object))
            logger.info(f"Saved {len(self.texts)} chunks to {temp_texts_path}")
        except IOError as e:
            raise IOError(f"Failed to save texts to temporary file: {str(e)}") from e
        
        # Prepare metadata
        metadata = {
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_dim": self.embedding_dim,
            "processed_files": list(self.processed_files),
            "last_updated": time.time(),
            "index_size": self.index.ntotal if hasattr(self.index, 'ntotal') else 0
        }
        
        # Save metadata to temporary file
        try:
            with open(temp_metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except IOError as e:
            raise IOError(f"Failed to write metadata to temporary file: {str(e)}") from e
        
        # Use atomic replace operations to ensure consistency
        try:
            # Atomic replace for index file
            os.replace(temp_index_path, index_path)
            
            # Atomic replace for texts file
            os.replace(temp_texts_path, texts_path)
            
            # Atomic replace for metadata file
            os.replace(temp_metadata_path, metadata_path)
            
            logger.info(f"Saved index and {len(self.texts)} chunks to {index_dir} using atomic operations")
        except OSError as e:
            error_msg = f"Error during atomic file operations: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg)
            raise OSError(error_msg) from e
        finally:
            # Remove the lock file regardless of success or failure
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except Exception as e:
                    logger.warning(f"Failed to remove lock file: {str(e)}")
    
    # This method is already defined above
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string
        
        Args:
            query: Query string to embed
            
        Returns:
            NumPy array of query embedding
        """
        return self.embedding_model.encode([query])
        
    def create_embeddings(self, texts: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Create embeddings for a list of text chunks
        
        Args:
            texts: List of text chunks (dictionaries with 'text' key)
            
        Returns:
            Tuple of (list of text chunks, numpy array of embeddings)
        """
        # Extract text from chunks if they are dictionaries
        text_content = []
        for chunk in texts:
            if isinstance(chunk, dict) and "text" in chunk:
                text_content.append(chunk["text"])
            elif isinstance(chunk, str):
                text_content.append(chunk)
            else:
                logger.warning(f"Skipping invalid chunk format: {type(chunk)}")
        
        # Create embeddings using batch processing for efficiency
        embeddings = self.batch_embed_chunks(texts)
        
        return texts, embeddings
    
    def load_index(self, index_dir: Optional[str] = None) -> None:
        """
        Load a FAISS index and chunks from disk
        
        Args:
            index_dir: Directory containing the index and chunks (defaults to config value)
            
        Raises:
            FileNotFoundError: If index files are not found
            ValueError: If index parameters don't match
        """
        # Use configured index path if not specified
        index_dir = index_dir or retrieval_config.index_path
        
        # Resolve the path to ensure consistency
        index_dir = resolve_path(index_dir, create_dir=False)
        
        # Check if index files exist
        index_path = os.path.join(index_dir, "index.faiss")
        texts_file = os.path.join(index_dir, "texts.npy")
        metadata_file = os.path.join(index_dir, "metadata.json")
        
        # Check for required files
        logger.debug(f"Resolved index directory for loading: {index_dir}")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at {index_path}")
        if not os.path.exists(texts_file):
            raise FileNotFoundError(f"Texts file not found at {texts_file}")
        
        # Load metadata first to get embedding dimension if available
        embedding_dim = self.embedding_dim  # Default to current model's dimension
        if os.path.exists(metadata_file):
            try:
                logger.debug(f"Resolved index directory for loading: {index_dir}")
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if "embedding_dim" in metadata:
                        embedding_dim = metadata["embedding_dim"]
                        logger.info(f"Using embedding dimension {embedding_dim} from metadata")
            except Exception as e:
                logger.warning(f"Error reading metadata: {str(e)}")
        
        # Load index
        try:
            self.index = faiss.read_index(index_path)
            logger.info(f"Successfully loaded FAISS index from {index_path}")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            # Create a new index with the current model's dimension
            logger.info(f"Creating new index with dimension {self.embedding_dim} due to load error")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            # We'll need to re-embed the texts with the current model
            logger.warning("Index could not be loaded - you may need to reprocess your documents")
        
        # Verify index dimensions match embedding model
        if self.index is not None and hasattr(self.index, 'd'):
            index_dim = self.index.d
            if index_dim != self.embedding_dim:
                logger.warning(f"Index dimension ({index_dim}) doesn't match embedding model dimension ({self.embedding_dim})")
                # Create a new index with the correct dimensions
                logger.info(f"Creating new index with dimension {self.embedding_dim}")
                new_index = faiss.IndexFlatL2(self.embedding_dim)
                self.index = new_index
                # We'll need to re-embed the texts with the current model
                logger.warning("Index dimensions mismatch - you may need to reprocess your documents")
        else:
            # If index is None or doesn't have 'd' attribute, create a new one
            logger.warning("Index is not properly initialized")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Created new index with dimension {self.embedding_dim}")
        # Use GPU if available and enabled
        if self.gpu_available and performance_config.use_gpu_for_faiss and self.index is not None:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU for loaded FAISS index")
            except Exception as e:
                logger.warning(f"Failed to use GPU for loaded FAISS index: {str(e)}")
        
        # Load texts
        try:
            texts_array = np.load(texts_file, allow_pickle=True)
            if texts_array is not None and hasattr(texts_array, 'tolist'):
                self.texts = texts_array.tolist()
                # Ensure texts is not None and is a list
                if self.texts is None:
                    self.texts = []
                # Ensure chunks alias is updated
                self.chunks = self.texts
                logger.info(f"Loaded {len(self.texts)} chunks from {texts_file}")
                
                # If we have texts and a new index, rebuild the index with the current embeddings
                if self.texts and len(self.texts) > 0 and self.index is not None and self.index.ntotal == 0:
                    try:
                        logger.info("Rebuilding index with loaded texts")
                        embeddings = self.batch_embed_chunks(self.texts)
                        self.index.add(embeddings)
                        logger.info(f"Rebuilt index with {len(self.texts)} chunks")
                    except Exception as e:
                        logger.error(f"Error rebuilding index: {str(e)}")
            else:
                logger.warning(f"Invalid texts array loaded from {texts_file}")
                self.texts = []
                self.chunks = []
        except Exception as e:
            logger.error(f"Error loading texts: {str(e)}")
            self.texts = []
            self.chunks = []
            raise ValueError(f"Failed to load texts: {str(e)}")
        
        # Load metadata if available
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    # Update processed files if available
                    if "processed_files" in metadata:
                        self.processed_files = set(metadata["processed_files"])
                    logger.info(f"Loaded metadata from {metadata_file}")
            except Exception as e:
                logger.warning(f"Error loading metadata: {str(e)}")
        
        logger.info(f"Successfully loaded index with {len(self.texts)} chunks from {index_dir}")
        
    def _compute_document_hash(self, file_path: str) -> str:
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
            return self._compute_fallback_hash(file_path)
        except Exception as e:
            logger.warning(f"Error computing hash for {file_path}: {str(e)}")
            # Return a fallback hash based on filename and modification time
            return self._compute_fallback_hash(file_path)
    
    def _compute_fallback_hash(self, file_path: str) -> str:
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
    
    @with_retry(max_attempts=3, backoff_factor=1.0, exception_types=(IOError, OSError))  # Add retry for transient I/O issues
    def _update_metadata_with_hash(self, file_path: str, doc_hash: str, chunk_count: int) -> None:
        """
        Update metadata.json with document hash information
        
        Args:
            file_path: Path to the document file
            doc_hash: Hash of the document
            chunk_count: Number of chunks extracted from the document
            
        Raises:
            IOError: If there's an I/O error during file operations
            OSError: If there's an OS error during file operations
            json.JSONDecodeError: If the metadata file is corrupted
        """
        import fcntl  # Import for file locking
        import threading
        import errno
        
        # Create paths for metadata files
        metadata_dir = resolve_path(retrieval_config.index_path, create_dir=True)
        metadata_path = os.path.join(metadata_dir, "metadata.json")
        temp_path = os.path.join(metadata_dir, f"metadata.json.tmp.{threading.get_ident()}")
        lock_path = os.path.join(metadata_dir, "metadata.lock")
                
        # Log resolved paths for debugging
        logger.debug(f"Resolved metadata directory: {metadata_dir}")
        logger.debug(f"Metadata file path: {metadata_path}")
        
        # Create metadata directory if it doesn't exist
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Use proper file locking to prevent race conditions
        metadata_file = None
        lock_acquired = False
        
        try:
            # Create a lock file to indicate an update is in progress (additional safety)
            with open(lock_path, "w") as lock_file:
                lock_file.write(f"Update in progress: {time.time()} by thread {threading.get_ident()}")
            
            # Open or create the metadata file with explicit mode for better compatibility
            metadata_file = open(metadata_path, "a+")
            
            # Acquire an exclusive lock on the file with timeout
            start_time = time.time()
            max_wait_time = 10.0  # Maximum time to wait for lock in seconds
            
            while time.time() - start_time < max_wait_time:
                try:
                    # Try non-blocking lock first
                    fcntl.flock(metadata_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True
                    break
                except IOError as e:
                    # If lock is held by another process, wait a bit and retry
                    if e.errno == errno.EAGAIN or e.errno == errno.EACCES:
                        logger.debug(f"Waiting for metadata file lock (thread {threading.get_ident()})")
                        time.sleep(0.1)
                    else:
                        # Re-raise for other IOErrors
                        raise
            
            # If we couldn't acquire the lock after waiting, use blocking lock as last resort
            if not lock_acquired:
                logger.warning(f"Using blocking lock for metadata file after waiting {max_wait_time} seconds")
                fcntl.flock(metadata_file.fileno(), fcntl.LOCK_EX)
                lock_acquired = True
            
            # Seek to the beginning of the file to read its content
            metadata_file.seek(0)
            content = metadata_file.read()
            
            # Parse existing metadata or create new if empty/invalid
            if content.strip():
                try:
                    metadata = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Corrupted metadata file: {str(e)}")
                    # Backup the corrupted file for debugging
                    backup_path = f"{metadata_path}.corrupted.{int(time.time())}"
                    try:
                        with open(backup_path, "w") as backup_file:
                            backup_file.write(content)
                        logger.info(f"Backed up corrupted metadata to {backup_path}")
                    except Exception as backup_error:
                        logger.warning(f"Failed to backup corrupted metadata: {str(backup_error)}")
                    
                    # Start with fresh metadata
                    metadata = {}
            else:
                metadata = {}
            
            # Initialize document_hashes if not present
            if "document_hashes" not in metadata:
                metadata["document_hashes"] = {}
            
            # Add or update document hash with enhanced metadata
            metadata["document_hashes"][file_path] = {
                "hash": doc_hash,
                "last_processed": time.time(),
                "chunk_count": chunk_count,
                "file_size": os.path.getsize(file_path),
                "file_name": os.path.basename(file_path),
                "file_extension": os.path.splitext(file_path)[1].lower(),
                "last_modified": os.path.getmtime(file_path),
                "processing_timestamp": time.time(),
                "updated_by": f"thread-{threading.get_ident()}"
            }
            
            # Write to a thread-specific temporary file first
            temp_file = open(temp_path, "w")
            json.dump(metadata, temp_file, indent=2)
            temp_file.flush()
            
            # Ensure the temporary file is fully written to disk
            os.fsync(temp_file.fileno())
            temp_file.close()
            
            # Use atomic replace to update the original file
            # This is done while still holding the lock on the original file
            os.replace(temp_path, metadata_path)
            logger.info(f"Updated metadata for {os.path.basename(file_path)} with hash {doc_hash}")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in metadata file: {str(e)}")
            log_error(f"JSON decode error in metadata file: {str(e)}", include_traceback=True)
            raise
            
        except (IOError, OSError) as e:
            logger.error(f"Error accessing metadata file: {str(e)}")
            log_error(f"Error accessing metadata file: {str(e)}", include_traceback=True)
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error updating metadata with hash: {str(e)}")
            log_error(f"Unexpected error updating metadata with hash: {str(e)}", include_traceback=True)
            raise
            
        finally:
            # Clean up resources
            if metadata_file is not None:
                if lock_acquired:
                    try:
                        # Release the lock explicitly
                        fcntl.flock(metadata_file.fileno(), fcntl.LOCK_UN)
                        logger.debug(f"Released metadata file lock (thread {threading.get_ident()})")
                    except Exception as unlock_error:
                        logger.warning(f"Error releasing file lock: {str(unlock_error)}")
                
                # Close the file
                try:
                    metadata_file.close()
                except Exception as close_error:
                    logger.warning(f"Error closing metadata file: {str(close_error)}")
            
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file: {str(cleanup_error)}")
            
            # Remove lock file
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except Exception as lock_cleanup_error:
                    logger.warning(f"Failed to remove lock file: {str(lock_cleanup_error)}")
    
    @with_timing(threshold=1.0)
    def _semantic_deduplication(self, chunks: List[Dict[str, Any]], similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
        """
        Perform semantic deduplication on chunks using optimized approximate nearest neighbors
        with efficient memory usage and improved processing speed.
        
        Args:
            chunks: List of document chunks with metadata
            similarity_threshold: Threshold for considering chunks as duplicates (0.0 to 1.0)
            
        Returns:
            List of deduplicated chunks
        """
        # WorkApp2 Progress: Last completed Doc-3
        # Next pending Doc-5 (Hardcoded File Paths)
        # Doc-4 - Completed (Inefficient Semantic Deduplication)
        
        if not chunks:
            return []
        
        # Early return if only one chunk
        if len(chunks) == 1:
            return chunks
            
        # Extract text from chunks for embedding
        texts = []
        valid_chunk_indices = []
        
        # First pass: identify valid chunks and their texts
        for i, chunk in enumerate(chunks):
            if "content" in chunk and chunk["content"]:
                texts.append(chunk["content"])
                valid_chunk_indices.append(i)
            elif "text" in chunk and chunk["text"]:
                texts.append(chunk["text"])
                valid_chunk_indices.append(i)
            else:
                # Skip invalid chunks instead of using empty strings
                logger.warning(f"Chunk without text content found during deduplication: {chunk.get('id', 'unknown')}")
        
        # Early return if no valid chunks
        if not texts:
            logger.warning("No valid text content found in chunks for deduplication")
            return chunks
        
        # Track processing time
        start_time = time.time()
        
        try:
            # Determine optimal parameters based on dataset size
            chunk_count = len(texts)
            
            # Dynamic batch sizing based on chunk count and available memory
            if chunk_count > 50000:
                batch_size = 32  # Very small batches for extremely large datasets
            elif chunk_count > 20000:
                batch_size = 64
            elif chunk_count > 10000:
                batch_size = 128
            elif chunk_count > 5000:
                batch_size = 256
            else:
                batch_size = 512
            
            logger.info(f"Deduplicating {chunk_count} chunks with batch size {batch_size}")
            
            # Create embeddings in batches with progress tracking
            embeddings_list = []
            total_batches = (chunk_count + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, chunk_count)
                batch_texts = texts[start_idx:end_idx]
                
                try:
                    # Time the embedding process
                    batch_start_time = time.time()
                    batch_embeddings = self.embedding_model.encode(batch_texts)
                    batch_time = time.time() - batch_start_time
                    
                    # Log progress for large datasets
                    if total_batches > 5:
                        logger.info(f"Embedding batch {batch_idx+1}/{total_batches} completed in {batch_time:.2f}s")
                    
                    embeddings_list.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error embedding batch {batch_idx+1}: {str(e)}")
                    # Skip this batch but continue with others
                    log_error(f"Skipping embedding batch {batch_idx+1} due to error: {str(e)}")
                    # Create empty embeddings as placeholder to maintain index alignment
                    if batch_idx == 0 and not embeddings_list:
                        # If this is the first batch, we need to know the embedding dimension
                        # Try with a single example to get the dimension
                        try:
                            sample_embedding = self.embedding_model.encode(["sample text"])
                            empty_batch = np.zeros((len(batch_texts), sample_embedding.shape[1]), dtype=np.float32)
                            embeddings_list.append(empty_batch)
                        except Exception:
                            # If even that fails, we can't continue
                            raise ValueError("Cannot determine embedding dimension for deduplication")
                    elif embeddings_list:
                        # Use the dimension from previous batches
                        empty_batch = np.zeros((len(batch_texts), embeddings_list[0].shape[1]), dtype=np.float32)
                        embeddings_list.append(empty_batch)
            
            # Concatenate all embeddings
            if not embeddings_list:
                logger.error("No embeddings were generated for deduplication")
                return chunks
                
            embeddings_np = np.vstack(embeddings_list).astype('float32')
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings_np, inplace=True)
            
            # Set of indices to keep (start with all valid chunks)
            indices_to_keep = set(valid_chunk_indices)
            
            # Choose deduplication strategy based on dataset size
            if chunk_count > 50000:
                # For extremely large datasets: use hierarchical clustering with HNSW index
                logger.info(f"Using hierarchical clustering with HNSW for very large dataset ({chunk_count} chunks)")
                self._dedup_with_hnsw(embeddings_np, valid_chunk_indices, chunks, indices_to_keep, similarity_threshold)
            elif chunk_count > 10000:
                # For large datasets: use optimized IVF index with PQ compression
                logger.info(f"Using IVF with PQ compression for large dataset ({chunk_count} chunks)")
                self._dedup_with_ivfpq(embeddings_np, valid_chunk_indices, chunks, indices_to_keep, similarity_threshold)
            else:
                # For smaller datasets: use standard IVF index
                logger.info(f"Using standard IVF index for dataset ({chunk_count} chunks)")
                self._dedup_with_ivf(embeddings_np, valid_chunk_indices, chunks, indices_to_keep, similarity_threshold)
            
            # Return deduplicated chunks
            deduplicated_chunks = [chunks[i] for i in sorted(indices_to_keep)]
            dedup_time = time.time() - start_time
            logger.info(f"Semantic deduplication completed in {dedup_time:.2f}s: {len(chunks)} chunks -> {len(deduplicated_chunks)} chunks")
            
            # Log detailed metrics
            removed_count = len(chunks) - len(deduplicated_chunks)
            removal_percentage = (removed_count / len(chunks)) * 100 if len(chunks) > 0 else 0
            logger.info(f"Deduplication removed {removed_count} chunks ({removal_percentage:.1f}%)")
            
            return deduplicated_chunks
            
        except Exception as e:
            logger.error(f"Error during semantic deduplication: {str(e)}")
            log_error(f"Error during semantic deduplication: {str(e)}", include_traceback=True)
            # Return original chunks if deduplication fails
            return chunks
            
        finally:
            # Log deduplication metrics
            dedup_time = time.time() - start_time
            logger.debug(f"Semantic deduplication processing time: {dedup_time:.2f}s")
    
    def _process_search_results(self, batch_start: int, similarities: np.ndarray, neighbors: np.ndarray,
                                 valid_indices: List[int], chunks: List[Dict[str, Any]],
                                 indices_to_keep: Set[int], similarity_threshold: float) -> None:
        """
        Process the search results from HNSW index to identify and remove duplicate chunks.

        Args:
            batch_start: Starting index of the current batch
            similarities: Matrix of similarity scores
            neighbors: Matrix of neighbor indices
            valid_indices: List mapping embedding indices to chunk indices
            chunks: Original chunks list
            indices_to_keep: Set of indices to keep (modified in-place)
            similarity_threshold: Similarity threshold for deduplication
        """
        try:
            num_queries = similarities.shape[0]

            for i in range(num_queries):
                original_index = valid_indices[batch_start + i]
                if original_index not in indices_to_keep:
                    continue  # Skip if already marked for removal

                for j in range(1, neighbors.shape[1]):  # Start from 1 to skip self-comparison
                    neighbor_index_in_batch = neighbors[i, j]
                    if neighbor_index_in_batch == -1:
                        break  # No more neighbors to process

                    neighbor_index = valid_indices[neighbor_index_in_batch]

                    if neighbor_index in indices_to_keep:
                        similarity = similarities[i, j]
                        if similarity > similarity_threshold:
                            # Mark the neighbor for removal
                            indices_to_keep.discard(neighbor_index)
                            logger.debug(f"Duplicate found: Chunk {neighbor_index} is similar to chunk {original_index} (similarity: {similarity:.4f})")

        except Exception as e:
            logger.error(f"Error processing search results: {str(e)}")
            log_error(f"Error processing search results: {str(e)}", include_traceback=True)

    def _dedup_with_ivfpq(self, embeddings: np.ndarray, valid_indices: List[int], chunks: List[Dict[str, Any]],
                         indices_to_keep: Set[int], similarity_threshold: float) -> None:
        """
        Deduplicate using an IVF index with Product Quantization (PQ) compression.
        Optimized for large datasets to balance speed and memory usage.

        Args:
            embeddings: Normalized embeddings array
            valid_indices: List mapping embedding indices to chunk indices
            chunks: Original chunks list
            indices_to_keep: Set of indices to keep (modified in-place)
            similarity_threshold: Similarity threshold for deduplication
        """
        try:
            dim = embeddings.shape[1]
            nlist = 100  # Number of Voronoi cells
            m = 16  # Number of subvectors for PQ
            nbits = 8  # Bits per subvector

            # Define the product quantizer
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

            # Train the index
            index.train(embeddings)
            index.add(embeddings)

            # Process in batches to avoid memory issues
            batch_size = 1000
            k = 10  # Number of neighbors to retrieve

            for batch_start in range(0, len(embeddings), batch_size):
                batch_end = min(batch_start + batch_size, len(embeddings))
                batch_embeddings = embeddings[batch_start:batch_end]

                # Search for similar chunks
                index.nprobe = 10  # Adjust nprobe for better recall
                similarities, neighbors = index.search(batch_embeddings, k=k)

                # Process results
                self._process_search_results(batch_start, similarities, neighbors, valid_indices,
                                         chunks, indices_to_keep, similarity_threshold)

        except Exception as e:
            logger.error(f"Error in IVF with PQ deduplication: {str(e)}")
            log_error(f"IVF with PQ deduplication failed: {str(e)}", include_traceback=True)

    def _dedup_with_ivf(self, embeddings: np.ndarray, valid_indices: List[int], chunks: List[Dict[str, Any]],
                      indices_to_keep: Set[int], similarity_threshold: float) -> None:
        """
        Deduplicate using an Inverted File (IVF) index.
        Suitable for medium-sized datasets where speed is important.

        Args:
            embeddings: Normalized embeddings array
            valid_indices: List mapping embedding indices to chunk indices
            chunks: Original chunks list
            indices_to_keep: Set of indices to keep (modified in-place)
            similarity_threshold: Similarity threshold for deduplication
        """
        try:
            dim = embeddings.shape[1]
            nlist = 100  # Number of Voronoi cells

            # Define the IVF quantizer
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train the index
            index.train(embeddings)
            index.add(embeddings)

            # Process in batches to avoid memory issues
            batch_size = 1000
            k = 10  # Number of neighbors to retrieve

            for batch_start in range(0, len(embeddings), batch_size):
                batch_end = min(batch_start + batch_size, len(embeddings))
                batch_embeddings = embeddings[batch_start:batch_end]

                # Search for similar chunks
                index.nprobe = 10  # Adjust nprobe for better recall
                similarities, neighbors = index.search(batch_embeddings, k=k)

                # Process results
                self._process_search_results(batch_start, similarities, neighbors, valid_indices,
                                         chunks, indices_to_keep, similarity_threshold)

        except Exception as e:
            logger.error(f"Error in IVF deduplication: {str(e)}")
            log_error(f"IVF deduplication failed: {str(e)}", include_traceback=True)
            
    def _dedup_with_hnsw(self, embeddings: np.ndarray, valid_indices: List[int], chunks: List[Dict[str, Any]], 
                         indices_to_keep: Set[int], similarity_threshold: float) -> None:
        """
        Deduplicate using Hierarchical Navigable Small World (HNSW) graph-based index.
        Optimized for very large datasets with millions of vectors.
        
        Args:
            embeddings: Normalized embeddings array
            valid_indices: List mapping embedding indices to chunk indices
            chunks: Original chunks list
            indices_to_keep: Set of indices to keep (modified in-place)
            similarity_threshold: Similarity threshold for deduplication
        """
        try:
            dim = embeddings.shape[1]
            
            # Create HNSW index with optimized parameters
            # M: number of connections per layer (higher = better recall but more memory)
            # efConstruction: build-time accuracy parameter (higher = better recall but slower build)
            M = 16  # Good balance between accuracy and memory usage
            ef_construction = 128  # Higher values give better recall but slower build
            
            index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = 64  # Search-time accuracy parameter
            
            # Add vectors to index
            index.add(embeddings)
            
            # Process in batches to avoid memory issues
            batch_size = 1000
            k = 10  # Number of neighbors to retrieve
            
            for batch_start in range(0, len(embeddings), batch_size):
                batch_end = min(batch_start + batch_size, len(embeddings))
                batch_embeddings = embeddings[batch_start:batch_end]
                
                # Search for similar chunks
                similarities, neighbors = index.search(batch_embeddings, k=k)
                
                # Process results
                self._process_search_results(batch_start, similarities, neighbors, valid_indices, 
                                            chunks, indices_to_keep, similarity_threshold)
                
        except Exception as e:
            logger.error(f"Error in HNSW deduplication: {str(e)}")
            log_error(f"HNSW deduplication failed: {str(e)}", include_traceback=True)

    # ==== Helper Methods Added by LLM ====
    def _ensure_index_dimension(self) -> int:
        """Safely retrieve the FAISS index dimension 'd', or raise if missing."""
        idx_dim = getattr(self.index, 'd', None)
        if idx_dim is None:
            logging.error("FAISS index has no attribute 'd'; cannot determine vector dimension.")
            raise AttributeError("Missing index dimension 'd'")
        return idx_dim

    def _normalize_embeddings(self, embeddings_np: np.ndarray) -> None:
        """Normalize embeddings in-place using FAISS normalize_L2."""
        faiss.normalize_L2(embeddings_np, embeddings_np)

    def _search_index(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform a FAISS search with explicit parameter names."""
        distances, labels = self.index.search(query_embedding, top_k)
        return distances, labels