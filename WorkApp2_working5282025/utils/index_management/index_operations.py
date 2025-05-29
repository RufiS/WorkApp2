# Index operations utilities
import os
import time
import json
import logging
import fcntl
import numpy as np
import faiss
from typing import Tuple, Any, Optional, Dict, List
from functools import lru_cache

from utils.config_unified import app_config, retrieval_config, performance_config, resolve_path
from utils.index_management.index_health import check_index_health

# Setup logging
logger = logging.getLogger(__name__)

# Global cache for index and texts
_index_cache = {}
_texts_cache = {}
_last_modified_time = {}

def save_index(index: Any, texts: list, gpu_available: bool = False, index_path: Optional[str] = None, dry_run: bool = False) -> None:
    """
    Save the FAISS index and texts to disk
    
    Args:
        index: FAISS index to save
        texts: List of text chunks to save
        gpu_available: Whether GPU is available for FAISS
        index_path: Path to save the index, defaults to app_config.index_path
        dry_run: If True, skip saving to disk (preview only)
    """
    if index_path is None:
        index_path = retrieval_config.index_path
        
    # Resolve the path
    resolved_index_path = resolve_path(index_path, create_dir=not dry_run)
        
    # If in dry-run mode, log what would happen but don't save
    if dry_run:
        logger.info(f"[DRY RUN] Would save index with {len(texts)} chunks to {resolved_index_path}")
        # Update cache even in dry-run mode for in-memory operations
        cache_key = resolved_index_path
        _index_cache[cache_key] = index
        _texts_cache[cache_key] = texts
        # Don't update _last_modified_time since we're not actually writing to disk
        return
        
    # Create directory if it doesn't exist
    os.makedirs(resolved_index_path, exist_ok=True)
    
    # Save metadata
    metadata = {
        "chunk_size": retrieval_config.chunk_size,
        "chunk_overlap": retrieval_config.chunk_overlap,
        "created_at": time.time(),
        "embedder": retrieval_config.embedding_model
    }
    with open(os.path.join(resolved_index_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    # Save chunk information to a file in the current_index folder
    chunks_dir = resolve_path("current_index", create_dir=True)
    if chunks_dir and texts and len(texts) > 0:
        try:
            os.makedirs(chunks_dir, exist_ok=True)
            with open(os.path.join(chunks_dir, "chunks.txt"), "w") as f:
                for i, chunk in enumerate(texts):
                    f.write(f"Chunk {i}:\n{chunk}\n\n")
            logger.info(f"Saved chunks to {os.path.join(chunks_dir, 'chunks.txt')}")
        except Exception as e:
            logger.warning(f"Failed to save chunks to {os.path.join(chunks_dir, 'chunks.txt')}: {str(e)}")
    
    # Ensure the index is a CPU index before saving
    if gpu_available:
        cpu_idx = faiss.index_gpu_to_cpu(index)
    else:
        cpu_idx = index
    
    faiss.write_index(cpu_idx, os.path.join(resolved_index_path, "index.faiss"))
    np.save(os.path.join(resolved_index_path, "texts.npy"), np.array(texts, dtype=object), allow_pickle=True)
    logger.info(f"Index saved to {resolved_index_path}")
    
    # Update cache
    cache_key = resolved_index_path
    _index_cache[cache_key] = index
    _texts_cache[cache_key] = texts
    _last_modified_time[cache_key] = os.path.getmtime(os.path.join(resolved_index_path, "index.faiss"))
    
def get_index_modified_time(index_path: Optional[str] = None) -> float:
    """
    Get the last modified time of the index file
    
    Args:
        index_path: Path to the index directory, defaults to app_config.index_path
        
    Returns:
        Last modified time as a float, or 0 if the file doesn't exist
    """
    if index_path is None:
        index_path = retrieval_config.index_path
        
    # Resolve the path
    resolved_index_path = resolve_path(index_path)
        
    index_file = os.path.join(resolved_index_path, "index.faiss")
    if os.path.exists(index_file):
        return os.path.getmtime(index_file)
    return 0

def load_index(gpu_available: bool = False, index_path: Optional[str] = None, force_reload: bool = False) -> Tuple[Any, list, int, str]:
    """
    Load the FAISS index from disk with caching
    
    Args:
        gpu_available: Whether GPU is available for FAISS
        index_path: Path to load the index from, defaults to app_config.index_path
        force_reload: Whether to force reload from disk even if cached
        
    Returns:
        Tuple of (index, texts, num_chunks, status_message)
        
    Raises:
        RuntimeError: If loading fails
    """
    if index_path is None:
        index_path = retrieval_config.index_path
        
    # Resolve the path
    resolved_index_path = resolve_path(index_path)
        
    # Check if index files exist
    index_file = os.path.join(resolved_index_path, "index.faiss")
    texts_file = os.path.join(resolved_index_path, "texts.npy")
    metadata_file = os.path.join(resolved_index_path, "metadata.json")
    
    if not os.path.exists(index_file) or not os.path.exists(texts_file):
        logger.warning(f"Index files not found at {resolved_index_path}")
        # Get embedding dimension from metadata if available
        dimension = 768  # Default dimension
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if 'embedding_dim' in metadata:
                        dimension = metadata['embedding_dim']
                        logger.info(f"Using embedding dimension {dimension} from metadata")
            except Exception as e:
                logger.warning(f"Error reading metadata file: {str(e)}")
        
        # Create empty index structure
        index = faiss.IndexFlatL2(dimension)
        texts = []
        return index, texts, 0, "No index found. Created empty index."
        
    cache_key = resolved_index_path
    current_mtime = get_index_modified_time(index_path)
    
    # Check if we can use the cached version
    if not force_reload and cache_key in _index_cache and cache_key in _texts_cache:
        # Check if the file has been modified since we cached it
        if cache_key in _last_modified_time and _last_modified_time[cache_key] >= current_mtime:
            logger.info(f"Using cached index for {resolved_index_path}")
            index = _index_cache[cache_key]
            texts = _texts_cache[cache_key]
            return index, texts, len(texts), f"Loaded cached index with {len(texts)} chunks."
    
    # If we get here, we need to load from disk
    return _load_index_from_disk(gpu_available, index_path)

def _load_index_from_disk(gpu_available: bool = False, index_path: Optional[str] = None) -> Tuple[Any, list, int, str]:
    """
    Internal function to load the FAISS index from disk
    
    Args:
        gpu_available: Whether GPU is available for FAISS
        index_path: Path to load the index from, defaults to app_config.index_path
        
    Returns:
        Tuple of (index, texts, num_chunks, status_message)
        
    Raises:
        RuntimeError: If loading fails
    """
    if index_path is None:
        index_path = retrieval_config.index_path
        
    # Resolve the path
    resolved_index_path = resolve_path(index_path, create_dir=True)
        
    lock_file = os.path.join(resolved_index_path, "index.lock")
    lock_fd = None
    
    try:
        # Create lock file if it doesn't exist
        if not os.path.exists(lock_file):
            open(lock_file, 'w').close()
            
        # Acquire lock with timeout
        start_time = time.time()
        lock_fd = open(lock_file, 'r')
        
        # Try to acquire lock with timeout
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break  # Lock acquired
            except IOError:
                # Check timeout (5 seconds)
                if time.time() - start_time > 5:
                    raise TimeoutError("Timeout waiting for index lock")
                time.sleep(0.1)  # Wait before retrying
                
        # Load the index
        logger.info("Lock acquired, loading index")
        index_file = os.path.join(resolved_index_path, "index.faiss")
        texts_file = os.path.join(resolved_index_path, "texts.npy")
        
        # Check if index file exists
        if not os.path.exists(index_file):
            logger.warning(f"Index file not found at {index_file}")
            dimension = 768  # Default dimension
            index = faiss.IndexFlatL2(dimension)
            texts = []
            return index, texts, 0, "Index file not found. Created empty index."
        
        try:
            idx = faiss.read_index(index_file)
        except Exception as e:
            logger.error(f"Failed to read index file: {str(e)}", exc_info=True)
            dimension = 768  # Default dimension
            index = faiss.IndexFlatL2(dimension)
            texts = []
            return index, texts, 0, f"Failed to read index file: {str(e)}. Created empty index."
        
        # Try to load on GPU if available
        if gpu_available:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, idx)
                logger.info("Index loaded on GPU")
            except Exception as e:
                logger.warning(f"Failed to load index on GPU: {str(e)}. Using CPU index.")
                index = idx
        else:
            index = idx
            
        # Load the texts
        texts = []
        try:
            if os.path.exists(texts_file):
                try:
                    texts_array = np.load(texts_file, allow_pickle=True)
                    if texts_array is not None and hasattr(texts_array, 'tolist'):
                        texts = texts_array.tolist()
                    else:
                        logger.warning("Loaded texts array is invalid")
                        texts = []
                except Exception as e:
                    logger.error(f"Error loading texts array: {str(e)}", exc_info=True)
                    texts = []
                    
                if not texts or len(texts) == 0:
                    logger.warning("Loaded texts array is empty")
                    texts = []
                    logger.info("Initialized with empty texts array")
                else:
                    logger.info(f"Loaded index with {len(texts)} chunks")
                    
                    # Create chunks file if it doesn't exist
                    chunks_dir = resolve_path("current_index", create_dir=True)
                    chunks_path = os.path.join(chunks_dir, "chunks.txt")
                    if not os.path.exists(chunks_path) and texts and len(texts) > 0:
                        try:
                            os.makedirs(chunks_dir, exist_ok=True)
                            with open(chunks_path, "w") as f:
                                for i, chunk in enumerate(texts):
                                    f.write(f"Chunk {i}:\n{chunk}\n\n")
                            logger.info(f"Created chunks file at {chunks_path}")
                        except Exception as e:
                            logger.warning(f"Failed to create chunks file at {chunks_path}: {str(e)}")
            else:
                logger.warning(f"Texts file not found at {texts_file}")
                texts = []
        except Exception as e:
            logger.error(f"Error loading texts: {str(e)}", exc_info=True)
            texts = []
            logger.info("Initialized with empty texts array due to error")
            
        # Verify the index is properly loaded
        if index is None:
            logger.error("Index failed to load properly")
            dimension = 768  # Default dimension
            index = faiss.IndexFlatL2(dimension)
            texts = []
            return index, texts, 0, "Index failed to load properly. Created empty index."
            
        # Update cache
        cache_key = resolved_index_path
        _index_cache[cache_key] = index
        _texts_cache[cache_key] = texts
        _last_modified_time[cache_key] = get_index_modified_time(index_path)
            
        # Return appropriate message based on whether texts are empty or not
        if len(texts) == 0:
            return index, texts, 0, "Loaded index with empty texts array."
        else:
            return index, texts, len(texts), f"Loaded index with {len(texts)} chunks."
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}", exc_info=True)
        # Create an empty index instead of raising an error
        dimension = 768  # Default dimension
        index = faiss.IndexFlatL2(dimension)
        texts = []
        return index, texts, 0, f"Error loading index: {str(e)}. Created empty index."
    finally:
        # Release lock
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
                logger.info("Index lock released")
            except Exception as e:
                logger.warning(f"Error releasing index lock: {str(e)}")
                
def clear_index(rebuild_immediately: bool = True, embedder: Any = None, gpu_available: bool = False, index_path: Optional[str] = None, dry_run: bool = False) -> None:
    """
    Clear the index and texts from memory and disk
    
    Args:
        rebuild_immediately: If True, rebuild the index immediately after clearing
        embedder: Sentence transformer model for creating empty index
        gpu_available: Whether GPU is available for FAISS
        index_path: Path to the index directory, defaults to app_config.index_path
        dry_run: If True, skip disk operations (preview only)
        
    Raises:
        RuntimeError: If clearing fails
    """
    if index_path is None:
        index_path = retrieval_config.index_path
        
    # Resolve the path
    resolved_index_path = resolve_path(index_path)
        
    # If in dry-run mode, log what would happen but don't modify disk
    if dry_run:
        logger.info(f"[DRY RUN] Would clear index at {resolved_index_path}")
        # Clear the index from memory
        cache_key = resolved_index_path
        if cache_key in _index_cache:
            del _index_cache[cache_key]
        if cache_key in _texts_cache:
            del _texts_cache[cache_key]
        if cache_key in _last_modified_time:
            del _last_modified_time[cache_key]
            
        # Rebuild empty index in memory if requested
        if rebuild_immediately and embedder is not None:
            # Create empty index with same dimensions as embedder
            dimension = embedder.get_sentence_embedding_dimension()
            index = faiss.IndexFlatL2(dimension)
            # Update cache with empty index
            _index_cache[cache_key] = index
            _texts_cache[cache_key] = []
            logger.info(f"[DRY RUN] Empty index rebuilt in memory for {resolved_index_path}")
        return
        
    lock_file = os.path.join(resolved_index_path, "index.lock")
    lock_fd = None
    
    try:
        # Create lock file if it doesn't exist
        if not os.path.exists(lock_file):
            open(lock_file, 'w').close()
            
        # Acquire lock with timeout
        start_time = time.time()
        lock_fd = open(lock_file, 'r')
        
        # Try to acquire lock with timeout
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break  # Lock acquired
            except IOError:
                # Check timeout (5 seconds)
                if time.time() - start_time > 5:
                    raise TimeoutError("Timeout waiting for index lock")
                time.sleep(0.1)  # Wait before retrying
                
        # Clear the index from memory
        cache_key = resolved_index_path
        if cache_key in _index_cache:
            del _index_cache[cache_key]
        if cache_key in _texts_cache:
            del _texts_cache[cache_key]
        if cache_key in _last_modified_time:
            del _last_modified_time[cache_key]
            
        # Remove index files
        try:
            if os.path.exists(os.path.join(resolved_index_path, "index.faiss")):
                os.remove(os.path.join(resolved_index_path, "index.faiss"))
            if os.path.exists(os.path.join(resolved_index_path, "texts.npy")):
                os.remove(os.path.join(resolved_index_path, "texts.npy"))
            logger.info(f"Index files removed from {resolved_index_path}")
            
            # Rebuild empty index if requested
            if rebuild_immediately and embedder is not None:
                # Create empty index with same dimensions as embedder
                dimension = embedder.get_sentence_embedding_dimension()
                index = faiss.IndexFlatL2(dimension)
                save_index(index, [], gpu_available, index_path)
                logger.info(f"Empty index rebuilt at {resolved_index_path}")
        except Exception as e:
            logger.error(f"Error removing index files: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error clearing index: {str(e)}")
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error clearing index: {str(e)}")
    finally:
        # Release lock
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
                logger.info("Index lock released")
            except Exception as e:
                logger.warning(f"Error releasing index lock: {str(e)}")
                
def rebuild_index_if_needed(index_path: Optional[str] = None, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Check if index needs rebuilding and rebuild if necessary
    
    Args:
        index_path: Path to the index directory, defaults to app_config.index_path
        dry_run: If True, skip disk operations (preview only)
        
    Returns:
        Tuple of (was_rebuilt, message)
    """
    if index_path is None:
        index_path = retrieval_config.index_path
        
    # Resolve the path
    resolved_index_path = resolve_path(index_path)
        
    # Ensure the index directory exists
    if not dry_run:
        os.makedirs(resolved_index_path, exist_ok=True)
        
    is_healthy, health_message = check_index_health(index_path)
    if is_healthy:
        return False, "Index is healthy"
        
    # Index is not healthy, try to rebuild it
    try:
        # Clear the index but don't rebuild immediately
        clear_index(rebuild_immediately=False, index_path=index_path, dry_run=dry_run)
        
        # Create an empty index with default dimensions
        dimension = 768  # Default dimension
        index = faiss.IndexFlatL2(dimension)
        save_index(index, [], False, index_path, dry_run=dry_run)
        
        if dry_run:
            return True, "[DRY RUN] Index would be reset due to corruption. Documents would need to be re-uploaded."
        else:
            return True, "Index was corrupted and has been reset. Please upload documents to rebuild the index."
    except Exception as e:
        logger.error(f"{'[DRY RUN] ' if dry_run else ''}Failed to rebuild index: {str(e)}", exc_info=True)
        return False, f"{'[DRY RUN] ' if dry_run else ''}Failed to rebuild index: {str(e)}"
        
def index_exists(index_path: Optional[str] = None) -> bool:
    """
    Check if an index exists at the specified path
    
    Args:
        index_path: Path to the index directory, defaults to app_config.index_path
        
    Returns:
        True if both index.faiss and texts.npy exist, False otherwise
    """
    if index_path is None:
        index_path = retrieval_config.index_path
        
    # Resolve the path
    resolved_index_path = resolve_path(index_path)
        
    index_file = os.path.join(resolved_index_path, "index.faiss")
    texts_file = os.path.join(resolved_index_path, "texts.npy")
    
    return os.path.exists(index_file) and os.path.exists(texts_file)


@lru_cache(maxsize=1)
def get_index_stats(index_path: Optional[str] = None, texts: Optional[List[str]] = None, index: Any = None) -> Dict[str, Any]:
    """
    Get statistics about the index
    
    Args:
        index_path: Path to the index directory, defaults to app_config.index_path
        texts: Optional list of text chunks (to avoid loading from disk)
        index: Optional FAISS index (to avoid loading from disk)
        
    Returns:
        Dictionary with index statistics
    """
    if index_path is None:
        index_path = retrieval_config.index_path
        
    # Resolve the path
    resolved_index_path = resolve_path(index_path)
        
    stats = {
        "exists": index_exists(index_path),
        "size_bytes": 0,
        "num_chunks": 0,
        "created_at": None,
        "embedder": None,
        "chunk_size": retrieval_config.chunk_size,
        "chunk_overlap": retrieval_config.chunk_overlap,
        "is_healthy": False,
        "health_message": "Index not checked"
    }
    
    # Get file sizes if they exist
    index_file = os.path.join(resolved_index_path, "index.faiss")
    texts_file = os.path.join(resolved_index_path, "texts.npy")
    metadata_file = os.path.join(resolved_index_path, "metadata.json")
    
    if os.path.exists(index_file):
        stats["size_bytes"] += os.path.getsize(index_file)
        
    if os.path.exists(texts_file):
        stats["size_bytes"] += os.path.getsize(texts_file)
        
        # Try to get number of chunks from provided texts or load from disk
        if texts is not None:
            stats["num_chunks"] = len(texts) if texts else 0
        else:
            try:
                texts_array = np.load(texts_file, allow_pickle=True)
                if texts_array is not None and hasattr(texts_array, 'tolist'):
                    texts = texts_array.tolist()
                    stats["num_chunks"] = len(texts) if texts else 0
                else:
                    stats["num_chunks"] = 0
            except Exception:
                stats["num_chunks"] = 0
            
    # Try to get metadata
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                stats["created_at"] = metadata.get("created_at")
                stats["embedder"] = metadata.get("embedder")
                stats["chunk_size"] = metadata.get("chunk_size", retrieval_config.chunk_size)
                stats["chunk_overlap"] = metadata.get("chunk_overlap", retrieval_config.chunk_overlap)
        except Exception:
            pass
            
    # Check health
    stats["is_healthy"], stats["health_message"] = check_index_health(index_path)
    
    return stats


def get_saved_chunk_params(index_path: Optional[str] = None) -> Tuple[int, int]:
    """
    Extract chunk parameters from index metadata
    
    Args:
        index_path: Path to the index directory, defaults to app_config.index_path
        
    Returns:
        Tuple of (chunk_size, chunk_overlap)
    """
    if index_path is None:
        index_path = retrieval_config.index_path
        
    # Resolve the path
    resolved_index_path = resolve_path(index_path, create_dir=True)
        
    metadata_file = os.path.join(resolved_index_path, "metadata.json")
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                chunk_size = metadata.get("chunk_size", retrieval_config.chunk_size)
                chunk_overlap = metadata.get("chunk_overlap", retrieval_config.chunk_overlap)
                return (chunk_size, chunk_overlap)
        else:
            logger.warning(f"Metadata file not found at {metadata_file}, using default chunk parameters")
            return (retrieval_config.chunk_size, retrieval_config.chunk_overlap)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        logger.error(f"Error reading metadata file: {str(e)}", exc_info=True)
        return (retrieval_config.chunk_size, retrieval_config.chunk_overlap)
