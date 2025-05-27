# Index health checking utilities
import os
import json
import logging
import numpy as np
import faiss
from typing import Tuple, Dict, Any, Optional

from utils.config_unified import app_config, retrieval_config, resolve_path

# Setup logging
logger = logging.getLogger(__name__)

def check_index_health(index_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check if the index is healthy and usable
    
    Args:
        index_path: Path to the index directory, defaults to app_config.index_path
        
    Returns:
        Tuple of (is_healthy, message)
    """
    if index_path is None:
        index_path = retrieval_config.index_path
        
    # Resolve the path
    resolved_index_path = resolve_path(index_path)
        
    try:
        # Check if index files exist
        if not os.path.exists(resolved_index_path):
            return False, "Index directory does not exist"
            
        index_file = os.path.join(resolved_index_path, "index.faiss")
        texts_file = os.path.join(resolved_index_path, "texts.npy")
        metadata_file = os.path.join(resolved_index_path, "metadata.json")
        
        if not os.path.exists(index_file):
            return False, "Index file does not exist"
            
        if not os.path.exists(texts_file):
            return False, "Texts file does not exist"
            
        # Check file sizes
        if os.path.getsize(index_file) == 0:
            return False, "Index file is empty"
            
        if os.path.getsize(texts_file) == 0:
            return False, "Texts file is empty"
            
        # Try to load metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                
            # Check metadata version
            if metadata.get("version") is None:
                logger.warning("Metadata is missing version information")
                
            # Check chunk parameters
            if "chunk_size" not in metadata or "chunk_overlap" not in metadata:
                logger.warning("Metadata is missing chunk parameters")
                
        # Try to load a small part of the texts file to verify it's valid numpy array
        try:
            texts = np.load(texts_file, allow_pickle=True)
            if texts.size == 0:
                logger.warning("Texts file is empty but continuing")
        except Exception as e:
            return False, f"Failed to read texts file: {str(e)}"
                
        # Try to load the index file to verify it's a valid FAISS index
        try:
            faiss.read_index(index_file)
        except Exception as e:
            return False, f"Failed to read FAISS index: {str(e)}"
            
        return True, "Index is healthy"
    except Exception as e:
        return False, f"Error checking index health: {str(e)}"
        
def has_index(index_path: Optional[str] = None) -> bool:
    """
    Check if an index exists on disk and is valid
    
    Args:
        index_path: Path to the index directory, defaults to app_config.index_path
        
    Returns:
        True if index exists and appears valid, False otherwise
    """
    # Use the more comprehensive health check but only return the boolean result
    is_healthy, _ = check_index_health(index_path)
    return is_healthy
    
def get_index_stats(index_path: Optional[str] = None, texts: Optional[list] = None, index: Optional[Any] = None) -> Dict[str, Any]:
    """
    Get statistics about the index
    
    Args:
        index_path: Path to the index directory, defaults to app_config.index_path
        texts: List of texts if already loaded, to avoid reloading
        index: FAISS index if already loaded, to avoid reloading
        
    Returns:
        Dictionary with index statistics
    """
    if index_path is None:
        index_path = retrieval_config.index_path
        
    # Resolve the path
    resolved_index_path = resolve_path(index_path)
        
    stats = {
        "has_index": has_index(index_path),
        "index_loaded": index is not None,
        "texts_loaded": texts is not None,
        "num_chunks": len(texts) if texts is not None else 0,
        "gpu_available": False,  # Will be set by the caller if GPU is available
        "index_path": resolved_index_path
    }
    
    # Add file existence and size information
    faiss_path = f"{resolved_index_path}/index.faiss"
    texts_path = f"{resolved_index_path}/texts.npy"
    stats["index_file_exists"] = os.path.exists(faiss_path)
    stats["texts_file_exists"] = os.path.exists(texts_path)
    
    if stats["index_file_exists"]:
        stats["index_file_size"] = os.path.getsize(faiss_path)
    if stats["texts_file_exists"]:
        stats["texts_file_size"] = os.path.getsize(texts_path)
        
    # Get sample texts if available
    if stats["texts_loaded"] and stats["num_chunks"] > 0:
        sample_size = min(5, stats["num_chunks"])
        stats["sample_texts"] = [texts[i][:200] + "..." for i in range(sample_size)]
        
    return stats

def fix_index(index_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Attempt to fix common index issues
    
    Args:
        index_path: Path to the index directory, defaults to app_config.index_path
        
    Returns:
        Tuple of (success, message)
    """
    if index_path is None:
        index_path = retrieval_config.index_path
        
    # Resolve the path
    resolved_index_path = resolve_path(index_path)
    
    try:
        # Check if index directory exists
        if not os.path.exists(resolved_index_path):
            os.makedirs(resolved_index_path, exist_ok=True)
            logger.info(f"Created index directory at {resolved_index_path}")
        
        # Check if current_index directory exists
        current_index_dir = resolve_path("current_index", create_dir=True)
        if not os.path.exists(current_index_dir):
            os.makedirs(current_index_dir, exist_ok=True)
            logger.info(f"Created current_index directory at {current_index_dir}")
        
        # Create empty chunks.txt file if it doesn't exist
        chunks_file = os.path.join(current_index_dir, "chunks.txt")
        if not os.path.exists(chunks_file) or os.path.getsize(chunks_file) == 0:
            with open(chunks_file, "w") as f:
                f.write("# Chunks file for current index\n")
            logger.info(f"Created empty chunks.txt file at {chunks_file}")
        
        # Check if index files exist
        index_file = os.path.join(resolved_index_path, "index.faiss")
        texts_file = os.path.join(resolved_index_path, "texts.npy")
        metadata_file = os.path.join(resolved_index_path, "metadata.json")
        
        # If any of the files are missing, we need to create a new empty index
        if not os.path.exists(index_file) or not os.path.exists(texts_file):
            logger.warning(f"Index files missing at {resolved_index_path}, creating new empty index")
            
            # Import DocumentProcessor here to avoid circular imports
            from utils.document_processor_unified import DocumentProcessor
            
            # Create a new document processor
            processor = DocumentProcessor()
            
            # Create a new empty index
            processor.create_empty_index()
            
            # Save the empty index
            processor.save_index(resolved_index_path)
            
            # Create metadata file if it doesn't exist
            if not os.path.exists(metadata_file):
                metadata = {
                    "embedding_model": retrieval_config.embedding_model,
                    "chunk_size": retrieval_config.chunk_size,
                    "chunk_overlap": retrieval_config.chunk_overlap,
                    "embedding_dim": processor.embedding_dim,
                    "processed_files": []
                }
                
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f)
                    
                logger.info(f"Created metadata file at {metadata_file}")
        
        return True, "Index fixed successfully"
    except Exception as e:
        error_msg = f"Error fixing index: {str(e)}"
        logger.error(error_msg)
        return False, error_msg