# Unified Index Manager class for handling vector index operations
import os
import time
import logging
import numpy as np
import faiss
from typing import Tuple, List, Optional, Any, Dict, cast

from utils.config_unified import app_config, retrieval_config, performance_config, resolve_path
from utils.index_management.index_operations import (
    load_index,
    save_index,
    clear_index,
    rebuild_index_if_needed,
    get_index_modified_time,
    get_index_stats
)
from utils.index_management.index_health import check_index_health, has_index
from utils.index_management.index_freshness import is_index_fresh

# Setup logging
logger = logging.getLogger(__name__)

class IndexManager:
    """Centralized manager for FAISS index operations"""
    
    def __init__(self, index_path: Optional[str] = None, gpu_available: bool = False):
        """
        Initialize the index manager
        
        Args:
            index_path: Path to the index directory, defaults to retrieval_config.index_path
            gpu_available: Whether GPU is available for FAISS
        """
        self.index_path = index_path if index_path is not None else retrieval_config.index_path
        self.gpu_available = gpu_available
        self.index = None
        self.texts = []
        self.last_modified_time = 0
        self.is_loaded = False
        self.last_load_time = 0
        self.load_count = 0
    
    def load(self, force_reload: bool = False, dry_run: bool = False) -> Tuple[bool, str]:
        """
        Load the index if needed
        
        Args:
            force_reload: Whether to force reload the index even if it's already loaded
            dry_run: If True, skip disk operations (preview only)
            
        Returns:
            Tuple of (success, message)
        """
        # Check if we need to reload the index
        if not force_reload and self.is_loaded:
            is_fresh, current_mtime = is_index_fresh(
                index_path=self.index_path,
                last_known_mtime=self.last_modified_time
            )
            if is_fresh:
                return True, f"Using cached index with {len(self.texts)} chunks"
        
        try:
            # Check if index needs rebuilding
            was_rebuilt, rebuild_message = rebuild_index_if_needed(self.index_path, dry_run=dry_run)
            if was_rebuilt:
                logger.warning(f"{'[DRY RUN] ' if dry_run else ''}Index was rebuilt: {rebuild_message}")
            
            # Load the index
            try:
                self.index, self.texts, num_chunks, message = load_index(
                    index_path=self.index_path,
                    gpu_available=self.gpu_available,
                    force_reload=force_reload
                )
            except Exception as e:
                logger.error(f"{'[DRY RUN] ' if dry_run else ''}Error loading index: {str(e)}", exc_info=True)
                # Create a new empty index
                from sentence_transformers import SentenceTransformer
                model_name = retrieval_config.embedding_model
                embedder = SentenceTransformer(model_name)
                dimension = embedder.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatL2(dimension)
                self.texts = []
                num_chunks = 0
                message = f"{'[DRY RUN] ' if dry_run else ''}Created new empty index with dimension {dimension} after load error: {str(e)}"
            
            # Update state
            self.is_loaded = True
            self.last_load_time = time.time()
            if not dry_run:
                self.last_modified_time = get_index_modified_time(self.index_path)
            self.load_count += 1
            
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Index loaded with {num_chunks} chunks")
            return True, message
        except Exception as e:
            logger.error(f"{'[DRY RUN] ' if dry_run else ''}Error loading index: {str(e)}", exc_info=True)
            self.is_loaded = False
            return False, f"{'[DRY RUN] ' if dry_run else ''}Failed to load index: {str(e)}"
    
    def save(self, dry_run: bool = False) -> Tuple[bool, str]:
        """
        Save the index to disk
        
        Args:
            dry_run: If True, skip saving to disk (preview only)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if self.index is None:
                return False, "No index to save"
                
            save_index(
                index=self.index,
                texts=self.texts,
                gpu_available=self.gpu_available,
                index_path=self.index_path,
                dry_run=dry_run
            )
            
            # Update state (only update last_modified_time if not in dry-run mode)
            if not dry_run:
                self.last_modified_time = get_index_modified_time(self.index_path)
                return True, f"Index saved with {len(self.texts)} chunks"
            else:
                return True, f"[DRY RUN] Index would be saved with {len(self.texts)} chunks"
        except Exception as e:
            logger.error(f"{'[DRY RUN] ' if dry_run else ''}Error saving index: {str(e)}", exc_info=True)
            return False, f"{'[DRY RUN] ' if dry_run else ''}Failed to save index: {str(e)}"
    
    def clear(self, rebuild_immediately: bool = False, embedder: Any = None, dry_run: bool = False) -> Tuple[bool, str]:
        """
        Clear the index
        
        Args:
            rebuild_immediately: Whether to rebuild the index immediately after clearing
            embedder: Sentence transformer model for creating empty index
            dry_run: If True, skip disk operations (preview only)
            
        Returns:
            Tuple of (success, message)
        """
        try:
            clear_index(
                rebuild_immediately=rebuild_immediately,
                embedder=embedder,
                gpu_available=self.gpu_available,
                index_path=self.index_path,
                dry_run=dry_run
            )
            
            # Update state
            self.index = None
            self.texts = []
            self.is_loaded = False
            
            if dry_run:
                return True, "[DRY RUN] Index would be cleared"
            else:
                return True, "Index cleared successfully"
        except Exception as e:
            logger.error(f"{'[DRY RUN] ' if dry_run else ''}Error clearing index: {str(e)}", exc_info=True)
            return False, f"{'[DRY RUN] ' if dry_run else ''}Failed to clear index: {str(e)}"
    
    def check_health(self) -> Tuple[bool, str]:
        """
        Check the health of the index
        
        Returns:
            Tuple of (is_healthy, message)
        """
        return check_index_health(self.index_path)
    
    def has_index(self) -> bool:
        """
        Check if the index exists
        
        Returns:
            True if the index exists, False otherwise
        """
        return has_index(self.index_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index
        
        Returns:
            Dictionary of index statistics
        """
        stats = get_index_stats(self.index_path, tuple(self.texts), self.index)
        
        # Add manager-specific stats
        stats.update({
            "is_loaded": self.is_loaded,
            "last_load_time": self.last_load_time,
            "load_count": self.load_count,
            "gpu_available": self.gpu_available
        })
        
        return stats
        
# Removed unused method handle_dimension_mismatch
    
    def replace_index(self, new_index: Any, new_texts: List[str]) -> Tuple[bool, str]:
        """
        Replace the current index with a new one

        Args:
            new_index: New FAISS index
            new_texts: New text chunks

        Returns:
            Tuple of (success, message)
        """
        self.index = new_index
        self.texts = new_texts
        self.is_loaded = True
        logger.info(f"Index replaced with {len(new_texts)} chunks.")

        # Save the new index
        return self.save(dry_run=False)
        
    def rebuild_if_needed(self, dry_run: bool = False) -> Tuple[bool, str]:
        """
        Check if index needs rebuilding and rebuild if necessary
        
        Args:
            dry_run: If True, skip disk operations (preview only)
            
        Returns:
            Tuple of (was_rebuilt, message)
        """
        return rebuild_index_if_needed(self.index_path, dry_run=dry_run)
    
    def check_freshness(self) -> bool:
        """
        Check if the index is fresh (hasn't been modified since last load)
        
        Returns:
            True if the index is fresh, False if it needs reloading
        """
        is_fresh, current_mtime = is_index_fresh(self.index_path, self.last_modified_time)
        if not is_fresh:
            logger.info("Index has been modified since last load, needs reloading")
            self.last_modified_time = current_mtime
        return is_fresh
    
    def update_index(self, texts: List[str], embeddings: np.ndarray, append: bool = False, dry_run: bool = False) -> Tuple[bool, str]:
        """
        Update the index with new texts and embeddings
        
        Args:
            texts: List of text chunks to add
            embeddings: Numpy array of embeddings for the texts
            append: If True, append to existing index; if False, replace existing index
            dry_run: If True, update index in memory only without saving to disk
            
        Returns:
            Tuple of (success, message)
        """
        if len(texts) != embeddings.shape[0]:
            error_msg = f"Number of texts ({len(texts)}) does not match number of embeddings ({embeddings.shape[0]})"
            logger.error(error_msg)
            return False, error_msg
            
        if not self.is_loaded or self.index is None:
            success, message = self.load()
            if not success:
                return False, f"Failed to load index: {message}"
        
        try:
            # Check if dimensions match
            if self.index is not None and hasattr(self.index, 'd') and embeddings.shape[1] != self.index.d:
                # Dimensions don't match, need to rebuild the index
                logger.warning(f"Embedding dimensions mismatch: index has {self.index.d}, but embeddings have {embeddings.shape[1]}")
                
                # Create a new index with the correct dimensions
                dimension = embeddings.shape[1]
                if self.index is not None and hasattr(self.index, 'is_trained') and self.index.is_trained:
                    # For IVF indexes, we need to recreate with the same parameters
                    if isinstance(self.index, faiss.IndexIVFFlat):
                        nlist = self.index.nlist
                        if not isinstance(nlist, int):
                            raise TypeError(f"Invalid nlist type: {type(nlist).__name__}. Expected integer")
                        quantizer = faiss.IndexFlatL2(dimension)
                        new_index = faiss.IndexIVFFlat(quantizer, dimension, int(nlist), faiss.METRIC_L2)
                        # Train the new index if we have enough data
                        if len(embeddings) > nlist:
                            new_index.train(embeddings)
                    else:
                        # Default to flat index for other types
                        new_index = faiss.IndexFlatL2(dimension)
                else:
                    # Create a simple flat index
                    new_index = faiss.IndexFlatL2(dimension)
                
                # Replace the old index
                self.index = new_index
                self.texts = []  # Clear texts as we're rebuilding
                logger.info(f"{'[DRY RUN] ' if dry_run else ''}Created new index with dimension {dimension}")
                append = False  # Force replace mode since we're rebuilding
            
            # If not appending, clear the existing index first
            if not append and self.index is not None and hasattr(self.index, 'reset'):
                self.index.reset()
                self.texts = []
                logger.info(f"{'[DRY RUN] ' if dry_run else ''}Cleared existing index before adding new data")
            
            # Add the embeddings to the index
            if self.index is not None:
                self.index.add(embeddings.astype(np.float32))
            
            # Add the texts to our list
            self.texts.extend(texts)
            
            # Log chunk count
            chunk_count = len(self.texts)
            if append:
                logger.info(f"{'[DRY RUN] ' if dry_run else ''}Index updated with {len(texts)} new chunks. Total chunks: {chunk_count}")
            else:
                logger.info(f"{'[DRY RUN] ' if dry_run else ''}Index replaced with {len(texts)} chunks.")
            
            # Check for abnormal chunk counts
            if chunk_count < 10:
                warning_msg = f"{'[DRY RUN] ' if dry_run else ''}WARNING: Index contains only {chunk_count} chunks, which is unusually low. This may indicate issues with document parsing."
                logger.warning(warning_msg)
                # Log to central error log
                from utils.error_logging import log_warning
                log_warning(warning_msg, include_traceback=False)
            elif chunk_count > 1000:
                warning_msg = f"{'[DRY RUN] ' if dry_run else ''}WARNING: Index contains {chunk_count} chunks, which is unusually high. This may cause performance issues."
                logger.warning(warning_msg)
                # Log to central error log
                from utils.error_logging import log_warning
                log_warning(warning_msg, include_traceback=False)
            
            # Save the updated index if not in dry-run mode
            if dry_run:
                logger.info("[DRY RUN] Index updated in memory only - changes not saved to disk")
                return True, f"[DRY RUN] Index updated in memory with {len(texts)} chunks (not saved to disk)"
            else:
                # Save the updated index to disk
                return self.save()
        except Exception as e:
            error_msg = f"{'[DRY RUN] ' if dry_run else ''}Error updating index: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Log to central error log
            from utils.error_logging import log_error
            log_error(error_msg)
            return False, error_msg
    
    def search(self, query_vector: np.ndarray, top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if self.index is None:
            empty = np.array([[]])
            return empty, empty, []

        query_array = query_vector.reshape(1, -1).astype(np.float32)
        actual_top_k = top_k if top_k is not None else 10

        if hasattr(self.index, 'ntotal'):
            index_size = cast(int, self.index.ntotal)
            actual_top_k = min(actual_top_k, index_size)

        distances: np.ndarray = np.array([[]])
        indices: np.ndarray = np.array([[]])

        if self.index is not None and hasattr(self.index, 'search'):
            distances, indices = self.index.search(
                query_array,
                actual_top_k
            )
        else:
            search_results = self.index.search(
                x=query_array,
                k=actual_top_k,
                distances=np.empty((1, actual_top_k), dtype=np.float32),
                labels=np.empty((1, actual_top_k), dtype=np.int64)
            )

        result_texts: List[str] = []
        if indices.size > 0:
            valid_indices = (indices[0] >= 0) & (indices[0] < len(self.texts))
            result_texts = [
                self.texts[idx] if valid else ""
                for idx, valid in zip(indices[0], valid_indices)
            ]

        return distances, indices, result_texts

# Create a global instance for convenience
index_manager = IndexManager()
