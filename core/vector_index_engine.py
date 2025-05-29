"""Index Manager Module

Refactored facade for vector index operations using modular components.
Extracted large components to core/vector_index/ package for better maintainability.
"""

import logging
from typing import List, Dict, Tuple, Any, Optional, Union, Sequence

import numpy as np
import faiss

from utils.config import retrieval_config, resolve_path
from utils.common.embedding_service import embedding_service
from utils.index_management.gpu_manager import gpu_manager
from core.vector_index.index_builder import IndexBuilder
from core.vector_index.storage_manager import StorageManager
from core.vector_index.search_engine import SearchEngine
from error_handling.enhanced_decorators import with_timing

# Setup logging
logger = logging.getLogger(__name__)


class IndexManager:
    """Refactored facade for vector index operations using modular components"""

    def __init__(self, embedding_model_name: str = None):
        """
        Initialize the index manager with modular components

        Args:
            embedding_model_name: Name of the embedding model to use (defaults to config value)
        """
        self.embedding_model_name = embedding_model_name or retrieval_config.embedding_model
        
        # Get embedding dimension from the service
        self.embedding_dim = embedding_service.embedding_dim
        
        # Initialize state variables for backward compatibility
        self.index = None
        self.texts = []
        self.chunks = []  # Alias for self.texts for backward compatibility
        
        # Initialize modular components
        self.index_builder = IndexBuilder(self.embedding_dim)
        self.storage_manager = StorageManager(self.embedding_model_name, self.embedding_dim)
        self.search_engine = SearchEngine(self.embedding_dim)
        
        # GPU availability from manager
        self.gpu_available = gpu_manager.gpu_available
        self.index_on_gpu = False
        
        logger.info(f"Index manager initialized with model {self.embedding_model_name}, delegating to modular components")

    def __del__(self):
        """
        Destructor to ensure GPU resources are cleaned up when the object is destroyed
        """
        try:
            # GPU cleanup is now handled by gpu_manager
            gpu_manager.cleanup_resources()
        except Exception as e:
            # Don't raise exceptions in destructor
            pass

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

        # Delegate to storage manager for disk check
        return self.storage_manager.index_exists(index_dir)

    def create_empty_index(self) -> None:
        """
        Create a new empty FAISS index
        """
        # Delegate to index builder
        self.index = self.index_builder.create_empty_index()
        
        # Initialize empty texts list
        self.texts = []
        self.chunks = []  # Alias for backward compatibility
        
        logger.info(f"Created empty index with dimension {self.embedding_dim}")

    @with_timing(threshold=1.0)
    def batch_embed_chunks(
        self, chunks: Sequence[Union[Dict[str, Any], str]], batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Embed document chunks in batches

        Args:
            chunks: List of document chunks (either dictionaries with 'text' key or strings)
            batch_size: Size of batches for embedding (None for default)

        Returns:
            NumPy array of embeddings
        """
        # Delegate to embedding service
        return embedding_service.embed_texts(chunks, batch_size)

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
        # Delegate to index builder
        return self.index_builder.build_index(embeddings)

    def create_index_from_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """
        Create a FAISS index from document chunks

        Args:
            chunks: List of document chunks

        Returns:
            Tuple of (FAISS index, list of chunks)
        """
        # Store chunks
        self.texts = chunks
        self.chunks = chunks  # Keep both references in sync

        # Embed chunks
        embeddings = self.batch_embed_chunks(chunks)

        # Build index
        self.index = self.build_index(embeddings)

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
            raise ValueError("No index has been built. Process documents first.")

        # Delegate to search engine
        return self.search_engine.search(query, self.index, self.chunks, top_k)

    def clear_index(self, index_dir: Optional[str] = None) -> None:
        """
        Clear the index and all cached data

        Args:
            index_dir: Directory containing the index files to clear (defaults to config value)
        """
        # Clear in-memory index and texts
        self.index = None
        self.texts = []
        self.chunks = []  # Keep alias in sync

        # Delegate disk clearing to storage manager
        self.storage_manager.clear_index(index_dir)
        
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

        # Delegate to storage manager
        self.storage_manager.save_index(self.index, self.texts, index_dir, dry_run)

    def load_index(self, index_dir: Optional[str] = None) -> None:
        """
        Load a FAISS index and chunks from disk

        Args:
            index_dir: Directory containing the index and chunks (defaults to config value)

        Raises:
            FileNotFoundError: If index files are not found
            ValueError: If index parameters don't match
        """
        # Delegate to storage manager
        self.index, self.texts = self.storage_manager.load_index(index_dir)
        
        # Update chunks alias for backward compatibility
        self.chunks = self.texts
        
        # If we have texts and a new index with no vectors, rebuild the index
        if (
            self.texts
            and len(self.texts) > 0
            and self.index is not None
            and hasattr(self.index, 'ntotal')
            and self.index.ntotal == 0
        ):
            try:
                logger.info("Rebuilding index with loaded texts")
                embeddings = self.batch_embed_chunks(self.texts)
                self.index.add(embeddings)
                logger.info(f"Rebuilt index with {len(self.texts)} chunks")
            except Exception as e:
                logger.error(f"Error rebuilding index: {str(e)}")
        
        logger.info(f"Successfully loaded index with {len(self.texts)} chunks")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string

        Args:
            query: Query string to embed

        Returns:
            NumPy array of query embedding
        """
        # Delegate to embedding service
        return embedding_service.embed_query(query)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get index manager metrics

        Returns:
            Dictionary with index manager metrics
        """
        # Base metrics from this manager
        metrics = {
            "gpu_available": self.gpu_available,
            "index_on_gpu": self.index_on_gpu,
            "embedding_dim": self.embedding_dim,
            "embedding_model": self.embedding_model_name,
            "total_chunks": len(self.chunks) if self.chunks else 0,
            "index_size": self.index.ntotal if self.index else 0,
        }

        # Get embedding metrics from embedding service
        embedding_metrics = embedding_service.get_metrics()
        metrics.update({
            "avg_embedding_time": embedding_metrics.get("avg_embedding_time", 0.0),
            "min_embedding_time": embedding_metrics.get("min_embedding_time", 0.0),
            "max_embedding_time": embedding_metrics.get("max_embedding_time", 0.0),
            "total_embeddings": embedding_metrics.get("total_embeddings", 0),
        })

        # Get GPU stats from GPU manager
        gpu_stats = gpu_manager.get_gpu_stats()
        metrics["gpu_stats"] = gpu_stats

        return metrics
