"""FAISS Index Builder

Handles FAISS index creation and optimization strategies.
Extracted from core/vector_index_engine.py
"""

import logging
from typing import Optional

import numpy as np
import faiss

from core.config import performance_config
from core.index_management.gpu_manager import gpu_manager
from utils.logging.error_logging import log_error

# Setup logging
logger = logging.getLogger(__name__)


class IndexBuilder:
    """Handles FAISS index creation and optimization"""

    def __init__(self, embedding_dim: int):
        """
        Initialize the index builder

        Args:
            embedding_dim: Dimension of embeddings for index creation
        """
        self.embedding_dim = embedding_dim
        logger.info(f"Index builder initialized with dimension {embedding_dim}")

    def create_empty_index(self) -> faiss.Index:
        """
        Create a new empty FAISS index

        Returns:
            Empty FAISS index
        """
        logger.info(f"Creating new empty FAISS index with dimension {self.embedding_dim}")

        # Create basic flat index
        index = faiss.IndexFlatL2(self.embedding_dim)

        logger.info(f"Created empty index with dimension {self.embedding_dim}")
        return index

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index from embeddings with optimization

        Args:
            embeddings: NumPy array of embeddings

        Returns:
            FAISS index

        Raises:
            ValueError: If embedding dimensions don't match or embeddings are invalid
        """
        # Validate embeddings
        self._validate_embeddings(embeddings)

        # Choose index type based on dataset size and configuration
        index = self._create_optimized_index(embeddings)

        # Add embeddings to index
        logger.info(f"Adding {len(embeddings)} embeddings to index")
        index.add(embeddings)

        # Move to GPU if configured
        if gpu_manager.should_use_gpu():
            index, success = gpu_manager.move_index_to_gpu(index)
            if success:
                logger.info("Index moved to GPU successfully")
            else:
                logger.warning("Failed to move index to GPU, using CPU")

        # Verify final index
        self._verify_index(index, embeddings)

        return index

    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Validate embeddings array

        Args:
            embeddings: NumPy array to validate

        Raises:
            ValueError: If embeddings are invalid
        """
        if embeddings is None or not isinstance(embeddings, np.ndarray):
            error_msg = f"Invalid embeddings array: {type(embeddings)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if len(embeddings) == 0:
            error_msg = "Empty embeddings array"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if embeddings.shape[1] != self.embedding_dim:
            error_msg = f"Embedding dimensions ({embeddings.shape[1]}) don't match index dimensions ({self.embedding_dim})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Embeddings validation passed: {embeddings.shape}")

    def _create_optimized_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Create optimized index based on dataset size and FAISS requirements

        Args:
            embeddings: Embeddings to build index for

        Returns:
            Optimized FAISS index that meets FAISS clustering requirements
        """
        num_embeddings = len(embeddings)
        logger.info(f"Creating optimized index for {num_embeddings} embeddings")

        # FAISS clustering requirements: minimum ~39 training points per cluster
        min_training_per_cluster = 39
        min_embeddings_for_clustering = 1000  # Conservative threshold
        
        # Use flat index for small datasets to avoid clustering warnings
        if not performance_config.enable_faiss_optimization or num_embeddings < min_embeddings_for_clustering:
            logger.info(f"Using flat index (dataset size {num_embeddings} < {min_embeddings_for_clustering} threshold)")
            return faiss.IndexFlatL2(self.embedding_dim)
        
        # Calculate proper cluster count based on FAISS requirements
        # Start with sqrt rule, then validate against training requirements
        nlist_sqrt = int(np.sqrt(num_embeddings))
        max_clusters_safe = num_embeddings // min_training_per_cluster
        
        # Use the smaller of sqrt rule or safe maximum
        nlist = min(nlist_sqrt, max_clusters_safe)
        
        # Ensure minimum cluster count of 4 for meaningful clustering
        nlist = max(nlist, 4)
        
        # Final safety cap at 256 (reasonable maximum)
        nlist = min(nlist, 256)
        
        # Validate that we have enough training points
        training_points_per_cluster = num_embeddings / nlist
        if training_points_per_cluster < min_training_per_cluster:
            logger.warning(f"Insufficient training points per cluster ({training_points_per_cluster:.1f} < {min_training_per_cluster}), using flat index")
            return faiss.IndexFlatL2(self.embedding_dim)
        
        logger.info(f"Creating IVF index with {nlist} clusters ({training_points_per_cluster:.1f} training points per cluster)")
        
        try:
            # Create IVF index
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_L2)
            
            # Train the index with the embeddings
            logger.info("Training IVF index")
            index.train(embeddings)
            
            logger.info(f"Successfully created and trained IVF index with {nlist} clusters")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create IVF index: {e}, falling back to flat index")
            return faiss.IndexFlatL2(self.embedding_dim)

    def _verify_index(self, index: faiss.Index, embeddings: np.ndarray) -> None:
        """
        Verify that index was built correctly

        Args:
            index: FAISS index to verify
            embeddings: Original embeddings

        Raises:
            AssertionError: If index verification fails
        """
        # Verify that index dimensions match embedding dimensions
        if hasattr(index, 'd'):
            assert index.d == self.embedding_dim, \
                f"Index dimensions ({index.d}) don't match embedding dimensions ({self.embedding_dim})"

        # Verify that all embeddings were added (for non-GPU indices)
        if hasattr(index, 'ntotal') and not (hasattr(index, 'getDevice') and index.getDevice() >= 0):
            expected_count = len(embeddings)
            actual_count = index.ntotal
            if actual_count != expected_count:
                logger.warning(f"Index contains {actual_count} vectors, expected {expected_count}")
            else:
                logger.debug(f"Index verification passed: {actual_count} vectors")

        logger.info("Index verification completed successfully")

    def get_index_info(self, index: faiss.Index) -> dict:
        """
        Get information about a FAISS index

        Args:
            index: FAISS index to analyze

        Returns:
            Dictionary with index information
        """
        info = {
            "index_type": type(index).__name__,
            "embedding_dim": getattr(index, 'd', None),
            "total_vectors": getattr(index, 'ntotal', None),
            "is_trained": getattr(index, 'is_trained', None),
            "metric_type": getattr(index, 'metric_type', None),
        }

        # Check GPU status
        if hasattr(index, 'getDevice'):
            try:
                device = index.getDevice()
                info["on_gpu"] = device >= 0
                info["gpu_device"] = device if device >= 0 else None
            except:
                info["on_gpu"] = False
                info["gpu_device"] = None
        else:
            info["on_gpu"] = False
            info["gpu_device"] = None

        # Add IVF-specific information
        if hasattr(index, 'nlist'):
            info["nlist"] = index.nlist
        if hasattr(index, 'nprobe'):
            info["nprobe"] = index.nprobe

        return info

    def optimize_search_params(self, index: faiss.Index, target_recall: float = 0.9) -> None:
        """
        Optimize search parameters for better recall/speed tradeoff

        Args:
            index: FAISS index to optimize
            target_recall: Target recall rate (0.0 to 1.0)
        """
        if hasattr(index, 'nprobe'):
            # For IVF indices, adjust nprobe based on target recall
            # Higher nprobe = better recall but slower search
            if target_recall >= 0.95:
                index.nprobe = min(50, getattr(index, 'nlist', 100))
            elif target_recall >= 0.9:
                index.nprobe = min(20, getattr(index, 'nlist', 100))
            else:
                index.nprobe = min(10, getattr(index, 'nlist', 100))

            logger.info(f"Optimized search parameters: nprobe={index.nprobe} for target_recall={target_recall}")
        else:
            logger.debug("Index does not support search parameter optimization")
