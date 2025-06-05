"""Centralized Embedding Service

Consolidates SentenceTransformer operations across the codebase to eliminate redundancy.
Extracted from core/vector_index_engine.py and other modules.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union, Sequence
from collections import deque

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from core.config import retrieval_config, performance_config
from utils.error_handling.enhanced_decorators import with_timing
from utils.logging.error_logging import log_error

# Setup logging
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Centralized service for all embedding operations"""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding service

        Args:
            model_name: Name of the embedding model to use (defaults to config value)
        """
        self.model_name = model_name or retrieval_config.embedding_model

        # Detect and configure device (GPU/CPU)
        self.device = self._get_device()

        # Initialize embedding model with proper device
        logger.info(f"Initializing SentenceTransformer model '{self.model_name}' on device: {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize metrics tracking
        self.embedding_times = deque(maxlen=100)  # Keep last 100 embedding times

        logger.info(f"Embedding service initialized with model {self.model_name}, dimension: {self.embedding_dim}, device: {self.device}")

    @with_timing(threshold=1.0)
    def embed_texts(
        self,
        texts: Union[List[str], Sequence[Union[Dict[str, Any], str]]],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Embed a list of texts with batch processing

        Args:
            texts: List of texts to embed (strings or dicts with 'text' key)
            batch_size: Size of batches for embedding (None for default)

        Returns:
            NumPy array of embeddings

        Raises:
            ValueError: If texts parameter is invalid
        """
        # Validate input
        if texts is None:
            logger.error("Cannot embed None texts")
            raise ValueError("texts parameter cannot be None")

        if not isinstance(texts, (list, tuple)):
            logger.error(f"texts must be a list or tuple, got {type(texts)}")
            raise TypeError(f"texts must be a list or tuple, got {type(texts)}")

        if len(texts) == 0:
            logger.warning("Empty texts list provided for embedding")
            return np.array([])

        # Use configured batch size if not specified
        batch_size = batch_size or performance_config.embedding_batch_size

        # Validate batch size
        if batch_size <= 0:
            logger.warning(f"Invalid batch size {batch_size}, using default of 32")
            batch_size = 32

        # Extract text strings from input
        text_strings = self._extract_text_strings(texts)

        # Embed in batches
        all_embeddings = []
        invalid_batches = 0

        for i in range(0, len(text_strings), batch_size):
            batch_texts = text_strings[i : i + batch_size]

            # Time the embedding process
            start_time = time.time()
            try:
                batch_embeddings = self.model.encode(batch_texts)
                embedding_time = time.time() - start_time

                # Validate embedding dimensions
                if batch_embeddings.shape[1] != self.embedding_dim:
                    error_msg = f"Embedding dimension mismatch: got {batch_embeddings.shape[1]}, expected {self.embedding_dim}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # Track embedding time
                self.embedding_times.append(embedding_time)
                all_embeddings.append(batch_embeddings)

            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
                invalid_batches += 1

                # If this is the first batch and it fails, re-raise the exception
                if i == 0:
                    raise

                # Otherwise, log the error and continue with remaining batches
                logger.warning(f"Continuing with remaining batches after error in batch {i//batch_size + 1}")

        # Validate results
        if not all_embeddings:
            logger.error("No embeddings were generated")
            return np.array([])

        # Log warnings if many batches failed
        if invalid_batches > 0:
            total_batches = (len(text_strings) + batch_size - 1) // batch_size
            failure_rate = invalid_batches / total_batches
            if failure_rate > 0.5:
                logger.error(f"High embedding failure rate: {invalid_batches}/{total_batches} batches failed")

        # Concatenate all embeddings
        result = np.vstack(all_embeddings)

        # Final validation
        if result.shape[0] != len(texts):
            logger.warning(f"Embedding count mismatch: got {result.shape[0]}, expected {len(texts)}")
        if result.shape[1] != self.embedding_dim:
            logger.error(f"Final embedding dimension mismatch: got {result.shape[1]}, expected {self.embedding_dim}")

        return result

    def embed_documents(self, texts: Union[List[str], Sequence[Union[Dict[str, Any], str]]]) -> np.ndarray:
        """Alias for embed_texts to maintain compatibility with model preloader.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            NumPy array of embeddings
        """
        return self.embed_texts(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string

        Args:
            query: Query string to embed

        Returns:
            NumPy array of query embedding

        Raises:
            ValueError: If query is invalid
        """
        if not query or not isinstance(query, str):
            logger.error(f"Invalid query: {query}")
            raise ValueError(f"Query must be a non-empty string, got {type(query)}")

        try:
            start_time = time.time()
            embedding = self.model.encode([query])
            embedding_time = time.time() - start_time

            # Track embedding time
            self.embedding_times.append(embedding_time)

            # Validate dimensions
            if embedding.shape[1] != self.embedding_dim:
                error_msg = f"Query embedding dimension mismatch: got {embedding.shape[1]}, expected {self.embedding_dim}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            return embedding

        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise ValueError(f"Failed to embed query: {str(e)}")

    def _extract_text_strings(self, texts: Sequence[Union[Dict[str, Any], str]]) -> List[str]:
        """
        Extract text strings from mixed input format

        Args:
            texts: List of texts (strings or dicts with 'text' key)

        Returns:
            List of text strings
        """
        text_strings = []
        invalid_count = 0

        for item in texts:
            if isinstance(item, dict) and "text" in item:
                if not item["text"] or not isinstance(item["text"], str):
                    logger.warning(f"Empty or non-string text in item: {item}")
                    text_strings.append("")
                    invalid_count += 1
                else:
                    text_strings.append(item["text"])
            elif isinstance(item, str):
                text_strings.append(item)
            else:
                logger.warning(f"Skipping invalid item format: {type(item)}")
                # Add empty string as placeholder to maintain index alignment
                text_strings.append("")
                invalid_count += 1

        # Log warning if too many invalid items
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid items out of {len(texts)} total items")
            if invalid_count / len(texts) > 0.5:
                logger.error(f"More than 50% of items are invalid ({invalid_count}/{len(texts)})")

        return text_strings

    def _get_device(self) -> str:
        """
        Determine the best device to use for embeddings

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        try:
            # Check if GPU usage is enabled in config
            if not performance_config.use_gpu_for_faiss:
                logger.info("GPU usage disabled in config, using CPU for embeddings")
                return 'cpu'

            # Check for CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                logger.info(f"CUDA GPU detected: {gpu_name} (devices: {gpu_count})")
                return 'cuda'

            # Check for MPS (Apple Silicon GPU)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Apple MPS GPU detected")
                return 'mps'

            # Fallback to CPU
            logger.info("No GPU detected, using CPU for embeddings")
            return 'cpu'

        except Exception as e:
            logger.warning(f"Error detecting GPU device: {str(e)}, falling back to CPU")
            return 'cpu'

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        info = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "max_seq_length": getattr(self.model, "max_seq_length", None),
        }

        # Add GPU info if available
        if self.device == 'cuda' and torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                "gpu_memory_cached": torch.cuda.memory_reserved(0)
            })

        return info

    def get_metrics(self) -> Dict[str, Any]:
        """Get embedding service metrics"""
        metrics = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "total_embeddings": len(self.embedding_times),
        }

        # Calculate timing statistics
        if self.embedding_times:
            times = list(self.embedding_times)
            metrics.update({
                "avg_embedding_time": sum(times) / len(times),
                "min_embedding_time": min(times),
                "max_embedding_time": max(times),
                "recent_embedding_time": times[-1] if times else 0.0,
            })
        else:
            metrics.update({
                "avg_embedding_time": 0.0,
                "min_embedding_time": 0.0,
                "max_embedding_time": 0.0,
                "recent_embedding_time": 0.0,
            })

        # Add GPU metrics if using GPU
        if self.device == 'cuda' and torch.cuda.is_available():
            try:
                metrics.update({
                    "gpu_memory_allocated_mb": torch.cuda.memory_allocated(0) / (1024 * 1024),
                    "gpu_memory_cached_mb": torch.cuda.memory_reserved(0) / (1024 * 1024),
                    "gpu_utilization": "N/A"  # Would need additional library like nvidia-ml-py
                })
            except Exception as e:
                logger.warning(f"Error getting GPU metrics: {str(e)}")

        return metrics

    def clear_metrics(self) -> None:
        """Clear embedding metrics"""
        self.embedding_times.clear()
        logger.info("Embedding metrics cleared")


# Global instance for shared use across the application
embedding_service = EmbeddingService()
