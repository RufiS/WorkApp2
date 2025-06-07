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
        Embed a single query string with intelligent GPU memory management

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
            # CRITICAL FIX: Check GPU memory pressure before embedding
            if self.device == 'cuda' and torch.cuda.is_available():
                if not self._ensure_sufficient_gpu_memory():
                    # If GPU memory is constrained, try aggressive cleanup
                    logger.warning("GPU memory pressure detected, attempting cleanup before embedding")
                    self._aggressive_gpu_cleanup()
                    
                    # Check again after cleanup
                    if not self._ensure_sufficient_gpu_memory():
                        # Last resort: temporarily move model to CPU for this query
                        return self._embed_query_cpu_fallback(query)

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

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory during query embedding: {str(e)}")
            # Try CPU fallback for this query
            return self._embed_query_cpu_fallback(query)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise ValueError(f"Failed to embed query: {str(e)}")
    
    def _ensure_sufficient_gpu_memory(self, required_mb: float = 512.0) -> bool:
        """
        Check if there's sufficient GPU memory available for embedding operations
        
        Args:
            required_mb: Minimum required memory in MB
            
        Returns:
            True if sufficient memory is available
        """
        if not torch.cuda.is_available():
            return True  # CPU mode, always sufficient
            
        try:
            # Get current memory usage
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)   # MB
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
            
            # Calculate available memory (use reserved as it's more accurate for fragmentation)
            available = total - reserved
            
            # Check if we have enough memory
            sufficient = available >= required_mb
            
            if not sufficient:
                logger.warning(f"Insufficient GPU memory: {available:.1f}MB available < {required_mb:.1f}MB required")
                logger.warning(f"Memory breakdown: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved, {total:.1f}MB total")
            
            return sufficient
            
        except Exception as e:
            logger.error(f"Error checking GPU memory: {e}")
            return False  # Assume insufficient if we can't check
    
    def _aggressive_gpu_cleanup(self) -> Dict[str, float]:
        """
        Perform aggressive GPU memory cleanup to free space for embedding operations
        
        Returns:
            Dictionary with freed memory amounts in MB
        """
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "cached": 0.0}
            
        try:
            # Record initial memory state
            initial_allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            initial_reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            
            logger.info(f"Starting aggressive GPU cleanup: {initial_allocated:.1f}MB allocated, {initial_reserved:.1f}MB reserved")
            
            # Stage 1: Clear CUDA cache multiple times
            for i in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Force garbage collection between cache clears
                import gc
                gc.collect()
            
            # Stage 2: Try to trigger defragmentation with small tensor allocation
            try:
                # Allocate and immediately free a small tensor to trigger defragmentation
                test_tensor = torch.empty(1024, dtype=torch.float32, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
            except Exception:
                pass  # If this fails, continue with cleanup
            
            # Stage 3: Final cleanup round
            for _ in range(2):
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # Calculate freed memory
            final_allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            final_reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            
            freed_memory = {
                "allocated": initial_allocated - final_allocated,
                "cached": initial_reserved - final_reserved
            }
            
            logger.info(f"Aggressive GPU cleanup complete: Freed {freed_memory['allocated']:.1f}MB allocated, {freed_memory['cached']:.1f}MB cached")
            logger.info(f"Current GPU state: {final_allocated:.1f}MB allocated, {final_reserved:.1f}MB reserved")
            
            return freed_memory
            
        except Exception as e:
            logger.error(f"Error during aggressive GPU cleanup: {str(e)}")
            return {"allocated": 0.0, "cached": 0.0}
    
    def _embed_query_cpu_fallback(self, query: str) -> np.ndarray:
        """
        Fallback method to embed query on CPU when GPU is out of memory
        
        Args:
            query: Query string to embed
            
        Returns:
            NumPy array of query embedding
        """
        try:
            logger.warning(f"Using CPU fallback for query embedding due to GPU memory constraints")
            
            # Temporarily move model to CPU
            original_device = None
            if hasattr(self.model, 'device'):
                original_device = self.model.device
            
            if hasattr(self.model, 'to'):
                self.model.to('cpu')
                logger.info("Temporarily moved embedding model to CPU")
            
            try:
                # Embed on CPU
                start_time = time.time()
                embedding = self.model.encode([query])
                embedding_time = time.time() - start_time
                
                # Track embedding time (note it's CPU time)
                self.embedding_times.append(embedding_time)
                
                logger.info(f"Query embedded on CPU in {embedding_time:.3f}s")
                
                return embedding
                
            finally:
                # Move model back to original device if possible
                if original_device is not None and hasattr(self.model, 'to'):
                    try:
                        # Only move back to GPU if we have sufficient memory
                        if str(original_device).startswith('cuda'):
                            if self._ensure_sufficient_gpu_memory(required_mb=1024.0):  # Require more memory for model transfer
                                self.model.to(original_device)
                                logger.info("Moved embedding model back to GPU")
                            else:
                                logger.warning("Keeping embedding model on CPU due to insufficient GPU memory")
                                self.device = 'cpu'  # Update device tracking
                        else:
                            self.model.to(original_device)
                    except Exception as e:
                        logger.warning(f"Could not move model back to {original_device}: {e}")
                        
        except Exception as e:
            logger.error(f"CPU fallback embedding failed: {str(e)}")
            raise ValueError(f"Failed to embed query with CPU fallback: {str(e)}")

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

    def clear_gpu_memory(self) -> Dict[str, float]:
        """
        Clear GPU memory and return freed amounts
        
        Returns:
            Dictionary with freed memory amounts in MB
        """
        freed_memory = {"allocated": 0.0, "cached": 0.0}
        
        if self.device == 'cuda' and torch.cuda.is_available():
            try:
                # Record memory before clearing
                allocated_before = torch.cuda.memory_allocated(0) / (1024 * 1024)
                cached_before = torch.cuda.memory_reserved(0) / (1024 * 1024)
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Record memory after clearing
                allocated_after = torch.cuda.memory_allocated(0) / (1024 * 1024)
                cached_after = torch.cuda.memory_reserved(0) / (1024 * 1024)
                
                freed_memory = {
                    "allocated": allocated_before - allocated_after,
                    "cached": cached_before - cached_after
                }
                
                logger.info(f"GPU memory cleared - Freed: {freed_memory['allocated']:.1f}MB allocated, {freed_memory['cached']:.1f}MB cached")
                
            except Exception as e:
                logger.error(f"Error clearing GPU memory: {str(e)}")
                
        return freed_memory

    def unload_model(self) -> bool:
        """
        Unload the embedding model from memory to free GPU/CPU resources
        
        Returns:
            True if model was successfully unloaded
        """
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Move model to CPU first if on GPU
                if self.device == 'cuda' and hasattr(self.model, 'to'):
                    self.model.to('cpu')
                
                # Delete the model
                del self.model
                self.model = None
                
                # Clear GPU memory
                self.clear_gpu_memory()
                
                logger.info(f"Successfully unloaded embedding model: {self.model_name}")
                return True
            else:
                logger.warning("No model to unload")
                return False
                
        except Exception as e:
            logger.error(f"Error unloading model: {str(e)}")
            return False

    def reload_model(self) -> bool:
        """
        Reload the embedding model after it was unloaded
        
        Returns:
            True if model was successfully reloaded
        """
        try:
            if hasattr(self, 'model') and self.model is not None:
                logger.warning("Model is already loaded")
                return True
                
            # Reinitialize the model
            logger.info(f"Reloading SentenceTransformer model '{self.model_name}' on device: {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            logger.info(f"Successfully reloaded embedding model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading model: {str(e)}")
            return False

    def is_model_loaded(self) -> bool:
        """Check if the embedding model is currently loaded"""
        return hasattr(self, 'model') and self.model is not None

    def clear_metrics(self) -> None:
        """Clear embedding metrics"""
        self.embedding_times.clear()
        logger.info("Embedding metrics cleared")


# Global instance cache for different embedding models - EVALUATION FIX
_embedding_service_cache = {}

def get_embedding_service(force_cpu: bool = False, model_name: Optional[str] = None) -> EmbeddingService:
    """Get embedding service instance with model-specific caching for evaluation.
    
    Args:
        force_cpu: Force CPU usage for worker processes to prevent VRAM exhaustion
        model_name: Specific model name to use (None for default)
        
    Returns:
        EmbeddingService instance for the specified model
    """
    global _embedding_service_cache
    
    # EVALUATION FIX: Use model-specific caching instead of single global instance
    model_name = model_name or retrieval_config.embedding_model
    cache_key = f"{model_name}_{force_cpu}"
    
    if cache_key not in _embedding_service_cache:
        import os
        
        logger.info(f"Creating new embedding service for model: {model_name}")
        
        # CRITICAL FIX: Force CPU for worker processes to prevent CUDA OOM
        if force_cpu or 'PYTEST_CURRENT_TEST' in os.environ or 'MULTIPROCESSING_WORKER' in os.environ:
            # Temporarily override config to force CPU
            original_gpu_setting = performance_config.use_gpu_for_faiss
            performance_config.use_gpu_for_faiss = False
            
            try:
                _embedding_service_cache[cache_key] = EmbeddingService(model_name)
                logger.info(f"Embedding service initialized on CPU for worker PID {os.getpid()}, model: {model_name}")
            finally:
                # Restore original setting
                performance_config.use_gpu_for_faiss = original_gpu_setting
        else:
            _embedding_service_cache[cache_key] = EmbeddingService(model_name)
            logger.info(f"Embedding service initialized on GPU for main process PID {os.getpid()}, model: {model_name}")
    else:
        logger.debug(f"Using cached embedding service for model: {model_name}")
    
    return _embedding_service_cache[cache_key]

def clear_embedding_service_cache():
    """Clear the embedding service cache - useful for testing different models."""
    global _embedding_service_cache
    
    # Unload all models and clear GPU memory before clearing cache
    total_freed = {"allocated": 0.0, "cached": 0.0}
    for cache_key, service in _embedding_service_cache.items():
        try:
            freed = service.clear_gpu_memory()
            total_freed["allocated"] += freed["allocated"]
            total_freed["cached"] += freed["cached"]
            service.unload_model()
        except Exception as e:
            logger.warning(f"Error cleaning up service {cache_key}: {str(e)}")
    
    _embedding_service_cache.clear()
    
    # Final GPU memory clear
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    logger.info(f"Embedding service cache cleared - Total freed: {total_freed['allocated']:.1f}MB allocated, {total_freed['cached']:.1f}MB cached")


def force_gpu_memory_cleanup():
    """Force aggressive GPU memory cleanup across all services"""
    try:
        # Clear embedding service cache
        clear_embedding_service_cache()
        
        # Force multiple rounds of garbage collection
        import gc
        for _ in range(3):
            gc.collect()
        
        # Clear CUDA cache multiple times
        if torch.cuda.is_available():
            for _ in range(3):
                torch.cuda.empty_cache()
            
            # Get final memory state
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            cached = torch.cuda.memory_reserved(0) / (1024 * 1024)
            
            logger.info(f"Aggressive GPU cleanup complete - Remaining: {allocated:.1f}MB allocated, {cached:.1f}MB cached")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during aggressive GPU cleanup: {str(e)}")
        return False

# Backward compatibility property
class EmbeddingServiceProxy:
    """Proxy to maintain backward compatibility while implementing lazy loading."""
    
    def __getattr__(self, name):
        service = get_embedding_service()
        return getattr(service, name)
    
    def __call__(self, *args, **kwargs):
        service = get_embedding_service()
        return service(*args, **kwargs)

embedding_service = EmbeddingServiceProxy()
