"""GPU Resource Management for FAISS Operations

Consolidates GPU handling across the codebase to eliminate redundancy.
Extracted from core/vector_index_engine.py
"""

import logging
import gc
from typing import Optional

import faiss

from core.config import performance_config
from utils.logging.error_logging import log_error, log_warning

# Setup logging
logger = logging.getLogger(__name__)


class GPUResourceManager:
    """Manages GPU resources for FAISS operations with proper cleanup"""

    def __init__(self):
        """Initialize GPU resource manager"""
        self.gpu_resources: Optional[faiss.StandardGpuResources] = None
        self.gpu_available = self._check_gpu_availability()
        
        if self.gpu_available:
            logger.info("GPU is available for FAISS operations")
        else:
            logger.info("GPU is not available, using CPU for FAISS operations")

    def __del__(self):
        """Destructor to ensure GPU resources are cleaned up"""
        try:
            self.cleanup_resources()
        except Exception:
            # Don't raise exceptions in destructor
            pass

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for embeddings"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def initialize_resources(self) -> bool:
        """
        Initialize GPU resources for FAISS operations

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if not self.gpu_available:
                logger.debug("GPU not available, skipping GPU resource initialization")
                return False

            if self.gpu_resources is None:
                logger.info("Initializing GPU resources for FAISS")
                self.gpu_resources = faiss.StandardGpuResources()

                # Configure memory allocations for better resource management
                # Set temporary memory to 256MB (can be adjusted based on available GPU memory)
                temp_memory = 256 * 1024 * 1024  # 256MB
                self.gpu_resources.setTempMemory(temp_memory)

                logger.info(
                    f"GPU resources initialized with {temp_memory // (1024*1024)}MB temporary memory"
                )
                return True

            return True

        except Exception as e:
            error_msg = f"Failed to initialize GPU resources: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg, include_traceback=True)
            # Don't raise exception, fall back to CPU
            self.gpu_resources = None
            logger.warning("Falling back to CPU-only FAISS operations")
            return False

    def cleanup_resources(self) -> None:
        """Explicitly cleanup GPU resources to prevent memory leaks"""
        try:
            # Clean up GPU resources
            if self.gpu_resources is not None:
                logger.info("Cleaning up GPU resources")
                # NOTE: faiss.StandardGpuResources doesn't have an explicit cleanup method
                # but setting to None should trigger garbage collection
                self.gpu_resources = None
                logger.info("GPU resources cleaned up successfully")

                # Force garbage collection to ensure GPU memory is released
                gc.collect()

                # Log GPU memory status if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        current_memory = torch.cuda.memory_allocated()
                        logger.debug(
                            f"GPU memory after cleanup: {current_memory / (1024*1024):.2f}MB"
                        )
                except ImportError:
                    pass

        except Exception as e:
            error_msg = f"Error during GPU resource cleanup: {str(e)}"
            logger.warning(error_msg)
            log_warning(error_msg, include_traceback=True)
            # Continue execution even if cleanup fails

    def move_index_to_gpu(self, index: faiss.Index) -> tuple[faiss.Index, bool]:
        """
        Move index to GPU if available and enabled

        Args:
            index: FAISS index to move to GPU

        Returns:
            Tuple of (index, success_flag)
        """
        try:
            if not self.gpu_available or not performance_config.use_gpu_for_faiss:
                return index, False

            if index is None:
                logger.warning("Cannot move None index to GPU")
                return index, False

            # Check if already on GPU
            if hasattr(index, "getDevice") and index.getDevice() >= 0:
                logger.debug("Index is already on GPU")
                return index, True

            # Initialize GPU resources if needed
            if self.gpu_resources is None:
                if not self.initialize_resources():
                    return index, False

            if self.gpu_resources is None:
                logger.warning("GPU resources not available, cannot move index to GPU")
                return index, False

            logger.info("Moving index to GPU")
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
            logger.info("Index successfully moved to GPU")
            return gpu_index, True

        except Exception as e:
            error_msg = f"Failed to move index to GPU: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg, include_traceback=True)
            return index, False

    def move_index_to_cpu(self, index: faiss.Index) -> tuple[faiss.Index, bool]:
        """
        Move index to CPU if it's currently on GPU

        Args:
            index: FAISS index to move to CPU

        Returns:
            Tuple of (index, success_flag)
        """
        try:
            if index is None:
                logger.debug("Index is None, cannot move to CPU")
                return index, True

            # Check if index is on GPU
            is_on_gpu = hasattr(index, "getDevice") and index.getDevice() >= 0
            if not is_on_gpu:
                logger.debug("Index is already on CPU")
                return index, True

            logger.info("Moving index from GPU to CPU")
            cpu_index = faiss.index_gpu_to_cpu(index)
            logger.info("Index successfully moved to CPU")
            return cpu_index, True

        except Exception as e:
            error_msg = f"Failed to move index to CPU: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg, include_traceback=True)
            return index, False

    def should_use_gpu(self) -> bool:
        """Check if GPU should be used for operations"""
        return self.gpu_available and performance_config.use_gpu_for_faiss

    def get_gpu_stats(self) -> dict:
        """Get GPU statistics"""
        stats = {
            "gpu_available": self.gpu_available,
            "gpu_enabled": performance_config.use_gpu_for_faiss,
            "resources_initialized": self.gpu_resources is not None,
        }

        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    stats["gpu_memory_allocated"] = torch.cuda.memory_allocated()
                    stats["gpu_memory_cached"] = torch.cuda.memory_reserved()
            except ImportError:
                pass

        return stats


# Global instance for shared use across the application
gpu_manager = GPUResourceManager()
