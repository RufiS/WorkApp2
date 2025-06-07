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

    def _get_available_vram_mb(self) -> int:
        """Get available VRAM in MB"""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0
            
            # Get VRAM stats
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            reserved_memory = torch.cuda.memory_reserved(0)
            
            # Calculate available as total - max(allocated, reserved)
            used_memory = max(allocated_memory, reserved_memory)
            available_memory = total_memory - used_memory
            
            # Convert to MB
            available_mb = int(available_memory / (1024 * 1024))
            
            return max(0, available_mb)
        except Exception as e:
            logger.warning(f"Error getting VRAM info: {e}")
            return 0

    def _get_available_host_memory_mb(self) -> int:
        """Get available host memory in MB (important for CUDA host allocations)"""
        try:
            import psutil
            # Get available system memory
            memory_info = psutil.virtual_memory()
            available_mb = int(memory_info.available / (1024 * 1024))
            return available_mb
        except ImportError:
            logger.warning("psutil not available, cannot check host memory")
            return 4096  # Assume 4GB available
        except Exception as e:
            logger.warning(f"Error getting host memory info: {e}")
            return 4096

    def _aggressive_gpu_cleanup(self) -> None:
        """Perform aggressive GPU memory cleanup before FAISS operations"""
        try:
            import torch
            if not torch.cuda.is_available():
                return
                
            logger.info("Performing aggressive GPU cleanup before FAISS operations")
            
            # Multiple rounds of cleanup
            for i in range(3):
                torch.cuda.empty_cache()
                gc.collect()
                
            # Force synchronization
            torch.cuda.synchronize()
            
            # Log memory status after cleanup
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            logger.info(f"Post-cleanup GPU memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
            
        except Exception as e:
            logger.warning(f"Error during aggressive GPU cleanup: {e}")

    def _warmup_gpu_index(self, gpu_index: faiss.Index) -> None:
        """
        Warm up GPU index to eliminate first-search latency
        
        Performs dummy searches to initialize GPU kernels and eliminate 
        the cold-start penalty that causes first searches to be slow.
        
        Args:
            gpu_index: GPU FAISS index to warm up
        """
        try:
            if not hasattr(gpu_index, 'getDevice') or gpu_index.getDevice() < 0:
                logger.debug("Index is not on GPU, skipping warmup")
                return
                
            if gpu_index.ntotal == 0:
                logger.debug("Index is empty, skipping warmup")
                return
                
            logger.info("Warming up GPU index to eliminate first-search latency")
            
            # Get index dimension
            index_dim = gpu_index.d
            
            # Generate a few dummy queries for warmup (OPTIMIZED - minimal searches)
            import numpy as np
            
            # Single quick warmup search with small top_k
            dummy_query = np.random.rand(1, index_dim).astype(np.float32)
            
            try:
                # Single fast search to warm up kernels
                _, _ = gpu_index.search(dummy_query, min(2, gpu_index.ntotal))
                logger.debug("Quick GPU warmup search completed")
            except Exception as warmup_error:
                logger.warning(f"GPU warmup search failed: {warmup_error}")
            
            logger.info("✅ GPU index warmup completed - first search should now be fast")
            
        except Exception as e:
            logger.warning(f"GPU index warmup failed: {e} - first search may be slow")
            # Don't raise exception, warmup is optional optimization

    def initialize_resources(self) -> bool:
        """
        Initialize GPU resources for FAISS operations with aggressive memory management
        
        Addresses the specific CUDA async copy buffer allocation failure by:
        1. Aggressive memory cleanup before allocation
        2. More conservative VRAM accounting with safety buffers
        3. Multi-tier fallback system
        4. Host memory validation

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if not self.gpu_available:
                logger.debug("GPU not available, skipping GPU resource initialization")
                return False

            if self.gpu_resources is None:
                logger.info("Initializing GPU resources for FAISS with aggressive memory management")
                
                # PHASE 1: Aggressive cleanup to maximize available memory
                self._aggressive_gpu_cleanup()
                
                # PHASE 2: Check both VRAM and host memory (critical for async copy buffer)
                available_vram_mb = self._get_available_vram_mb()
                available_host_mb = self._get_available_host_memory_mb()
                
                logger.info(f"Memory status: {available_vram_mb}MB VRAM, {available_host_mb}MB host memory available")
                
                # Check if we have enough host memory for FAISS async copy buffer (needs ~256MB)
                if available_host_mb < 512:  # Need 512MB host memory as safety buffer
                    logger.error(f"Insufficient host memory ({available_host_mb}MB) for CUDA async copy buffer, need 512MB+")
                    return False
                
                if available_vram_mb < 200:  # Increased minimum from 100MB to 200MB
                    logger.warning(f"Insufficient VRAM available ({available_vram_mb}MB), skipping GPU initialization")
                    return False
                
                # PHASE 3: Initialize GPU resources with retry logic
                for attempt in range(3):  # Up to 3 attempts with progressively smaller allocations
                    try:
                        if attempt > 0:
                            logger.info(f"GPU resource initialization attempt {attempt + 1}/3")
                            # Additional cleanup between attempts
                            self._aggressive_gpu_cleanup()
                        
                        self.gpu_resources = faiss.StandardGpuResources()

                        # AGGRESSIVE SCALING: Use your suggested range (16MB - 2GB+)
                        if available_vram_mb >= 16000:    # 16GB+ available - High-end cards
                            temp_memory = 2048 * 1024 * 1024  # 2GB for maximum performance
                        elif available_vram_mb >= 12000:  # 12-16GB available
                            temp_memory = 1536 * 1024 * 1024  # 1.5GB
                        elif available_vram_mb >= 8000:   # 8-12GB available
                            temp_memory = 1024 * 1024 * 1024  # 1GB
                        elif available_vram_mb >= 6000:   # 6-8GB available
                            temp_memory = 768 * 1024 * 1024   # 768MB
                        elif available_vram_mb >= 4000:   # 4-6GB available
                            temp_memory = 512 * 1024 * 1024   # 512MB
                        elif available_vram_mb >= 2000:   # 2-4GB available
                            temp_memory = 256 * 1024 * 1024   # 256MB
                        elif available_vram_mb >= 1000:   # 1-2GB available
                            temp_memory = 128 * 1024 * 1024   # 128MB
                        elif available_vram_mb >= 500:    # 500MB-1GB available
                            temp_memory = 64 * 1024 * 1024    # 64MB
                        else:  # < 500MB available
                            temp_memory = 32 * 1024 * 1024    # 32MB
                        
                        # Apply attempt-based reduction for retries
                        if attempt == 1:
                            temp_memory = temp_memory // 2  # Half memory on second attempt
                        elif attempt == 2:
                            temp_memory = 16 * 1024 * 1024   # Minimal 16MB on third attempt
                        
                        logger.info(f"Attempting GPU resource allocation with {temp_memory // (1024*1024)}MB temp memory (attempt {attempt + 1})")
                        
                        self.gpu_resources.setTempMemory(temp_memory)
                        
                        logger.info(f"✅ GPU resources initialized successfully with {temp_memory // (1024*1024)}MB temporary memory")
                        logger.info(f"Memory available: {available_vram_mb}MB VRAM, {available_host_mb}MB host")
                        return True
                        
                    except Exception as init_error:
                        logger.warning(f"GPU resource initialization attempt {attempt + 1} failed: {init_error}")
                        if self.gpu_resources is not None:
                            try:
                                self.gpu_resources = None
                                gc.collect()
                            except:
                                pass
                        
                        if attempt == 2:  # Last attempt failed
                            logger.error("All GPU resource initialization attempts failed")
                            self.gpu_resources = None
                            return False

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
        
        Enhanced with aggressive memory management to prevent async copy buffer failures

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

            # CRITICAL: Aggressive cleanup before GPU transfer to maximize available memory
            self._aggressive_gpu_cleanup()
            
            # Check memory availability before transfer
            available_vram_mb = self._get_available_vram_mb()
            available_host_mb = self._get_available_host_memory_mb()
            
            logger.info(f"Pre-transfer memory: {available_vram_mb}MB VRAM, {available_host_mb}MB host")
            
            # Retry logic for GPU transfer with multiple attempts
            for attempt in range(2):  # 2 attempts max
                try:
                    if attempt > 0:
                        logger.info(f"GPU transfer retry attempt {attempt + 1}")
                        self._aggressive_gpu_cleanup()  # Additional cleanup on retry
                    
                    logger.info("Moving index to GPU")
                    gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
                    logger.info("✅ Index successfully moved to GPU")
                    
                    # CRITICAL FIX: GPU warmup to prevent first-search latency
                    self._warmup_gpu_index(gpu_index)
                    
                    return gpu_index, True
                    
                except Exception as transfer_error:
                    logger.warning(f"GPU transfer attempt {attempt + 1} failed: {transfer_error}")
                    
                    if "failed to cudaHostAlloc" in str(transfer_error) or "out of memory" in str(transfer_error):
                        if attempt == 0:  # First attempt failed, try with smaller temp memory
                            logger.info("Reducing temp memory allocation for retry")
                            try:
                                # Reinitialize with minimal memory for problematic cases
                                self.cleanup_resources()
                                self.gpu_resources = faiss.StandardGpuResources()
                                minimal_memory = 16 * 1024 * 1024  # 16MB minimal
                                self.gpu_resources.setTempMemory(minimal_memory)
                                logger.info(f"Reinitialized GPU resources with minimal {minimal_memory // (1024*1024)}MB for transfer retry")
                                continue  # Try transfer again
                            except Exception as reinit_error:
                                logger.error(f"Failed to reinitialize GPU resources for retry: {reinit_error}")
                                break
                        else:
                            logger.error("GPU transfer failed on final attempt, falling back to CPU")
                            break
                    else:
                        logger.error(f"GPU transfer failed with non-memory error: {transfer_error}")
                        break

            # If all attempts failed, fall back to CPU
            logger.warning("GPU transfer failed, keeping index on CPU")
            return index, False

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
