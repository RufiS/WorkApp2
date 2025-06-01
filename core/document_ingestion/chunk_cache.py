"""Chunk caching functionality for document ingestion"""

import time
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from core.config import performance_config
from utils.common.error_handler import CommonErrorHandler

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


class ChunkCache:
    """Manages caching of document chunks"""

    def __init__(self, cache_size: Optional[int] = None, enable_cache: Optional[bool] = None):
        """
        Initialize the chunk cache

        Args:
            cache_size: Maximum number of cache entries (None for config default)
            enable_cache: Whether caching is enabled (None for config default)
        """
        self.cache_size = cache_size or performance_config.chunk_cache_size
        self.enable_cache = enable_cache if enable_cache is not None else performance_config.enable_chunk_cache

        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"Chunk cache initialized: size={self.cache_size}, enabled={self.enable_cache}")

    def get_cache_key(self, file_path: str, chunk_size: int, chunk_overlap: int) -> str:
        """
        Generate a cache key for document chunks

        Args:
            file_path: Path to the document file
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks

        Returns:
            Cache key string
        """
        try:
            import os

            # Get file modification time
            mtime = os.path.getmtime(file_path)

            # Create a dictionary of chunking parameters
            params_dict = {
                "file_path": file_path,
                "mtime": mtime,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }

            # Convert to JSON and hash
            params_json = json.dumps(params_dict, sort_keys=True)
            return hashlib.md5(params_json.encode()).hexdigest()
        except Exception as e:
            error_msg = CommonErrorHandler.handle_processing_error("ChunkCache", "cache key generation", e)
            # Return a fallback key
            fallback = f"{file_path}:{chunk_size}:{chunk_overlap}"
            return hashlib.md5(fallback.encode()).hexdigest()

    def add_to_cache(self, key: str, chunks: List[Dict[str, Any]], ttl: Optional[float] = None) -> None:
        """
        Add chunks to the cache

        Args:
            key: Cache key
            chunks: Chunks to cache
            ttl: Time to live in seconds (None for default)
        """
        if not self.enable_cache:
            return

        try:
            # Create cache entry
            entry = ChunkCacheEntry(chunks=chunks, ttl=ttl or 3600 * 24)

            # Add to cache
            self.cache[key] = entry

            # Trim cache if needed
            self._trim_cache_if_needed(key, entry)

            logger.debug(f"Added {len(chunks)} chunks to cache with key {key[:8]}")
        except Exception as e:
            CommonErrorHandler.handle_processing_error("ChunkCache", "cache addition", e)

    def get_from_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get chunks from the cache

        Args:
            key: Cache key

        Returns:
            Cached chunks or None if not found
        """
        if not self.enable_cache:
            return None

        try:
            # Check if key exists in cache
            if key in self.cache:
                entry = self.cache[key]

                # Check if entry is expired
                if entry.is_expired():
                    # Remove expired entry
                    del self.cache[key]
                    self.cache_misses += 1
                    logger.debug(f"Cache entry expired for key {key[:8]}")
                    return None

                # Update timestamp to keep entry fresh
                entry.timestamp = time.time()
                self.cache_hits += 1
                logger.debug(f"Cache hit for key {key[:8]}")
                return entry.chunks

            self.cache_misses += 1
            logger.debug(f"Cache miss for key {key[:8]}")
            return None
        except Exception as e:
            CommonErrorHandler.handle_processing_error("ChunkCache", "cache retrieval", e)
            self.cache_misses += 1
            return None

    def _trim_cache_if_needed(self, new_key: str, new_entry: ChunkCacheEntry) -> None:
        """
        Trim cache if it exceeds the maximum size

        Args:
            new_key: Key of the newly added entry
            new_entry: The newly added entry
        """
        if len(self.cache) <= self.cache_size:
            return

        try:
            # Remove oldest entry
            if self.cache:
                oldest_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k].timestamp
                )
                del self.cache[oldest_key]
                logger.debug(f"Removed oldest cache entry with key {oldest_key[:8]}")
        except (ValueError, KeyError) as e:
            logger.warning(f"Error removing oldest cache entry: {str(e)}")
            # Fallback: remove a random entry to prevent memory leaks
            if len(self.cache) > self.cache_size and self.cache:
                try:
                    random_key = next(iter(self.cache.keys()))
                    del self.cache[random_key]
                    logger.debug(f"Removed random cache entry with key {random_key[:8]} as fallback")
                except (StopIteration, KeyError) as inner_e:
                    logger.warning(f"Cache management failed: {str(inner_e)}, clearing entire cache")
                    self.cache = {}
                    self.cache[new_key] = new_entry
                    logger.info(f"Cache reset with single entry for key {new_key[:8]}")

    def clear_cache(self) -> None:
        """Clear all cached entries"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        total_accesses = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.cache),
            "cache_max_size": self.cache_size,
            "hit_rate": hit_rate,
            "enabled": self.enable_cache
        }

    def cleanup_expired_entries(self) -> int:
        """
        Remove all expired entries from cache

        Returns:
            Number of entries removed
        """
        if not self.enable_cache:
            return 0

        try:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self.cache[key]

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)
        except Exception as e:
            CommonErrorHandler.handle_processing_error("ChunkCache", "expired entry cleanup", e)
            return 0
