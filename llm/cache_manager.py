"""
LLM response cache management

Extracted from llm/llm_service.py
"""
import time
import hashlib
import json
import logging
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMCacheEntry:
    """Cache entry for LLM responses"""

    response: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    ttl: float = 3600  # Time to live in seconds (default: 1 hour)

    def is_expired(self) -> bool:
        """Check if the cache entry is expired"""
        return time.time() - self.timestamp > self.ttl


class CacheManager:
    """Manages LLM response caching"""

    def __init__(self, cache_size: int = 100, enable_cache: bool = True):
        """
        Initialize cache manager

        Args:
            cache_size: Maximum number of entries to cache
            enable_cache: Whether caching is enabled
        """
        self.response_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size = cache_size
        self.enable_cache = enable_cache

        logger.info(f"Cache manager initialized with size {self.cache_size}")

    def get_cache_key(
        self, prompt: str, model: str, max_tokens: int, temperature: Optional[float]
    ) -> str:
        """
        Generate a cache key for a request

        Args:
            prompt: The prompt string
            model: The model name
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation

        Returns:
            Cache key string
        """
        # Create a dictionary of request parameters
        request_dict = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Convert to JSON and hash
        request_json = json.dumps(request_dict, sort_keys=True)
        return hashlib.md5(request_json.encode()).hexdigest()

    def add_to_cache(
        self, key: str, response: Dict[str, Any], ttl: Optional[float] = None
    ) -> None:
        """
        Add a response to the cache

        Args:
            key: Cache key
            response: Response to cache
            ttl: Time to live in seconds (None for default)
        """
        if not self.enable_cache:
            return

        # Create cache entry
        entry = LLMCacheEntry(response=response, ttl=ttl or 3600)

        # Add to cache
        self.response_cache[key] = entry

        # Trim cache if needed
        if len(self.response_cache) > self.cache_size:
            # Remove oldest entry
            if self.response_cache:
                try:
                    oldest_key = min(
                        self.response_cache.keys(), key=lambda k: self.response_cache[k].timestamp
                    )
                    del self.response_cache[oldest_key]
                except ValueError:
                    # This would happen if the cache became empty during processing
                    # Log the issue and ensure we don't have a memory leak
                    logger.warning("Cache became empty during trimming - possible race condition")

                    # Force trim the cache to prevent memory leaks
                    while len(self.response_cache) > self.cache_size and self.response_cache:
                        # Just remove any key if we can't determine the oldest
                        try:
                            some_key = next(iter(self.response_cache.keys()))
                            del self.response_cache[some_key]
                            logger.info(f"Removed random cache entry to prevent memory leak")
                        except (StopIteration, RuntimeError) as e:
                            # If we still can't trim, clear the cache as a last resort
                            logger.warning(
                                f"Failed to trim cache properly: {str(e)}. Clearing cache."
                            )
                            self.response_cache.clear()
                            break

                    # Log to central error log
                    try:
                        from utils.config import app_config
                        from utils.error_logging import log_error

                        log_error(
                            "Cache became empty during trimming - possible race condition",
                            error_type="WARNING",
                            source="CacheManager.add_to_cache",
                        )
                    except ImportError:
                        # Fallback to simple file logging
                        try:
                            from utils.config import resolve_path

                            fallback_log_path = resolve_path(
                                "./logs/workapp_errors.log", create_dir=True
                            )
                        except ImportError:
                            fallback_log_path = "./logs/workapp_errors.log"
                            os.makedirs(os.path.dirname(fallback_log_path), exist_ok=True)

                        with open(fallback_log_path, "a") as error_log:
                            error_log.write(
                                f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - Cache became empty during trimming - possible race condition\n"
                            )

    def get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a response from the cache

        Args:
            key: Cache key

        Returns:
            Cached response or None if not found
        """
        if not self.enable_cache:
            return None

        # Check if key exists in cache
        if key in self.response_cache:
            entry = self.response_cache[key]

            # Check if entry is expired
            if entry.is_expired():
                # Remove expired entry
                del self.response_cache[key]
                self.cache_misses += 1
                return None

            # Update timestamp to keep entry fresh
            entry.timestamp = time.time()
            self.cache_hits += 1
            return entry.response

        self.cache_misses += 1
        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        total_cache_accesses = self.cache_hits + self.cache_misses
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.response_cache),
            "cache_max_size": self.cache_size,
            "cache_hit_rate": self.cache_hits / total_cache_accesses if total_cache_accesses > 0 else 0.0
        }

    def clear_cache(self) -> None:
        """Clear all cache entries"""
        self.response_cache.clear()
        logger.info("Cache cleared")

    def remove_expired_entries(self) -> int:
        """
        Remove all expired entries from cache
        
        Returns:
            Number of entries removed
        """
        if not self.enable_cache:
            return 0
            
        expired_keys = []
        for key, entry in self.response_cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.response_cache[key]
            
        if expired_keys:
            logger.info(f"Removed {len(expired_keys)} expired cache entries")
            
        return len(expired_keys)
