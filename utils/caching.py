import time
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ChunkCacheEntry:
    """Cache entry for document chunks"""

    chunks: List[Dict[str, Any]]
    timestamp: float = field(default_factory=time.time)
    ttl: float = 3600 * 24  # Time to live in seconds (default: 24 hours)

    def is_expired(self) -> bool:
        """Check if the cache entry is expired"""
        return time.time() - self.timestamp > self.ttl


def get_cache_key(file_path: str, chunk_size: int, chunk_overlap: int) -> str:
    """
    Generate a cache key for document chunks

    Args:
        file_path: Path to the document file
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks

    Returns:
        Cache key string
    """
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


def add_to_cache(
    cache: Dict[str, Any], key: str, chunks: List[Dict[str, Any]], ttl: Optional[float] = None
) -> None:
    """
    Add chunks to the cache

    Args:
        cache: The cache dictionary to add to
        key: Cache key
        chunks: Chunks to cache
        ttl: Time to live in seconds (None for default)
    """
    # Create cache entry
    entry = ChunkCacheEntry(chunks=chunks, ttl=ttl or 3600 * 24)

    # Add to cache
    cache[key] = entry

    # Trim cache if needed
    if len(cache) > 1000:  # Configurable cache size
        # Remove oldest entry
        try:
            if cache:
                oldest_key = min(cache.keys(), key=lambda k: cache[k].timestamp)
                del cache[oldest_key]
        except (ValueError, KeyError):
            # This would happen if the cache became empty during processing
            # or if the key was already removed by another thread
            # Fallback: clear a random entry if cache is still too large
            if len(cache) > 1000 and cache:  # Configurable cache size
                try:
                    random_key = next(iter(cache.keys()))
                    del cache[random_key]
                except (StopIteration, KeyError):
                    # Last resort: clear the entire cache if we can't remove a single entry
                    # Create a new cache with just the current entry to prevent memory leaks
                    cache = {key: entry}
        except Exception as e:
            # Log the error and continue with the current entry
            print(f"Error removing oldest cache entry: {str(e)}")


def get_from_cache(cache: Dict[str, Any], key: str) -> Optional[List[Dict[str, Any]]]:
    """
    Get chunks from the cache

    Args:
        cache: The cache dictionary to check
        key: Cache key

    Returns:
        Cached chunks or None if not found
    """
    # Check if key exists in cache
    if key in cache:
        entry = cache[key]

        # Check if entry is expired
        if entry.is_expired():
            # Remove expired entry
            del cache[key]
            return None

        # Update timestamp to keep entry fresh
        entry.timestamp = time.time()
        return entry.chunks

    return None
