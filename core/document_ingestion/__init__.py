"""Document ingestion module - split for better maintainability"""

from .ingestion_manager import DocumentIngestion
from .chunk_cache import ChunkCacheEntry, ChunkCache
from .file_processors import FileProcessor
from .deduplication_engine import DeduplicationEngine
from .metadata_handler import MetadataHandler
from .chunk_optimizer import ChunkOptimizer

__all__ = [
    'DocumentIngestion',
    'ChunkCacheEntry',
    'ChunkCache',
    'FileProcessor',
    'DeduplicationEngine',
    'MetadataHandler',
    'ChunkOptimizer'
]
