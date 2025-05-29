"""Document Ingestion Module - Backward Compatibility Layer

This module now imports from the refactored document ingestion components
for better modularity while maintaining backward compatibility.
"""

import logging
from typing import List, Dict, Any, Set

# Import from the new modular structure
from .document_ingestion.ingestion_manager import DocumentIngestion
from .document_ingestion.chunk_cache import ChunkCacheEntry

# Setup logging
logger = logging.getLogger(__name__)

# Re-export main classes for backward compatibility
__all__ = ['DocumentIngestion', 'ChunkCacheEntry']

logger.info("Document ingestion module loaded with new modular architecture")
