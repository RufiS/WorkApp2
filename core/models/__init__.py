"""Modern Data Models package for WorkApp2.

This package provides Pydantic-based data models replacing dictionary usage
throughout the application for better type safety and validation.
"""

__version__ = "0.1.0"

from .document_models import DocumentModel, ChunkModel, MetadataModel
from .query_models import QueryRequest, QueryResponse, SearchResult
from .config_models import AppConfig, RetrievalConfig, PerformanceConfig

__all__ = [
    "DocumentModel", 
    "ChunkModel", 
    "MetadataModel",
    "QueryRequest", 
    "QueryResponse", 
    "SearchResult",
    "AppConfig", 
    "RetrievalConfig", 
    "PerformanceConfig"
]
