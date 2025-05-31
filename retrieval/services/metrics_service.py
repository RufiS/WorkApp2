"""Metrics Service for Retrieval System.

Handles aggregation and management of metrics from all retrieval engines.
Extracted from UnifiedRetrievalSystem to reduce complexity.
"""

import logging
from typing import Dict, Any

# Type hints for internal modules are allowed to use # type: ignore with reason
from core.config import performance_config  # type: ignore[import] # TODO: Add proper config types

logger = logging.getLogger(__name__)


class MetricsService:
    """Service for aggregating and managing retrieval metrics."""
    
    def __init__(self, document_processor, vector_engine, hybrid_engine, reranking_engine):
        """Initialize metrics service with engine references.
        
        Args:
            document_processor: Document processor instance
            vector_engine: Vector search engine instance
            hybrid_engine: Hybrid search engine instance 
            reranking_engine: Reranking engine instance
        """
        self.document_processor = document_processor
        self.vector_engine = vector_engine
        self.hybrid_engine = hybrid_engine
        self.reranking_engine = reranking_engine
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all engines.
        
        Returns:
            Dictionary with retrieval system metrics
        """
        # Get metrics from document processor
        processor_metrics = self.document_processor.get_metrics()
        
        # Get metrics from all engines
        vector_metrics = self.vector_engine.get_metrics()
        hybrid_metrics = self.hybrid_engine.get_metrics() 
        reranking_metrics = self.reranking_engine.get_metrics()
        
        return {
            "processor": processor_metrics,
            "vector_engine": vector_metrics,
            "hybrid_engine": hybrid_metrics,
            "reranking_engine": reranking_metrics,
            "routing": {
                "reranking_enabled": performance_config.enable_reranking,
                "enhanced_mode": performance_config.enhanced_mode if hasattr(performance_config, 'enhanced_mode') else False,
                "current_method": self._get_current_method(),
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset metrics for all engines."""
        self.vector_engine.reset_metrics()
        self.hybrid_engine.reset_metrics()
        self.reranking_engine.reset_metrics()
        logger.info("All retrieval engine metrics reset")
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about available engines and their status.
        
        Returns:
            Dictionary with engine status information
        """
        return {
            "vector_engine": {"available": True, "description": "Basic vector similarity search"},
            "hybrid_engine": {"available": True, "description": "Vector + keyword search combination"},
            "reranking_engine": {
                "available": self.reranking_engine.is_available(),
                "description": "Cross-encoder reranking for highest quality",
                "model_info": self.reranking_engine.get_model_info(),
            },
            "current_method": self._get_current_method(),
        }
    
    def _get_current_method(self) -> str:
        """Get the currently active search method based on configuration.
        
        Returns:
            String describing the current search method
        """
        if performance_config.enable_reranking:
            return "reranking"
        elif hasattr(performance_config, 'enhanced_mode') and performance_config.enhanced_mode:
            return "hybrid"
        else:
            return "vector"
