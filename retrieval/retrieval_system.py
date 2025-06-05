"""Unified Retrieval System Orchestrator (Refactored).

Transformed from 798-line monolith to ~100-line orchestrator using specialized engines.
Coordinates vector search, hybrid search, and reranking engines based on configuration.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple

# Type hints for internal modules are allowed to use # type: ignore with reason
from core.config import retrieval_config, performance_config  # type: ignore[import] # TODO: Add proper config types
from core.document_processor import DocumentProcessor  # type: ignore[import] # TODO: Add proper types
from .engines import VectorEngine, HybridEngine, RerankingEngine
from .services import MetricsService
from utils.error_handling.enhanced_decorators import with_timing, with_error_tracking  # type: ignore[import] # TODO: Add proper types
from utils.logging.retrieval_logger import retrieval_logger

logger = logging.getLogger(__name__)


class UnifiedRetrievalSystem:
    """Orchestrator for intelligent retrieval routing across specialized engines."""

    def __init__(self, document_processor: Optional[DocumentProcessor] = None) -> None:
        """Initialize the unified retrieval system.

        Args:
            document_processor: Document processor instance (creates new one if None)
        """
        self.document_processor = document_processor or DocumentProcessor()
        self.top_k = retrieval_config.top_k
        
        # Check for SPLADE flag
        self.use_splade = False

        # Initialize specialized engines with shared vector engine to prevent duplicates
        self.vector_engine = VectorEngine(self.document_processor)
        self.hybrid_engine = HybridEngine(self.document_processor, shared_vector_engine=self.vector_engine)
        self.reranking_engine = RerankingEngine(self.document_processor, shared_vector_engine=self.vector_engine)
        
        # Initialize SPLADE engine if available
        self.splade_engine = None
        try:
            from .engines import SpladeEngine
            self.splade_engine = SpladeEngine(self.document_processor)
            logger.info("SPLADE engine initialized and available")
        except ImportError as e:
            logger.info("SPLADE engine not available - transformers library may not be installed")
        except Exception as e:
            logger.warning(f"SPLADE engine initialization failed: {e}")

        # Initialize metrics service
        self.metrics_service = MetricsService(
            self.document_processor, self.vector_engine,
            self.hybrid_engine, self.reranking_engine
        )

        logger.info(f"Unified retrieval system initialized with intelligent routing (top_k={self.top_k})")

    @with_timing(threshold=0.5)
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Tuple[str, float, int, List[float]]:
        """Intelligently route retrieval based on configuration settings.

        Args:
            query: Query string
            top_k: Number of top results to return (None for default)

        Returns:
            Tuple of (formatted context string, retrieval time, number of chunks, retrieval scores)
        """
        # Use default top_k if not specified
        top_k = top_k or self.top_k

        # Create configuration snapshot for enhanced logging
        config_snapshot = {
            "similarity_threshold": retrieval_config.similarity_threshold,
            "top_k": top_k,
            "enhanced_mode": retrieval_config.enhanced_mode,
            "enable_reranking": performance_config.enable_reranking,
            "vector_weight": getattr(retrieval_config, 'vector_weight', 0.7),
            "chunk_size": retrieval_config.chunk_size,
            "chunk_overlap": retrieval_config.chunk_overlap,
            "max_context_length": retrieval_config.max_context_length
        }

        # Log the query and routing decision
        query_preview = query[:50] + "..." if len(query) > 50 else query

        # Determine engine and routing logic
        if self.use_splade and self.splade_engine is not None:
            selected_engine = "splade"
            routing_reason = "--splade flag enabled (experimental sparse+dense hybrid)"

            # Log engine routing decision (detailed info goes to file only)
            retrieval_logger.log_engine_routing_decision(
                query=query,
                available_engines=["vector", "hybrid", "reranking", "splade"],
                selected_engine=selected_engine,
                config_state=config_snapshot,
                routing_logic=routing_reason
            )

            # Execute search
            results = self.splade_engine.search(query, top_k)

        elif performance_config.enable_reranking:
            selected_engine = "reranking"
            routing_reason = "enable_reranking=True in performance_config"

            # Log engine routing decision (detailed info goes to file only)
            retrieval_logger.log_engine_routing_decision(
                query=query,
                available_engines=["vector", "hybrid", "reranking"],
                selected_engine=selected_engine,
                config_state=config_snapshot,
                routing_logic=routing_reason
            )

            # Execute search
            results = self.reranking_engine.search(query, top_k)

        elif retrieval_config.enhanced_mode:
            selected_engine = "hybrid"
            routing_reason = f"enhanced_mode=True, vector_weight={config_snapshot['vector_weight']}"

            # Log engine routing decision (detailed info goes to file only)
            retrieval_logger.log_engine_routing_decision(
                query=query,
                available_engines=["vector", "hybrid", "reranking"],
                selected_engine=selected_engine,
                config_state=config_snapshot,
                routing_logic=routing_reason
            )

            # Execute search
            results = self.hybrid_engine.search(query, top_k)

        else:
            selected_engine = "vector"
            routing_reason = "enhanced_mode=False, reranking=False (basic vector search)"

            # Log engine routing decision (detailed info goes to file only)
            retrieval_logger.log_engine_routing_decision(
                query=query,
                available_engines=["vector", "hybrid", "reranking"],
                selected_engine=selected_engine,
                config_state=config_snapshot,
                routing_logic=routing_reason
            )

            # Execute search
            results = self.vector_engine.search(query, top_k)

        # Assess context quality
        context, retrieval_time, chunk_count, similarity_scores = results
        context_quality = retrieval_logger.assess_context_quality(
            query=query,
            retrieved_context=context,
            chunk_scores=similarity_scores
        )

        # Log comprehensive retrieval operation
        session_id = retrieval_logger.log_retrieval_operation(
            query=query,
            engine_used=selected_engine,
            config_snapshot=config_snapshot,
            retrieval_results=results,
            context_quality_score=context_quality,
            routing_reason=routing_reason
        )

        logger.info(f"RETRIEVAL COMPLETED: {session_id} - Quality: {context_quality:.2f}, Engine: {selected_engine}")

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all engines."""
        return self.metrics_service.get_metrics()

    def _get_current_method(self) -> str:
        """Get the currently active search method based on configuration.

        Returns:
            String describing the current search method
        """
        if performance_config.enable_reranking:
            return "reranking"
        elif retrieval_config.enhanced_mode:
            return "hybrid"
        else:
            return "vector"



    # Additional convenience methods for advanced use cases
    def retrieve_with_reranking(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None
    ) -> Tuple[str, float, int, List[float]]:
        """Force reranking retrieval regardless of configuration.

        Args:
            query: Query string
            top_k: Number of top results to return
            rerank_top_k: Number of results to rerank

        Returns:
            Tuple of (formatted context string, retrieval time, number of chunks, retrieval scores)
        """
        return self.reranking_engine.search(query, top_k or self.top_k, rerank_top_k)

    def retrieve_with_hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Tuple[str, float, int, List[float]]:
        """Force hybrid search regardless of configuration.

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            Tuple of (formatted context string, retrieval time, number of chunks, retrieval scores)
        """
        return self.hybrid_engine.search(query, top_k or self.top_k)



    def reset_metrics(self) -> None:
        """Reset metrics for all engines."""
        self.metrics_service.reset_metrics()

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about available engines and their status."""
        return self.metrics_service.get_engine_info()
