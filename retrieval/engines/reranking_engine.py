"""Reranking Search Engine for WorkApp2.

Handles advanced reranking using cross-encoder models for highest quality results.
Extracted from the monolithic retrieval_system.py for better maintainability.
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional

# Type hints for internal modules are allowed to use # type: ignore with reason
from core.config import retrieval_config, performance_config  # type: ignore[import] # TODO: Add proper config types
from core.document_processor import DocumentProcessor  # type: ignore[import] # TODO: Add proper types
from .vector_engine import VectorEngine
from utils.error_handling.enhanced_decorators import with_timing, with_error_tracking  # type: ignore[import] # TODO: Add proper types

logger = logging.getLogger(__name__)


class RerankingEngine:
    """Engine for reranking search results using cross-encoder models for enhanced quality."""

    def __init__(self, document_processor: DocumentProcessor, shared_vector_engine=None) -> None:
        """Initialize the reranking search engine.

        Args:
            document_processor: Document processor instance with loaded index
            shared_vector_engine: Optional shared VectorEngine instance to prevent duplicates
        """
        self.document_processor = document_processor
        # Use shared vector engine if provided, otherwise create new one (backwards compatibility)
        self.vector_engine = shared_vector_engine or VectorEngine(document_processor)
        self.similarity_threshold = retrieval_config.similarity_threshold
        self.logger = logger

        # Initialize metrics
        self.total_queries = 0
        self.query_times: List[float] = []
        self.reranking_times: List[float] = []
        self.max_query_times = 100

        # Initialize cross-encoder model (lazy loading)
        self._cross_encoder = None
        self._reranker_model = getattr(
            retrieval_config, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        engine_source = "shared" if shared_vector_engine else "new"
        self.logger.info(f"Reranking engine initialized with model: {self._reranker_model} ({engine_source} vector engine)")

    @property
    def cross_encoder(self):
        """Lazy load the cross-encoder model."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self._reranker_model)
                self.logger.info(f"Cross-encoder model loaded: {self._reranker_model}")
            except ImportError:
                self.logger.warning("CrossEncoder not available - falling back to vector search")
                self._cross_encoder = False  # Mark as unavailable
            except Exception as e:
                self.logger.error(f"Error loading cross-encoder model: {str(e)}")
                self._cross_encoder = False

        return self._cross_encoder if self._cross_encoder is not False else None

    @with_timing(threshold=1.0)  # Higher threshold for reranking as it's expected to be slower
    @with_error_tracking()
    def search(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_k: Optional[int] = None
    ) -> Tuple[str, float, int, List[float]]:
        """Perform search with reranking for highest quality results.

        Args:
            query: Query string to search for
            top_k: Number of final results to return
            rerank_top_k: Number of initial results to retrieve for reranking (defaults to top_k * 3)

        Returns:
            Tuple of (formatted context string, search time, number of chunks, retrieval scores)
        """
        start_time = time.time()
        rerank_top_k = rerank_top_k or (top_k * 6)  # Increased to 6x top_k for better workflow coverage

        query_preview = query[:50] + "..." if len(query) > 50 else query
        self.logger.info(f"Reranking search for query: '{query_preview}' with top_k={top_k}, rerank_top_k={rerank_top_k}")

        try:
            # Get initial results (more than needed for reranking)
            initial_results = self.document_processor.search(query, top_k=rerank_top_k)

            if not initial_results:
                self.logger.warning("No initial results found for reranking")
                return "No relevant information found.", time.time() - start_time, 0, []

            # If we have fewer results than requested, or reranking is unavailable, fall back to vector search
            if len(initial_results) <= top_k or not self.cross_encoder:
                if not self.cross_encoder:
                    self.logger.info("Cross-encoder unavailable, falling back to vector search")
                else:
                    self.logger.info(f"Too few results ({len(initial_results)}) for effective reranking")

                return self.vector_engine.search(query, top_k=top_k)

            # Perform reranking
            rerank_start = time.time()
            reranked_results = self._rerank_results(query, initial_results)
            rerank_time = time.time() - rerank_start

            self.reranking_times.append(rerank_time)
            if len(self.reranking_times) > self.max_query_times:
                self.reranking_times = self.reranking_times[-self.max_query_times:]

            # Take top_k reranked results
            results = reranked_results[:top_k]

            # Filter by similarity threshold if enabled
            if self.similarity_threshold > 0:
                pre_filter_count = len(results)
                results = [r for r in results if r.get("score", 0) >= self.similarity_threshold]
                filtered_count = pre_filter_count - len(results)

                self.logger.info(
                    f"Similarity filtering: {pre_filter_count} chunks → {len(results)} chunks "
                    f"(filtered {filtered_count} below threshold {self.similarity_threshold})"
                )

            # Update metrics
            search_time = time.time() - start_time
            # Ensure search time is never negative (can happen due to timing precision or system clock issues)
            search_time = max(0.0, search_time)
            self._update_metrics(search_time)

            # Format context using vector engine's formatting
            context = self.vector_engine._format_context(results)

            # Extract scores for progress tracking
            retrieval_scores = [result.get("score", 0.0) for result in results]

            # Log results
            self.logger.info(
                f"Reranking search complete: {len(results)} chunks, {len(context)} chars, "
                f"{search_time:.3f}s (rerank: {rerank_time:.3f}s)"
            )

            return context, search_time, len(results), retrieval_scores

        except Exception as e:
            self.logger.error(f"Error in reranking search: {str(e)}", exc_info=True)
            # Fall back to vector search
            self.logger.info("Falling back to vector search due to reranking error")
            return self.vector_engine.search(query, top_k)

    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using a cross-encoder model.

        Args:
            query: Query string
            results: Initial search results

        Returns:
            Reranked results with updated scores
        """
        try:
            if not self.cross_encoder:
                self.logger.warning("Cross-encoder not available for reranking")
                return results

            # Prepare query-text pairs for reranking
            pairs = [(query, result.get("text", "")) for result in results]

            self.logger.debug(f"Reranking {len(pairs)} query-document pairs")

            # Get scores from cross-encoder
            scores = self.cross_encoder.predict(pairs)

            # Update scores in results
            reranked_results = []
            for i, score in enumerate(scores):
                result = results[i].copy()
                result["original_score"] = result.get("score", 0.0)  # Keep original for debugging
                result["score"] = float(score)
                result["reranked"] = True
                reranked_results.append(result)

            # Sort by new reranking scores
            reranked_results.sort(key=lambda x: x["score"], reverse=True)

            # Log reranking effectiveness
            if len(reranked_results) > 0:
                top_original = max(r.get("original_score", 0) for r in results)
                top_reranked = reranked_results[0]["score"]
                self.logger.debug(
                    f"Reranking: top original score {top_original:.4f} → top reranked score {top_reranked:.4f}"
                )

            return reranked_results

        except ImportError:
            self.logger.warning("CrossEncoder not available for reranking")
            return results
        except Exception as e:
            self.logger.error(f"Error in reranking: {str(e)}", exc_info=True)
            return results

    def _update_metrics(self, search_time: float) -> None:
        """Update search performance metrics.

        Args:
            search_time: Time taken for the search in seconds
        """
        self.total_queries += 1
        self.query_times.append(search_time)

        # Keep only the most recent query times
        if len(self.query_times) > self.max_query_times:
            self.query_times = self.query_times[-self.max_query_times:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the reranking engine.

        Returns:
            Dictionary containing performance metrics
        """
        metrics = {"total_queries": self.total_queries}

        # Calculate query time statistics
        if self.query_times:
            metrics["avg_query_time"] = sum(self.query_times) / len(self.query_times)
            metrics["min_query_time"] = min(self.query_times)
            metrics["max_query_time"] = max(self.query_times)
        else:
            metrics["avg_query_time"] = 0.0
            metrics["min_query_time"] = 0.0
            metrics["max_query_time"] = 0.0

        # Calculate reranking time statistics
        if self.reranking_times:
            metrics["avg_reranking_time"] = sum(self.reranking_times) / len(self.reranking_times)
            metrics["min_reranking_time"] = min(self.reranking_times)
            metrics["max_reranking_time"] = max(self.reranking_times)
        else:
            metrics["avg_reranking_time"] = 0.0
            metrics["min_reranking_time"] = 0.0
            metrics["max_reranking_time"] = 0.0

        # Include vector engine metrics
        vector_metrics = self.vector_engine.get_metrics()
        metrics.update({f"vector_{k}": v for k, v in vector_metrics.items()})

        # Add model information
        metrics["reranker_model"] = self._reranker_model
        metrics["cross_encoder_available"] = self.cross_encoder is not None

        return metrics

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.total_queries = 0
        self.query_times = []
        self.reranking_times = []
        self.vector_engine.reset_metrics()
        self.logger.info("Reranking engine metrics reset")

    def is_available(self) -> bool:
        """Check if reranking is available (cross-encoder model loaded).

        Returns:
            True if reranking is available, False otherwise
        """
        return self.cross_encoder is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranking model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self._reranker_model,
            "available": self.is_available(),
            "loaded": self._cross_encoder is not None,
        }
