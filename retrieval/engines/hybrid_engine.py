"""Hybrid Search Engine for WorkApp2.

Handles hybrid search combining vector similarity and keyword matching.
Extracted from the monolithic retrieval_system.py for better maintainability.
"""

import logging
import time
from typing import List, Dict, Any, Tuple

# Type hints for internal modules are allowed to use # type: ignore with reason
from core.config import retrieval_config  # type: ignore[import] # TODO: Add proper config types
from core.document_processor import DocumentProcessor  # type: ignore[import] # TODO: Add proper types
from .vector_engine import VectorEngine
from utils.error_handling.enhanced_decorators import with_timing, with_error_tracking  # type: ignore[import] # TODO: Add proper types

logger = logging.getLogger(__name__)


class HybridEngine:
    """Engine for hybrid search combining vector similarity and keyword matching."""

    def __init__(self, document_processor: DocumentProcessor, shared_vector_engine=None) -> None:
        """Initialize the hybrid search engine.

        Args:
            document_processor: Document processor instance with loaded index
            shared_vector_engine: Optional shared VectorEngine instance to prevent duplicates
        """
        self.document_processor = document_processor
        # Use shared vector engine if provided, otherwise create new one (backwards compatibility)
        self.vector_engine = shared_vector_engine or VectorEngine(document_processor)
        self.similarity_threshold = retrieval_config.similarity_threshold
        self.vector_weight = retrieval_config.vector_weight
        self.logger = logger

        # Initialize metrics
        self.total_queries = 0
        self.query_times: List[float] = []
        self.max_query_times = 100

        engine_source = "shared" if shared_vector_engine else "new"
        self.logger.info(f"Hybrid search engine initialized with vector_weight={self.vector_weight} ({engine_source} vector engine)")

    @with_timing(threshold=0.5)
    @with_error_tracking()
    def search(self, query: str, top_k: int = 5) -> Tuple[str, float, int, List[float]]:
        """Perform hybrid search combining vector and keyword approaches.

        Args:
            query: Query string to search for
            top_k: Number of top results to return

        Returns:
            Tuple of (formatted context string, search time, number of chunks, retrieval scores)
        """
        start_time = time.time()
        query_preview = query[:50] + "..." if len(query) > 50 else query

        self.logger.info(f"Hybrid search for query: '{query_preview}' with top_k={top_k}")

        try:
            # Check if index exists and is loaded
            if not self.document_processor.has_index():
                self.logger.warning("No index has been built. Process documents first.")
                return "No relevant information found.", time.time() - start_time, 0, []

            # Get vector search results (more than needed for hybrid combination)
            vector_results = self.document_processor.search(query, top_k=top_k * 2)

            # Get keyword search results
            keyword_results = self._perform_keyword_search(query, top_k * 2)

            # Combine and rerank results based on vector weight
            hybrid_results = self._combine_search_results(vector_results, keyword_results, self.vector_weight)

            # Take top_k results
            results = hybrid_results[:top_k]

            # Filter by similarity threshold if enabled
            if self.similarity_threshold > 0:
                pre_filter_count = len(results)
                results = [r for r in results if r.get("score", 0) >= self.similarity_threshold]
                filtered_count = pre_filter_count - len(results)

                self.logger.debug(
                    f"Similarity filtering: {pre_filter_count} chunks → {len(results)} chunks "
                    f"(filtered {filtered_count} below threshold {self.similarity_threshold})"
                )

            # Update metrics
            search_time = time.time() - start_time
            self._update_metrics(search_time)

            # Format context using vector engine's formatting
            context = self.vector_engine._format_context(results)

            # Extract scores for progress tracking
            retrieval_scores = [result.get("score", 0.0) for result in results]

            # Log results
            self.logger.info(
                f"Hybrid search complete: {len(results)} chunks, {len(context)} chars, {search_time:.3f}s "
                f"(vector_weight={self.vector_weight})"
            )

            return context, search_time, len(results), retrieval_scores

        except Exception as e:
            self.logger.error(f"Error in hybrid search: {str(e)}", exc_info=True)
            # Fall back to vector search only
            self.logger.info("Falling back to vector search only")
            return self.vector_engine.search(query, top_k)

    def _perform_keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform keyword-based search on the document chunks.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of search results with keyword-based scores
        """
        try:
            # Get all chunks from the document processor
            if not hasattr(self.document_processor, 'chunks'):
                self.logger.warning("Document processor has no 'chunks' attribute")
                return []

            if not self.document_processor.chunks:
                self.logger.warning("Document processor chunks is empty or None")
                return []

            # Simple keyword matching with TF-IDF-like scoring
            query_words = set(query.lower().split())
            scored_results = []

            for chunk in self.document_processor.chunks:
                # Handle both string and dict chunk formats
                if isinstance(chunk, str):
                    chunk_text = chunk.lower()
                    chunk_dict = {"text": chunk, "metadata": {}}
                elif isinstance(chunk, dict):
                    chunk_text = chunk.get("text", "").lower()
                    chunk_dict = chunk
                else:
                    self.logger.warning(f"Unexpected chunk type: {type(chunk)}, skipping")
                    continue

                chunk_words = set(chunk_text.split())

                # Calculate keyword overlap score
                if query_words and chunk_words:
                    intersection = len(query_words.intersection(chunk_words))
                    keyword_score = intersection / len(query_words) if query_words else 0.0

                    # Boost score if query words appear multiple times
                    for word in query_words:
                        if word in chunk_text:
                            keyword_score += chunk_text.count(word) * 0.1

                    if keyword_score > 0:
                        result = chunk_dict.copy()
                        result["score"] = keyword_score
                        scored_results.append(result)

            # Sort by keyword score and return top results
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = scored_results[:top_k]

            self.logger.debug(f"Keyword search found {len(scored_results)} matches, returning top {len(final_results)}")

            return final_results

        except Exception as e:
            self.logger.error(f"Error in keyword search: {str(e)}", exc_info=True)
            return []

    def _combine_search_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        vector_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine vector and keyword search results using weighted scoring.

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            vector_weight: Weight for vector scores (0.0 to 1.0)

        Returns:
            Combined and ranked results
        """
        keyword_weight = 1.0 - vector_weight
        combined_scores = {}
        all_results = {}

        # Normalize vector scores (assuming they're cosine similarities between 0 and 1)
        if vector_results:
            max_vector_score = max(r.get("score", 0) for r in vector_results)
            min_vector_score = min(r.get("score", 0) for r in vector_results)
            vector_range = max_vector_score - min_vector_score if max_vector_score != min_vector_score else 1.0
        else:
            min_vector_score = 0
            vector_range = 1.0

        # Process vector results
        for result in vector_results:
            chunk_id = id(result.get("text", ""))  # Use text as unique identifier
            normalized_score = (result.get("score", 0) - min_vector_score) / vector_range if vector_results else 0
            combined_scores[chunk_id] = vector_weight * normalized_score
            all_results[chunk_id] = result.copy()

        # Normalize keyword scores
        if keyword_results:
            max_keyword_score = max(r.get("score", 0) for r in keyword_results)
            min_keyword_score = min(r.get("score", 0) for r in keyword_results)
            keyword_range = max_keyword_score - min_keyword_score if max_keyword_score != min_keyword_score else 1.0
        else:
            min_keyword_score = 0
            keyword_range = 1.0

        # Process keyword results
        for result in keyword_results:
            chunk_id = id(result.get("text", ""))
            normalized_score = (result.get("score", 0) - min_keyword_score) / keyword_range if keyword_results else 0

            if chunk_id in combined_scores:
                # Combine with existing vector score
                combined_scores[chunk_id] += keyword_weight * normalized_score
            else:
                # Only keyword score
                combined_scores[chunk_id] = keyword_weight * normalized_score
                all_results[chunk_id] = result.copy()

        # Create final results with combined scores
        final_results = []
        for chunk_id, combined_score in combined_scores.items():
            result = all_results[chunk_id]
            result["score"] = combined_score
            result["hybrid_score"] = combined_score  # Keep original for debugging
            final_results.append(result)

        # Sort by combined score
        final_results.sort(key=lambda x: x["score"], reverse=True)

        self.logger.debug(f"Combined {len(vector_results)} vector + {len(keyword_results)} keyword results into {len(final_results)} hybrid results")

        return final_results

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
        """Get performance metrics for the hybrid engine.

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

        # Include vector engine metrics
        vector_metrics = self.vector_engine.get_metrics()
        metrics.update({f"vector_{k}": v for k, v in vector_metrics.items()})

        return metrics

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.total_queries = 0
        self.query_times = []
        self.vector_engine.reset_metrics()
        self.logger.info("Hybrid engine metrics reset")
