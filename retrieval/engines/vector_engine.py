"""Vector Search Engine for WorkApp2.

Handles basic vector-based similarity search using FAISS indexes.
Extracted from the monolithic retrieval_system.py for better maintainability.
"""

import logging
import time
import os
from typing import List, Dict, Any, Optional, Tuple

# Type hints for internal modules are allowed to use # type: ignore with reason
from core.config import retrieval_config  # type: ignore[import] # TODO: Add proper config types
from core.document_processor import DocumentProcessor  # type: ignore[import] # TODO: Add proper types
from utils.error_handling.enhanced_decorators import with_timing, with_error_tracking  # type: ignore[import] # TODO: Add proper types

logger = logging.getLogger(__name__)


class VectorEngine:
    """Engine for basic vector-based similarity search operations."""

    def __init__(self, document_processor: DocumentProcessor) -> None:
        """Initialize the vector search engine.

        Args:
            document_processor: Document processor instance with loaded index
        """
        self.document_processor = document_processor
        self.similarity_threshold = retrieval_config.similarity_threshold
        self.max_context_length = retrieval_config.max_context_length
        self.logger = logger

        # Initialize metrics
        self.total_queries = 0
        self.query_times: List[float] = []
        self.max_query_times = 100

        self.logger.info("Vector search engine initialized")

    @with_timing(threshold=0.5)
    @with_error_tracking()
    def search(self, query: str, top_k: int = 5) -> Tuple[str, float, int, List[float]]:
        """Perform vector-based similarity search.

        Args:
            query: Query string to search for
            top_k: Number of top results to return

        Returns:
            Tuple of (formatted context string, search time, number of chunks, retrieval scores)
        """
        start_time = time.time()

        # Log the search request
        query_preview = query[:50] + "..." if len(query) > 50 else query
        self.logger.info(f"Vector search for query: '{query_preview}' with top_k={top_k}")

        try:
            # Check if index exists and is loaded
            if not self.document_processor.has_index():
                self.logger.warning("No index has been built. Process documents first.")
                results = []
            else:
                # Perform the actual search
                results = self.document_processor.search(query, top_k=top_k)

                # Filter by similarity threshold if enabled
                if self.similarity_threshold > 0:
                    pre_filter_count = len(results)
                    results = [r for r in results if r.get("score", 0) >= self.similarity_threshold]
                    filtered_count = pre_filter_count - len(results)

                    self.logger.info(
                        f"Similarity filtering: {pre_filter_count} chunks → {len(results)} chunks "
                        f"(filtered {filtered_count} below threshold {self.similarity_threshold})"
                    )
                else:
                    self.logger.info(f"Retrieved {len(results)} chunks (no similarity threshold applied)")

        except ValueError as e:
            self.logger.warning(f"Search failed: {str(e)}")
            results = []
        except Exception as e:
            self.logger.error(f"Error in vector search: {str(e)}", exc_info=True)
            results = []

        # Run deduplication after retrieval
        if len(results) > 1:
            results = self._deduplicate_results(results)

        # Format context
        context = self._format_context(results)

        # Extract scores for progress tracking
        retrieval_scores = [result.get("score", 0.0) for result in results]

        # Calculate and log performance metrics
        search_time = time.time() - start_time
        self._update_metrics(search_time)

        # Log results
        if context:
            self.logger.info(
                f"Vector search complete: {len(results)} chunks, {len(context)} chars, {search_time:.3f}s"
            )
            if self.logger.isEnabledFor(logging.DEBUG):
                context_preview = context[:200] + "..." if len(context) > 200 else context
                self.logger.debug(f"Context preview: {context_preview}")
        else:
            self._log_empty_context_warning(query[:50])

        return context, search_time, len(results), retrieval_scores

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on text similarity.

        Args:
            results: List of search results

        Returns:
            List of deduplicated results
        """
        try:
            deduplicated_results = []
            seen_texts = set()
            duplicates_found = 0

            for result in results:
                text = result.get("text", "")
                # Create a simplified version for comparison (lowercase, whitespace normalized)
                simplified_text = " ".join(text.lower().split())

                # Skip if we've seen a very similar text (reduced threshold for workflow preservation)
                if any(self._calculate_text_similarity(simplified_text, seen) > 0.6 for seen in seen_texts):
                    self.logger.debug(f"Skipping duplicate chunk: {text[:50]}...")
                    duplicates_found += 1
                    continue

                # Add to results and mark as seen
                deduplicated_results.append(result)
                seen_texts.add(simplified_text)

            self.logger.info(
                f"Deduplication: {len(results)} chunks → {len(deduplicated_results)} chunks "
                f"(removed {duplicates_found} duplicates)"
            )

            return deduplicated_results

        except Exception as e:
            self.logger.warning(f"Error during deduplication: {str(e)}")
            return results  # Return original results if deduplication fails

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score between 0 and 1
        """
        # Convert texts to sets of words
        words1 = set(text1.split())
        words2 = set(text2.split())

        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into a context string.

        Args:
            results: List of search results

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."

        # Format each chunk with source information
        formatted_chunks = []
        total_length = 0

        for i, result in enumerate(results):
            # Format source information
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", "")
            page_info = f" (page {page})" if page else ""

            # Format chunk
            chunk_text = result.get("text", "No text available")
            formatted_chunk = f"[{i+1}] From {source}{page_info}:\n{chunk_text}\n"

            # Check if adding this chunk would exceed max context length
            if (
                self.max_context_length > 0
                and total_length + len(formatted_chunk) > self.max_context_length
            ):
                # If this is the first chunk, include it anyway (truncated)
                if i == 0:
                    truncated_length = max(0, self.max_context_length - total_length - 3)
                    formatted_chunk = (
                        formatted_chunk[:truncated_length] + "..."
                        if truncated_length > 0
                        else "..."
                    )
                    formatted_chunks.append(formatted_chunk)
                break

            # Add chunk
            formatted_chunks.append(formatted_chunk)
            total_length += len(formatted_chunk)

        # Join chunks
        return "\n".join(formatted_chunks)

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

    def _log_empty_context_warning(self, query_preview: str) -> None:
        """Log warning when no context is retrieved.

        Args:
            query_preview: Truncated query for logging
        """
        error_msg = f"Retrieved empty context for query: {query_preview}..."
        self.logger.warning(error_msg)

        # Log to central error log
        try:
            from utils.error_logging import log_error  # type: ignore[import] # TODO: Add proper types
            log_error(error_msg, include_traceback=False)
        except ImportError:
            # Fallback to simple file logging if error_logging module is not available
            try:
                from core.config import resolve_path  # type: ignore[import] # TODO: Add proper config types
                fallback_log_path = resolve_path(os.path.join(".", "logs", "workapp_errors.log"), create_dir=True)
            except ImportError:
                fallback_log_path = "./logs/workapp_errors.log"
                os.makedirs(os.path.dirname(fallback_log_path), exist_ok=True)

            with open(fallback_log_path, "a") as error_log:
                error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {error_msg}\n")

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the vector engine.

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

        return metrics

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.total_queries = 0
        self.query_times = []
        self.logger.info("Vector engine metrics reset")
