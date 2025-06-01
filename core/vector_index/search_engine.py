"""Vector Search Engine

Handles vector similarity search operations with result formatting.
Extracted from core/vector_index_engine.py
"""

import time
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import faiss

from core.config import performance_config
from core.embeddings.embedding_service import embedding_service
from utils.logging.error_logging import query_logger
from utils.logging.error_logging import log_error

# Setup logging
logger = logging.getLogger(__name__)


class SearchEngine:
    """Handles vector similarity search operations"""

    def __init__(self, embedding_dim: int):
        """
        Initialize search engine

        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        logger.info(f"Search engine initialized with dimension {embedding_dim}")

    def search(
        self,
        query: str,
        index: faiss.Index,
        texts: List[Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using the query

        Args:
            query: Query string
            index: FAISS index to search
            texts: List of text chunks corresponding to index
            top_k: Number of top results to return

        Returns:
            List of relevant chunks with scores

        Raises:
            ValueError: If parameters are invalid or index is not ready
        """
        # Validate inputs
        self._validate_search_inputs(query, index, texts, top_k)

        # Log query for instrumentation
        start_time = time.time()
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"Searching for query: '{query_preview}' with top_k={top_k}")

        # Initialize metrics
        search_type = "vector"
        fallback_used = False

        try:
            # Embed query
            query_embedding = self._embed_query(query)

            # Perform search
            results = self._perform_search(query_embedding, index, texts, top_k)

            # Log metrics
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.4f}s, returned {len(results)} results")

            # Log query metrics if enabled
            if performance_config.log_query_metrics:
                query_logger.log_query(
                    query=query,
                    latency=search_time,
                    hit_count=len(results),
                    metadata={
                        "search_type": search_type,
                        "top_k": top_k,
                        "fallback_used": fallback_used,
                    },
                )

            return results

        except Exception as e:
            search_time = time.time() - start_time
            error_msg = f"Search failed after {search_time:.4f}s: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg, include_traceback=True)
            raise

    def _validate_search_inputs(
        self,
        query: str,
        index: faiss.Index,
        texts: List[Any],
        top_k: int
    ) -> None:
        """Validate search input parameters"""
        # Validate query
        if not query or not isinstance(query, str):
            logger.error(f"Invalid query: {query}")
            raise ValueError(f"Query must be a non-empty string, got {type(query)}")

        # Validate top_k
        if not isinstance(top_k, int) or top_k <= 0:
            logger.error(f"Invalid top_k value: {top_k}")
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        # Validate index
        if index is None:
            raise ValueError("No index provided for search")

        # Validate texts
        if not texts:
            raise ValueError("No texts provided for search")

        if not isinstance(texts, list):
            logger.error(f"texts must be a list, got {type(texts)}")
            raise TypeError(f"texts must be a list, got {type(texts)}")

        if len(texts) == 0:
            logger.warning("Empty texts list, search will return no results")

    def _embed_query(self, query: str) -> np.ndarray:
        """
        Embed query string using the embedding service

        Args:
            query: Query string to embed

        Returns:
            Query embedding array

        Raises:
            ValueError: If embedding fails
        """
        try:
            query_embedding = embedding_service.embed_query(query)

            # Validate query embedding dimensions
            if query_embedding.shape[1] != self.embedding_dim:
                error_msg = f"Query embedding dimension mismatch: got {query_embedding.shape[1]}, expected {self.embedding_dim}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            return query_embedding

        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise ValueError(f"Failed to embed query: {str(e)}")

    def _perform_search(
        self,
        query_embedding: np.ndarray,
        index: faiss.Index,
        texts: List[Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Perform the actual FAISS search and format results

        Args:
            query_embedding: Embedded query vector
            index: FAISS index to search
            texts: List of text chunks
            top_k: Number of results to return

        Returns:
            List of formatted search results
        """
        # Adjust top_k to not exceed available chunks
        safe_top_k = min(top_k, len(texts))
        if safe_top_k < top_k:
            logger.info(f"Adjusted top_k from {top_k} to {safe_top_k} based on available chunks")

        # Perform FAISS search
        try:
            scores, indices = index.search(query_embedding, safe_top_k)
        except Exception as e:
            logger.error(f"Error during FAISS search: {str(e)}")
            raise ValueError(f"Search operation failed: {str(e)}")

        # Validate search results
        if indices.shape[0] == 0 or scores.shape[0] == 0:
            logger.warning("Search returned empty results")
            return []

        if indices.shape[1] != safe_top_k or scores.shape[1] != safe_top_k:
            logger.warning(
                f"Search returned unexpected dimensions: indices {indices.shape}, scores {scores.shape}, expected ({1}, {safe_top_k})"
            )

        # Format and return results
        return self._format_search_results(scores[0], indices[0], texts)

    def _format_search_results(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
        texts: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Format search results into standardized format

        Args:
            scores: Search scores from FAISS
            indices: Result indices from FAISS
            texts: Original text chunks

        Returns:
            List of formatted result dictionaries
        """
        results = []

        for i, idx in enumerate(indices):
            if 0 <= idx < len(texts):  # Ensure index is valid
                try:
                    # Handle both dict and string text formats
                    if isinstance(texts[idx], dict):
                        chunk = texts[idx].copy()
                    else:
                        chunk = {"text": texts[idx]}

                    # Add search score
                    chunk["score"] = float(scores[i])  # Convert numpy float to Python float

                    # Ensure metadata exists
                    if "metadata" not in chunk:
                        chunk["metadata"] = {"source": "unknown"}

                    results.append(chunk)

                except (IndexError, KeyError, AttributeError) as e:
                    logger.warning(f"Error processing search result at index {idx}: {str(e)}")
                    # Create a minimal valid result
                    results.append({
                        "text": f"Error retrieving chunk {idx}",
                        "score": float(scores[i]),
                        "metadata": {"source": "unknown", "error": str(e)},
                    })
            else:
                logger.warning(
                    f"Invalid index {idx} returned by search, valid range is 0-{len(texts)-1}"
                )

        return results

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            "embedding_dim": self.embedding_dim,
            "embedding_service_metrics": embedding_service.get_metrics(),
        }

    def validate_index_compatibility(self, index: faiss.Index) -> bool:
        """
        Validate that index is compatible with search engine

        Args:
            index: FAISS index to validate

        Returns:
            True if compatible, False otherwise
        """
        if index is None:
            logger.warning("Index is None, not compatible")
            return False

        if hasattr(index, 'd'):
            index_dim = index.d
            if index_dim != self.embedding_dim:
                logger.warning(
                    f"Index dimension ({index_dim}) doesn't match search engine dimension ({self.embedding_dim})"
                )
                return False

        if hasattr(index, 'ntotal'):
            if index.ntotal == 0:
                logger.warning("Index is empty (ntotal=0)")
                return False

        logger.debug("Index compatibility validated successfully")
        return True
