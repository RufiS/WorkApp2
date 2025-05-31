# Semantic grouping processor for grouping chunks by semantic similarity
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logger = logging.getLogger(__name__)


class SemanticGroupingProcessor:
    """Processor for grouping chunks by semantic similarity"""

    def __init__(self, embedder=None, debug_mode: bool = False):
        """
        Initialize the semantic grouping processor

        Args:
            embedder: Sentence embedder model (optional)
            debug_mode: Whether to enable debug mode for detailed logging
        """
        self.debug_mode = debug_mode
        self.embedder = embedder
        self.distance_threshold = 0.25  # Threshold for hierarchical clustering

    def process(self, query: str, chunks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Process chunks to group them by semantic similarity

        Args:
            query: The user's question
            chunks: List of chunks with text and scores

        Returns:
            List of semantic groups (each group is a list of chunks)
        """
        if not chunks:
            logger.warning("No chunks provided for semantic grouping")
            return []

        # If we have very few chunks, don't group
        if len(chunks) < 3:
            logger.info(f"Only {len(chunks)} chunks, skipping semantic grouping")
            return [chunks]

        try:
            # If we have an embedder, use it for better similarity
            if self.embedder:
                return self._group_with_embedder(query, chunks)
            else:
                # Fall back to simple grouping by consecutive IDs
                return self._group_by_consecutive_ids(chunks)
        except Exception as e:
            logger.error(f"Error in semantic grouping: {str(e)}", exc_info=True)
            # Fall back to simple grouping
            return self._group_by_consecutive_ids(chunks)

    def _group_with_embedder(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Group chunks using embedder for semantic similarity

        Args:
            query: The user's question
            chunks: List of chunks with text and scores

        Returns:
            List of semantic groups
        """
        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]

        # Encode texts
        embeddings = self.embedder.encode(texts)

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix

        # Apply hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            affinity="precomputed",
            linkage="average",
        )
        cluster_labels = clustering.fit_predict(distance_matrix)

        # Group chunks by cluster label
        groups = {}
        for i, label in enumerate(cluster_labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(chunks[i])

        # Convert to list of groups
        semantic_groups = list(groups.values())

        # Sort groups by average score (descending)
        semantic_groups.sort(
            key=lambda group: sum(chunk.get("score", 0) for chunk in group) / len(group),
            reverse=True,
        )

        if self.debug_mode:
            logger.debug(
                f"Created {len(semantic_groups)} semantic groups from {len(chunks)} chunks"
            )
            for i, group in enumerate(semantic_groups[:3]):
                logger.debug(f"Group {i}: {len(group)} chunks")

        return semantic_groups

    def _group_by_consecutive_ids(self, chunks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group chunks by consecutive IDs

        Args:
            chunks: List of chunks with text and scores

        Returns:
            List of semantic groups
        """
        # Sort chunks by ID
        sorted_chunks = sorted(chunks, key=lambda x: x.get("id", 0))

        # Group consecutive chunks
        groups = []
        current_group = [sorted_chunks[0]]

        for i in range(1, len(sorted_chunks)):
            # If this chunk is consecutive with the previous one
            if sorted_chunks[i].get("id", 0) == sorted_chunks[i - 1].get("id", 0) + 1:
                current_group.append(sorted_chunks[i])
            else:
                # Start a new group
                groups.append(current_group)
                current_group = [sorted_chunks[i]]

        # Add the last group
        groups.append(current_group)

        if self.debug_mode:
            logger.debug(
                f"Created {len(groups)} groups by consecutive IDs from {len(chunks)} chunks"
            )

        return groups
