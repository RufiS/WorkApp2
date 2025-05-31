# Enhanced context processor for improving context quality
import logging
from typing import List, Dict, Any, Optional, Tuple

# Try to import modules, with fallback implementations if not available
try:
    from utils.text_processing.context_enhancement.semantic_grouping import (
        SemanticGroupingProcessor,
    )
    from utils.text_processing.context_enhancement.topic_clustering import TopicClusteringProcessor
    from utils.text_processing.context_enhancement.context_deduplication import (
        ContextDeduplicationProcessor,
    )
    from utils.text_processing.context_enhancement.context_enrichment import (
        ContextEnrichmentProcessor,
    )
    from utils.text_processing.context_processing import clean_context
except ImportError:
    # Define fallback classes if modules are not available
    logger = logging.getLogger(__name__)
    logger.warning("Context enhancement modules not found, using fallback implementations")

    class SemanticGroupingProcessor:
        def __init__(self, embedder=None, debug_mode=False):
            self.debug_mode = debug_mode
            self.embedder = embedder

        def process(self, query, chunks):
            # Simple fallback implementation that doesn't do any grouping
            return [chunks]

    class TopicClusteringProcessor:
        def __init__(self, embedder=None, debug_mode=False):
            self.debug_mode = debug_mode
            self.embedder = embedder

        def process(self, query, semantic_groups):
            # Simple fallback implementation that doesn't do any clustering
            return semantic_groups

        def extract_topic(self, cluster):
            return "Unknown Topic"

    class ContextDeduplicationProcessor:
        def __init__(self, debug_mode=False):
            self.debug_mode = debug_mode

        def process(self, chunks_with_scores):
            # Simple fallback implementation that doesn't deduplicate
            return chunks_with_scores

    class ContextEnrichmentProcessor:
        def __init__(self, debug_mode=False):
            self.debug_mode = debug_mode
            self.added_elements = 0

        def process(self, query, topic_clusters):
            # Simple fallback implementation that just flattens the clusters
            all_chunks = []
            for cluster in topic_clusters:
                all_chunks.extend(cluster)
            return "\n\n".join([chunk["text"] for chunk in all_chunks]), all_chunks

        def get_added_elements(self):
            return self.added_elements

    def clean_context(text):
        return text


# Setup logging
logger = logging.getLogger(__name__)


class EnhancedContextProcessor:
    """Enhanced context processor for improving context quality"""

    def __init__(self, embedder=None, debug_mode: bool = False):
        """
        Initialize the enhanced context processor

        Args:
            embedder: Sentence embedder model (will be passed to processors that need it)
            debug_mode: Whether to enable debug mode for detailed logging
        """
        self.debug_mode = debug_mode
        self.embedder = embedder

        # Initialize processors
        self.topic_clustering = TopicClusteringProcessor(embedder=embedder, debug_mode=debug_mode)
        self.semantic_grouping = SemanticGroupingProcessor(embedder=embedder, debug_mode=debug_mode)
        self.deduplication = ContextDeduplicationProcessor(debug_mode=debug_mode)
        self.enrichment = ContextEnrichmentProcessor(debug_mode=debug_mode)

        logger.info("Enhanced context processor initialized")

    def process(
        self, query: str, chunks_with_scores: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process chunks to improve context quality

        Args:
            query: The user's question
            chunks_with_scores: List of chunks with scores from retrieval

        Returns:
            Tuple of (processed_context, processed_chunks_with_scores)
        """
        try:
            if not chunks_with_scores:
                logger.warning("No chunks provided for context processing")
                return "", []

            # Log initial chunks if in debug mode
            if self.debug_mode:
                logger.debug(f"Initial chunks: {len(chunks_with_scores)} chunks")
                for i, chunk in enumerate(chunks_with_scores[:3]):
                    logger.debug(f"Chunk {i}: {chunk['text'][:100]}... (score: {chunk['score']})")

            # Step 1: Deduplicate content to remove redundant information
            deduplicated_chunks = self.deduplication.process(chunks_with_scores)
            if self.debug_mode:
                logger.debug(f"After deduplication: {len(deduplicated_chunks)} chunks")

            # Step 2: Group chunks by semantic similarity to identify related information
            semantic_groups = self.semantic_grouping.process(query, deduplicated_chunks)
            if self.debug_mode:
                logger.debug(f"Semantic groups: {len(semantic_groups)} groups")

            # Step 3: Cluster chunks by topic to organize information
            topic_clusters = self.topic_clustering.process(query, semantic_groups)
            if self.debug_mode:
                logger.debug(f"Topic clusters: {len(topic_clusters)} clusters")

            # Step 4: Enrich context with cross-references and structure
            enriched_context, enriched_chunks = self.enrichment.process(query, topic_clusters)
            if self.debug_mode:
                logger.debug(f"Enriched context length: {len(enriched_context)} characters")

            # Clean the context
            clean_enhanced_context = clean_context(enriched_context)

            return clean_enhanced_context, enriched_chunks
        except Exception as e:
            logger.error(f"Error in enhanced context processing: {str(e)}", exc_info=True)
            # Fall back to simple context
            simple_context = "\n\n".join([chunk["text"] for chunk in chunks_with_scores])
            return clean_context(simple_context), chunks_with_scores

    def process_with_metadata(
        self, query: str, chunks_with_scores: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process chunks and return detailed metadata about the processing

        Args:
            query: The user's question
            chunks_with_scores: List of chunks with scores from retrieval

        Returns:
            Dictionary with processed context, chunks, and metadata
        """
        try:
            if not chunks_with_scores:
                logger.warning("No chunks provided for context processing")
                return {
                    "context": "",
                    "chunks": [],
                    "metadata": {
                        "original_chunk_count": 0,
                        "processed_chunk_count": 0,
                        "deduplication": {"removed_chunks": 0},
                        "semantic_groups": {"group_count": 0},
                        "topic_clusters": {"cluster_count": 0},
                        "processing_steps": [],
                    },
                }

            # Initialize metadata
            metadata = {"original_chunk_count": len(chunks_with_scores), "processing_steps": []}

            # Step 1: Deduplicate content
            deduplicated_chunks = self.deduplication.process(chunks_with_scores)
            metadata["deduplication"] = {
                "removed_chunks": len(chunks_with_scores) - len(deduplicated_chunks),
                "remaining_chunks": len(deduplicated_chunks),
            }
            metadata["processing_steps"].append("deduplication")

            # Step 2: Group chunks by semantic similarity
            semantic_groups = self.semantic_grouping.process(query, deduplicated_chunks)
            metadata["semantic_groups"] = {
                "group_count": len(semantic_groups),
                "groups": [
                    {
                        "size": len(group),
                        "avg_score": (
                            sum(chunk["score"] for chunk in group) / len(group) if group else 0
                        ),
                    }
                    for group in semantic_groups
                ],
            }
            metadata["processing_steps"].append("semantic_grouping")

            # Step 3: Cluster chunks by topic
            topic_clusters = self.topic_clustering.process(query, semantic_groups)
            metadata["topic_clusters"] = {
                "cluster_count": len(topic_clusters),
                "clusters": [
                    {
                        "size": len(cluster),
                        "topic": self.topic_clustering.extract_topic(cluster) if cluster else "",
                    }
                    for cluster in topic_clusters
                ],
            }
            metadata["processing_steps"].append("topic_clustering")

            # Step 4: Enrich context
            enriched_context, enriched_chunks = self.enrichment.process(query, topic_clusters)
            metadata["enrichment"] = {
                "context_length": len(enriched_context),
                "added_elements": self.enrichment.get_added_elements(),
            }
            metadata["processing_steps"].append("context_enrichment")
            metadata["processed_chunk_count"] = len(enriched_chunks)

            # Clean the context
            clean_enhanced_context = clean_context(enriched_context)

            return {
                "context": clean_enhanced_context,
                "chunks": enriched_chunks,
                "metadata": metadata,
            }
        except Exception as e:
            logger.error(f"Error in enhanced context processing: {str(e)}", exc_info=True)
            # Fall back to simple context
            simple_context = "\n\n".join([chunk["text"] for chunk in chunks_with_scores])
            return {
                "context": clean_context(simple_context),
                "chunks": chunks_with_scores,
                "metadata": {
                    "error": str(e),
                    "fallback": True,
                    "chunk_count": len(chunks_with_scores),
                },
            }


# Standalone function for processing context
def process_context(
    query: str, chunks: List[Dict[str, Any]], embedder=None, debug_mode: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Process chunks to enhance context quality

    Args:
        query: The user's question
        chunks: List of chunks with text and scores
        embedder: Optional embedder model to use
        debug_mode: Whether to enable debug mode

    Returns:
        Tuple of (enhanced_context, metadata)
    """
    processor = EnhancedContextProcessor(embedder=embedder, debug_mode=debug_mode)
    result = processor.process_with_metadata(query, chunks)
    return result["context"], result["metadata"]
