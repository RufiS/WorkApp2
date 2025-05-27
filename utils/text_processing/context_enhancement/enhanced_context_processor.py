# Enhanced context processor for unified retrieval
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    from utils.text_processing.context_enhancement.semantic_grouping import SemanticGroupingProcessor
    from utils.text_processing.context_enhancement.topic_clustering import TopicClusteringProcessor
    from utils.text_processing.context_enhancement.context_enrichment import ContextEnrichmentProcessor
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
    """Processor for enhancing context with semantic grouping, topic clustering, and enrichment"""
    
    def __init__(self, embedder=None, debug_mode: bool = False):
        """
        Initialize the enhanced context processor
        
        Args:
            embedder: Sentence embedder model (optional)
            debug_mode: Whether to enable debug mode for detailed logging
        """
        self.debug_mode = debug_mode
        self.embedder = embedder
        
        # Initialize sub-processors
        self.semantic_grouping = SemanticGroupingProcessor(embedder=embedder, debug_mode=debug_mode)
        self.topic_clustering = TopicClusteringProcessor(embedder=embedder, debug_mode=debug_mode)
        self.context_enrichment = ContextEnrichmentProcessor(debug_mode=debug_mode)
        
    def process(self, query: str, chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process chunks to enhance context
        
        Args:
            query: The user's question
            chunks: List of chunks with text and scores
            
        Returns:
            Tuple of (enhanced_context, processed_chunks)
        """
        try:
            # Apply semantic grouping
            semantic_groups = self.semantic_grouping.process(query, chunks)
            
            # Apply topic clustering
            topic_clusters = self.topic_clustering.process(query, semantic_groups)
            
            # Apply context enrichment
            enhanced_context, processed_chunks = self.context_enrichment.process(query, topic_clusters)
            
            # Clean the context
            clean_enhanced_context = clean_context(enhanced_context)
            
            if self.debug_mode:
                logger.debug(f"Enhanced context processing complete. "
                           f"Semantic groups: {len(semantic_groups)}, "
                           f"Topic clusters: {len(topic_clusters)}, "
                           f"Added elements: {self.context_enrichment.added_elements}")
                
            return clean_enhanced_context, processed_chunks
        except Exception as e:
            logger.error(f"Error in enhanced context processing: {str(e)}", exc_info=True)
            # Fall back to simple context
            simple_context = "\n\n".join([chunk["text"] for chunk in chunks])
            return clean_context(simple_context), chunks
            
    def process_with_metadata(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process chunks to enhance context and return metadata
        
        Args:
            query: The user's question
            chunks: List of chunks with text and scores
            
        Returns:
            Dictionary with enhanced context, processed chunks, and metadata
        """
        try:
            # Apply semantic grouping
            semantic_groups = self.semantic_grouping.process(query, chunks)
            
            # Apply topic clustering
            topic_clusters = self.topic_clustering.process(query, semantic_groups)
            
            # Apply context enrichment
            enhanced_context, processed_chunks = self.context_enrichment.process(query, topic_clusters)
            
            # Clean the context
            clean_enhanced_context = clean_context(enhanced_context)
            
            # Collect metadata
            metadata = {
                "semantic_groups": len(semantic_groups),
                "topic_clusters": len(topic_clusters),
                "added_elements": self.context_enrichment.added_elements,
                "chunk_count": len(chunks),
                "processed_chunk_count": len(processed_chunks)
            }
            
            if self.debug_mode:
                logger.debug(f"Enhanced context processing complete. "
                           f"Semantic groups: {len(semantic_groups)}, "
                           f"Topic clusters: {len(topic_clusters)}, "
                           f"Added elements: {self.context_enrichment.added_elements}")
                
            return {
                "context": clean_enhanced_context,
                "chunks": processed_chunks,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error in enhanced context processing: {str(e)}", exc_info=True)
            # Fall back to simple context
            simple_context = "\n\n".join([chunk["text"] for chunk in chunks])
            return {
                "context": clean_context(simple_context),
                "chunks": chunks,
                "metadata": {
                    "error": str(e),
                    "fallback": True,
                    "chunk_count": len(chunks)
                }
            }

# Standalone function for processing context
def process_context(query: str, chunks: List[Dict[str, Any]], embedder=None, debug_mode: bool = False) -> Tuple[str, Dict[str, Any]]:
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