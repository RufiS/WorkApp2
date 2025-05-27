# Context deduplication processor for removing redundant information
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from difflib import SequenceMatcher

# Setup logging
logger = logging.getLogger(__name__)

class ContextDeduplicationProcessor:
    """Processor for removing redundant information from chunks"""
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the context deduplication processor
        
        Args:
            debug_mode: Whether to enable debug mode for detailed logging
        """
        self.debug_mode = debug_mode
        self.similarity_threshold = 0.8  # Threshold for considering text duplicated
        self.min_duplicate_length = 50  # Minimum length of text to consider for duplication
        
    def process(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process chunks to remove redundant information
        
        Args:
            chunks: List of chunks with scores
            
        Returns:
            List of deduplicated chunks
        """
        if not chunks:
            logger.warning("No chunks provided for deduplication")
            return []
            
        # Sort chunks by relevance score (ascending, as lower distance is better)
        sorted_chunks = sorted(chunks, key=lambda x: x["score"])
        
        # Initialize result with the most relevant chunk
        result = [sorted_chunks[0]]
        
        # For each remaining chunk, check if it's redundant
        for chunk in sorted_chunks[1:]:
            is_redundant = False
            
            # Check against all chunks already in result
            for existing_chunk in result:
                # Calculate text similarity
                similarity = self._calculate_text_similarity(chunk["text"], existing_chunk["text"])
                
                # If similarity is above threshold, consider redundant
                if similarity >= self.similarity_threshold:
                    is_redundant = True
                    if self.debug_mode:
                        logger.debug(f"Chunk with score {chunk['score']} is redundant with similarity {similarity:.2f}")
                    break
                    
            # If not redundant, add to result
            if not is_redundant:
                result.append(chunk)
                
        if self.debug_mode:
            logger.debug(f"Deduplication removed {len(chunks) - len(result)} chunks")
            
        return result
        
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # If either text is too short, return 0
        if len(text1) < self.min_duplicate_length or len(text2) < self.min_duplicate_length:
            return 0.0
            
        # Normalize texts for comparison
        text1 = self._normalize_text(text1)
        text2 = self._normalize_text(text2)
        
        # Use SequenceMatcher for similarity
        matcher = SequenceMatcher(None, text1, text2)
        similarity = matcher.ratio()
        
        # Check for contained text (one text being a subset of the other)
        contained_similarity = self._check_contained_text(text1, text2)
        
        # Return the maximum similarity
        return max(similarity, contained_similarity)
        
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
        
    def _check_contained_text(self, text1: str, text2: str) -> float:
        """
        Check if one text is contained within the other
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Check if text1 is contained in text2
        if text1 in text2:
            return len(text1) / len(text2)
            
        # Check if text2 is contained in text1
        if text2 in text1:
            return len(text2) / len(text1)
            
        return 0.0