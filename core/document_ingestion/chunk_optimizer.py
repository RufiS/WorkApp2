"""Chunk optimization functionality for handling edge cases"""

import os
import logging
from typing import List, Dict, Any

from utils.common.error_handler import CommonErrorHandler
from utils.logging.error_logging import log_warning

logger = logging.getLogger(__name__)


class ChunkOptimizer:
    """Handles optimization of chunks, particularly small chunk merging"""
    
    def __init__(self, min_chunk_size: int = 50):
        """
        Initialize the chunk optimizer
        
        Args:
            min_chunk_size: Minimum acceptable chunk size in characters
        """
        self.min_chunk_size = min_chunk_size
        logger.info(f"Chunk optimizer initialized with min_size={min_chunk_size}")
    
    def handle_small_chunks(self, chunks: List[Dict[str, Any]], file_path: str = None) -> List[Dict[str, Any]]:
        """
        Handle abnormally small chunks by merging them with adjacent chunks or removing them
        
        Args:
            chunks: List of document chunks
            file_path: Path to the document file for logging purposes
            
        Returns:
            List of processed chunks with small chunks handled
        """
        if not chunks:
            return []
        
        # Early return if only one chunk and it's too small
        if len(chunks) == 1 and len(chunks[0].get("text", "")) < self.min_chunk_size:
            logger.warning(
                f"Document contains only one small chunk ({len(chunks[0].get('text', ''))} chars)"
            )
            return chunks  # Return as is, can't merge with anything
        
        try:
            optimized_chunks = self._merge_small_chunks(chunks, file_path)
            
            if len(optimized_chunks) != len(chunks):
                logger.info(
                    f"Chunk optimization complete: {len(chunks)} -> {len(optimized_chunks)} chunks"
                )
            
            return optimized_chunks
            
        except Exception as e:
            error_msg = CommonErrorHandler.handle_processing_error(
                "ChunkOptimizer", "small chunk handling", e
            )
            logger.warning(f"Chunk optimization failed, returning original chunks: {error_msg}")
            return chunks
    
    def _merge_small_chunks(self, chunks: List[Dict[str, Any]], file_path: str = None) -> List[Dict[str, Any]]:
        """
        Merge small chunks with adjacent chunks
        
        Args:
            chunks: List of document chunks
            file_path: Path to the document file for logging
            
        Returns:
            List of optimized chunks
        """
        result = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            current_text = current.get("text", "")
            
            # If current chunk is too small
            if len(current_text) < self.min_chunk_size:
                file_name = os.path.basename(file_path) if file_path else "unknown"
                logger.warning(
                    f"Small chunk detected in {file_name}, chunk {i}: {len(current_text)} chars"
                )
                log_warning(
                    f"Small chunk detected in {file_name}, chunk {i}: {len(current_text)} chars"
                )
                
                # Try to merge with next chunk if available
                if i + 1 < len(chunks):
                    merged_chunk = self._merge_with_next(current, chunks[i + 1], i)
                    result.append(merged_chunk)
                    i += 2  # Skip the next chunk since we merged it
                    logger.debug(f"Merged small chunk {i-1} with chunk {i}")
                else:
                    # If this is the last chunk, try to merge with previous
                    if result:
                        merged_chunk = self._merge_with_previous(result.pop(), current, i)
                        result.append(merged_chunk)
                        logger.debug(f"Merged small chunk {i} with previous chunk")
                    else:
                        # If no previous chunks, just add it despite being small
                        result.append(current)
                    i += 1
            else:
                # Current chunk is large enough, add it as is
                result.append(current)
                i += 1
        
        return result
    
    def _merge_with_next(self, current: Dict[str, Any], next_chunk: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Merge current chunk with the next chunk
        
        Args:
            current: Current chunk
            next_chunk: Next chunk to merge with
            index: Current chunk index
            
        Returns:
            Merged chunk
        """
        current_text = current.get("text", "")
        next_text = next_chunk.get("text", "")
        
        # Create merged chunk based on current chunk
        merged = current.copy()
        merged["text"] = current_text + " " + next_text
        merged["merged"] = True
        merged["original_indices"] = [index, index + 1]
        
        # Update metadata to reflect merge
        if "metadata" in merged:
            merged["metadata"]["merged_chunks"] = 2
            merged["metadata"]["chunk_size"] = len(merged["text"])
        
        return merged
    
    def _merge_with_previous(self, previous: Dict[str, Any], current: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Merge current chunk with the previous chunk
        
        Args:
            previous: Previous chunk
            current: Current chunk to merge
            index: Current chunk index
            
        Returns:
            Merged chunk
        """
        prev_text = previous.get("text", "")
        current_text = current.get("text", "")
        
        # Create merged chunk based on previous chunk
        merged = previous.copy()
        merged["text"] = prev_text + " " + current_text
        merged["merged"] = True
        
        # Handle original indices
        if "original_indices" in previous:
            merged["original_indices"] = previous["original_indices"] + [index]
        else:
            merged["original_indices"] = [index - 1, index]
        
        # Update metadata to reflect merge
        if "metadata" in merged:
            chunk_count = merged["metadata"].get("merged_chunks", 1) + 1
            merged["metadata"]["merged_chunks"] = chunk_count
            merged["metadata"]["chunk_size"] = len(merged["text"])
        
        return merged
    
    def analyze_chunk_distribution(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the distribution of chunk sizes
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with chunk size statistics
        """
        if not chunks:
            return {"error": "No chunks to analyze"}
        
        try:
            sizes = [len(chunk.get("text", "")) for chunk in chunks]
            
            stats = {
                "total_chunks": len(chunks),
                "min_size": min(sizes),
                "max_size": max(sizes),
                "avg_size": sum(sizes) / len(sizes),
                "median_size": sorted(sizes)[len(sizes) // 2],
                "small_chunks": len([s for s in sizes if s < self.min_chunk_size]),
                "normal_chunks": len([s for s in sizes if self.min_chunk_size <= s <= 2000]),
                "large_chunks": len([s for s in sizes if s > 2000]),
            }
            
            # Calculate percentages
            if stats["total_chunks"] > 0:
                stats["small_chunks_pct"] = (stats["small_chunks"] / stats["total_chunks"]) * 100
                stats["normal_chunks_pct"] = (stats["normal_chunks"] / stats["total_chunks"]) * 100
                stats["large_chunks_pct"] = (stats["large_chunks"] / stats["total_chunks"]) * 100
            
            return stats
            
        except Exception as e:
            error_msg = CommonErrorHandler.handle_processing_error(
                "ChunkOptimizer", "chunk distribution analysis", e
            )
            return {"error": error_msg}
    
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
        """
        Validate chunk structure and content
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            for i, chunk in enumerate(chunks):
                # Check basic structure
                if not isinstance(chunk, dict):
                    issues.append(f"Chunk {i}: Not a dictionary")
                    continue
                
                # Check required fields
                if "text" not in chunk:
                    issues.append(f"Chunk {i}: Missing 'text' field")
                elif not isinstance(chunk["text"], str):
                    issues.append(f"Chunk {i}: 'text' field is not a string")
                elif not chunk["text"].strip():
                    issues.append(f"Chunk {i}: Empty or whitespace-only text")
                
                # Check metadata
                if "metadata" in chunk and not isinstance(chunk["metadata"], dict):
                    issues.append(f"Chunk {i}: 'metadata' field is not a dictionary")
                
                # Check chunk size
                text_length = len(chunk.get("text", ""))
                if text_length < self.min_chunk_size:
                    issues.append(f"Chunk {i}: Size {text_length} below minimum {self.min_chunk_size}")
            
            is_valid = len(issues) == 0
            return is_valid, issues
            
        except Exception as e:
            error_msg = CommonErrorHandler.handle_processing_error(
                "ChunkOptimizer", "chunk validation", e
            )
            return False, [f"Validation error: {error_msg}"]
    
    def optimize_chunk_boundaries(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize chunk boundaries to avoid splitting sentences
        
        Args:
            chunks: List of chunks to optimize
            
        Returns:
            List of optimized chunks
        """
        if not chunks:
            return chunks
        
        try:
            optimized = []
            
            for chunk in chunks:
                text = chunk.get("text", "")
                if not text:
                    continue
                
                # Check if chunk ends mid-sentence
                if not text.rstrip().endswith(('.', '!', '?', '\n')):
                    # Try to find a better break point
                    better_text = self._find_better_boundary(text)
                    if better_text != text:
                        updated_chunk = chunk.copy()
                        updated_chunk["text"] = better_text
                        updated_chunk["optimized_boundary"] = True
                        optimized.append(updated_chunk)
                    else:
                        optimized.append(chunk)
                else:
                    optimized.append(chunk)
            
            return optimized
            
        except Exception as e:
            CommonErrorHandler.handle_processing_error(
                "ChunkOptimizer", "boundary optimization", e
            )
            return chunks
    
    def _find_better_boundary(self, text: str) -> str:
        """
        Find a better boundary for chunk ending
        
        Args:
            text: Text to find boundary for
            
        Returns:
            Text with better boundary
        """
        # Look for sentence endings near the end of the text
        candidates = ['.', '!', '?', '\n']
        
        # Search backwards from the end for sentence endings
        for i in range(len(text) - 1, max(0, len(text) - 100), -1):
            if text[i] in candidates:
                # Found a sentence ending, use it as boundary
                return text[:i + 1].rstrip()
        
        # If no good boundary found, return original text
        return text
