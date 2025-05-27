# Context enrichment processor for enhancing context with structure and cross-references
import logging
from typing import List, Dict, Any, Optional, Tuple
import re

# Setup logging
logger = logging.getLogger(__name__)

class ContextEnrichmentProcessor:
    """Processor for enriching context with structure and cross-references"""
    
    def __init__(self, debug_mode: bool = False):
        """
        Initialize the context enrichment processor
        
        Args:
            debug_mode: Whether to enable debug mode for detailed logging
        """
        self.debug_mode = debug_mode
        self.added_elements = {
            "section_headers": 0,
            "cross_references": 0,
            "topic_markers": 0
        }
        
    def process(self, query: str, topic_clusters: List[List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process topic clusters to enrich context with structure and cross-references
        
        Args:
            query: The user's question
            topic_clusters: List of topic clusters from topic clustering
            
        Returns:
            Tuple of (enriched_context, enriched_chunks)
        """
        if not topic_clusters:
            logger.warning("No topic clusters provided for context enrichment")
            return "", []
            
        # Reset added elements counter
        self.added_elements = {
            "section_headers": 0,
            "cross_references": 0,
            "topic_markers": 0
        }
        
        # Flatten clusters to get all chunks
        all_chunks = [chunk for cluster in topic_clusters for chunk in cluster]
        
        # If we have only one cluster, process it directly
        if len(topic_clusters) == 1:
            enriched_text = self._process_single_cluster(topic_clusters[0])
            return enriched_text, all_chunks
            
        # Process multiple clusters
        sections = []
        cross_references = {}
        
        # Process each cluster
        for i, cluster in enumerate(topic_clusters):
            # Extract topic for the cluster
            topic = self._extract_topic(cluster)
            
            # Create section header
            section_header = f"SECTION {i+1}: {topic.upper()}"
            self.added_elements["section_headers"] += 1
            
            # Process cluster content
            cluster_text = self._process_cluster_content(cluster)
            
            # Add section to sections
            sections.append(f"{section_header}\n\n{cluster_text}")
            
            # Find potential cross-references to other clusters
            cross_refs = self._find_cross_references(cluster, topic_clusters, i)
            if cross_refs:
                cross_references[i] = cross_refs
                
        # Add cross-references to sections
        for i, refs in cross_references.items():
            if refs:
                ref_text = "\n\nRELATED INFORMATION:\n" + ", ".join([f"Section {ref+1}" for ref in refs])
                sections[i] += ref_text
                self.added_elements["cross_references"] += 1
                
        # Join sections with separators
        enriched_text = "\n\n" + "\n\n---\n\n".join(sections) + "\n\n"
        
        # Add a table of contents if we have multiple sections
        if len(sections) > 1:
            toc = "TABLE OF CONTENTS:\n"
            for i, cluster in enumerate(topic_clusters):
                topic = self._extract_topic(cluster)
                toc += f"Section {i+1}: {topic}\n"
            enriched_text = toc + "\n\n" + enriched_text
            
        return enriched_text, all_chunks
        
    def _process_single_cluster(self, cluster: List[Dict[str, Any]]) -> str:
        """
        Process a single cluster
        
        Args:
            cluster: List of chunks in the cluster
            
        Returns:
            Processed text
        """
        # Extract topic for the cluster
        topic = self._extract_topic(cluster)
        
        # Create section header
        section_header = f"INFORMATION: {topic.upper()}"
        self.added_elements["section_headers"] += 1
        
        # Process cluster content
        cluster_text = self._process_cluster_content(cluster)
        
        # Return enriched text
        return f"{section_header}\n\n{cluster_text}"
        
    def _process_cluster_content(self, cluster: List[Dict[str, Any]]) -> str:
        """
        Process the content of a cluster
        
        Args:
            cluster: List of chunks in the cluster
            
        Returns:
            Processed text
        """
        # Sort chunks by ID to maintain document order
        sorted_chunks = sorted(cluster, key=lambda x: x["id"])
        
        # Group consecutive chunks
        groups = []
        current_group = [sorted_chunks[0]]
        
        for i in range(1, len(sorted_chunks)):
            # If this chunk is consecutive with the previous one
            if sorted_chunks[i]["id"] == sorted_chunks[i-1]["id"] + 1:
                current_group.append(sorted_chunks[i])
            else:
                # Start a new group
                groups.append(current_group)
                current_group = [sorted_chunks[i]]
                
        # Add the last group
        groups.append(current_group)
        
        # Process each group
        processed_groups = []
        for group in groups:
            # Join texts in the group
            group_text = " ".join([chunk["text"] for chunk in group])
            
            # Clean up the text
            group_text = self._clean_text(group_text)
            
            processed_groups.append(group_text)
            
        # Join groups with paragraph breaks
        return "\n\n".join(processed_groups)
        
    def _extract_topic(self, cluster: List[Dict[str, Any]]) -> str:
        """
        Extract a topic for a cluster
        
        Args:
            cluster: List of chunks in the cluster
            
        Returns:
            Topic string
        """
        if not cluster:
            return "General Information"
            
        # Try to extract a meaningful topic from the first chunk
        first_chunk = cluster[0]["text"]
        
        # Look for a title or heading
        title_match = re.search(r'^([\w\s\-]+)(?:\n|:)', first_chunk)
        if title_match:
            return title_match.group(1).strip()
            
        # If no title found, use the first sentence
        sentence_match = re.search(r'^([^.!?]+[.!?])', first_chunk)
        if sentence_match:
            return sentence_match.group(1).strip()
            
        # If all else fails, use the first 50 characters
        return first_chunk[:50].strip() + "..."
        
    def _find_cross_references(self, cluster: List[Dict[str, Any]], 
                              all_clusters: List[List[Dict[str, Any]]], 
                              current_index: int) -> List[int]:
        """
        Find potential cross-references to other clusters
        
        Args:
            cluster: Current cluster
            all_clusters: All topic clusters
            current_index: Index of the current cluster
            
        Returns:
            List of indices of related clusters
        """
        cross_refs = []
        
        # Extract key terms from the cluster
        cluster_text = " ".join([chunk["text"] for chunk in cluster])
        key_terms = self._extract_key_terms(cluster_text)
        
        # Check each other cluster for key terms
        for i, other_cluster in enumerate(all_clusters):
            if i == current_index:
                continue
                
            other_text = " ".join([chunk["text"] for chunk in other_cluster])
            
            # Count how many key terms appear in the other cluster
            term_count = sum(1 for term in key_terms if term.lower() in other_text.lower())
            
            # If enough key terms appear, consider it related
            if term_count >= 2:  # Require at least 2 key terms
                cross_refs.append(i)
                
        return cross_refs
        
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text
        
        Args:
            text: Text to extract terms from
            
        Returns:
            List of key terms
        """
        # Simple extraction of capitalized terms and phrases
        capitalized = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,5})\b', text)
        
        # Filter out common words and short terms
        common_words = {"The", "A", "An", "And", "Or", "But", "If", "Then", "So", "As", "In", "On", "At"}
        filtered = [term for term in capitalized if term not in common_words and len(term) > 2]
        
        # Limit to top 10 terms
        return filtered[:10]
        
    def _clean_text(self, text: str) -> str:
        """
        Clean text for presentation
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove duplicate whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix hyphenation at line breaks
        text = re.sub(r'(\w)-\s*(\w)', r'\1\2', text)
        
        # Remove table of contents
        text = re.sub(r"Table of Contents.*?(?:\n|$)", "", text, flags=re.I | re.DOTALL)
        
        return text.strip()
        
    def get_added_elements(self) -> Dict[str, int]:
        """
        Get count of added elements
        
        Returns:
            Dictionary with counts of added elements
        """
        return self.added_elements