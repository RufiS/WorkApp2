# Topic clustering processor for organizing chunks by topic
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logger = logging.getLogger(__name__)

class TopicClusteringProcessor:
    """Processor for clustering chunks by topic"""
    
    def __init__(self, embedder=None, debug_mode: bool = False):
        """
        Initialize the topic clustering processor
        
        Args:
            embedder: Sentence embedder model (optional)
            debug_mode: Whether to enable debug mode for detailed logging
        """
        self.debug_mode = debug_mode
        self.embedder = embedder
        self.vectorizer = TfidfVectorizer(
            max_df=0.85,
            min_df=2,
            max_features=1000,
            stop_words='english'
        )
        self.eps = 0.3  # DBSCAN epsilon parameter
        self.min_samples = 2  # DBSCAN min_samples parameter
        
    def process(self, query: str, semantic_groups: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """
        Process semantic groups to cluster them by topic
        
        Args:
            query: The user's question
            semantic_groups: List of semantic groups from semantic grouping
            
        Returns:
            List of topic clusters
        """
        if not semantic_groups:
            logger.warning("No semantic groups provided for topic clustering")
            return []
            
        # Flatten semantic groups to get all chunks
        all_chunks = [chunk for group in semantic_groups for chunk in group]
        
        # If we have very few chunks, don't cluster
        if len(all_chunks) < 5:
            logger.info(f"Only {len(all_chunks)} chunks, skipping topic clustering")
            return semantic_groups
            
        # Extract text from chunks
        texts = [chunk["text"] for chunk in all_chunks]
        
        try:
            # Create TF-IDF matrix
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Convert similarity to distance (1 - similarity)
            distance_matrix = 1 - similarity_matrix
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed')
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Group chunks by cluster label
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(all_chunks[i])
                
            # Sort clusters by size (descending)
            sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
            
            # If we have a cluster with label -1 (noise), process it separately
            noise_cluster = None
            if -1 in clusters:
                noise_cluster = clusters[-1]
                # Remove the noise cluster from sorted_clusters
                for i, cluster in enumerate(sorted_clusters):
                    if cluster == noise_cluster:
                        sorted_clusters.pop(i)
                        break
                    
            # If we have noise points, try to assign them to the closest cluster
            if noise_cluster:
                self._assign_noise_points(noise_cluster, sorted_clusters, query)
                
            # Ensure we don't have empty clusters
            sorted_clusters = [cluster for cluster in sorted_clusters if cluster]
            
            if self.debug_mode:
                logger.debug(f"Created {len(sorted_clusters)} topic clusters from {len(semantic_groups)} semantic groups")
                for i, cluster in enumerate(sorted_clusters[:3]):
                    logger.debug(f"Cluster {i}: {len(cluster)} chunks, topic: {self.extract_topic(cluster)}")
                    
            return sorted_clusters
        except Exception as e:
            logger.error(f"Error in topic clustering: {str(e)}", exc_info=True)
            # Fall back to semantic groups if clustering fails
            return semantic_groups
            
    def _assign_noise_points(self, noise_cluster: List[Dict[str, Any]], 
                             clusters: List[List[Dict[str, Any]]], 
                             query: str) -> None:
        """
        Assign noise points to the closest cluster
        
        Args:
            noise_cluster: List of chunks labeled as noise
            clusters: List of clusters
            query: The user's question
        """
        if not noise_cluster or not clusters:
            return
            
        # If we have the embedder, use it for better similarity
        if self.embedder:
            try:
                # Encode query and all chunks
                query_embedding = self.embedder.encode([query])[0]
                
                # For each noise point
                for noise_chunk in noise_cluster:
                    # Find the best cluster based on similarity to query
                    best_cluster = None
                    best_similarity = -1
                    
                    for cluster in clusters:
                        # Calculate average similarity of cluster to query
                        cluster_texts = [chunk["text"] for chunk in cluster]
                        cluster_embeddings = self.embedder.encode(cluster_texts)
                        cluster_similarity = np.mean([np.dot(query_embedding, emb) for emb in cluster_embeddings])
                        
                        if cluster_similarity > best_similarity:
                            best_similarity = cluster_similarity
                            best_cluster = cluster
                            
                    # Assign to best cluster if found
                    if best_cluster is not None:
                        best_cluster.append(noise_chunk)
            except Exception as e:
                logger.error(f"Error assigning noise points with embedder: {str(e)}", exc_info=True)
                # Fall back to simple assignment
                self._simple_assign_noise(noise_cluster, clusters)
        else:
            # Simple assignment without embedder
            self._simple_assign_noise(noise_cluster, clusters)
            
    def _simple_assign_noise(self, noise_cluster: List[Dict[str, Any]], 
                            clusters: List[List[Dict[str, Any]]]) -> None:
        """
        Simple method to assign noise points to clusters
        
        Args:
            noise_cluster: List of chunks labeled as noise
            clusters: List of clusters
        """
        # If we have only one cluster, assign all noise to it
        if len(clusters) == 1:
            clusters[0].extend(noise_cluster)
            return
            
        # Otherwise, distribute noise points among clusters based on TF-IDF similarity
        try:
            # Extract text from noise and clusters
            noise_texts = [chunk["text"] for chunk in noise_cluster]
            cluster_texts = [[chunk["text"] for chunk in cluster] for cluster in clusters]
            
            # For each noise point
            for i, noise_text in enumerate(noise_texts):
                # Find the best cluster based on TF-IDF similarity
                best_cluster_idx = 0
                best_similarity = -1
                
                for j, texts in enumerate(cluster_texts):
                    # Create a mini corpus with the noise text and all cluster texts
                    mini_corpus = [noise_text] + texts
                    mini_vectorizer = TfidfVectorizer(stop_words='english')
                    mini_tfidf = mini_vectorizer.fit_transform(mini_corpus)
                    
                    # Calculate average similarity between noise and cluster
                    mini_similarity = cosine_similarity(mini_tfidf[0:1], mini_tfidf[1:])[0].mean()
                    
                    if mini_similarity > best_similarity:
                        best_similarity = mini_similarity
                        best_cluster_idx = j
                        
                # Assign to best cluster
                clusters[best_cluster_idx].append(noise_cluster[i])
        except Exception as e:
            logger.error(f"Error in simple noise assignment: {str(e)}", exc_info=True)
            # If all else fails, just add noise points to the largest cluster
            largest_cluster = max(clusters, key=len)
            largest_cluster.extend(noise_cluster)
            
    def extract_topic(self, cluster: List[Dict[str, Any]]) -> str:
        """
        Extract a representative topic for a cluster
        
        Args:
            cluster: List of chunks in the cluster
            
        Returns:
            A string representing the topic
        """
        if not cluster:
            return ""
            
        try:
            # Extract text from chunks
            texts = [chunk["text"] for chunk in cluster]
            
            # Create TF-IDF matrix
            mini_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = mini_vectorizer.fit_transform(texts)
            
            # Get feature names
            feature_names = mini_vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores for each term across all documents
            tfidf_sums = tfidf_matrix.sum(axis=0).A1
            
            # Get top terms
            top_indices = tfidf_sums.argsort()[-5:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            # Join terms to create topic
            topic = ", ".join(top_terms)
            
            return topic
        except Exception as e:
            logger.error(f"Error extracting topic: {str(e)}", exc_info=True)
            # Fall back to first few words of first chunk
            return cluster[0]["text"][:50] + "..." if cluster else ""