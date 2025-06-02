"""Semantic deduplication functionality for document chunks"""

import time
import logging
from typing import List, Dict, Any, Set
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from utils.common.error_handler import CommonErrorHandler
from utils.logging.error_logging import log_error
from utils.error_handling.enhanced_decorators import with_timing

logger = logging.getLogger(__name__)


class DeduplicationEngine:
    """Handles semantic deduplication of document chunks"""

    def __init__(self, embedding_model: SentenceTransformer):
        """
        Initialize the deduplication engine

        Args:
            embedding_model: Pre-initialized sentence transformer model
        """
        self.embedding_model = embedding_model
        logger.info("Deduplication engine initialized")

    @with_timing(threshold=1.0)
    def semantic_deduplication(self, chunks: List[Dict[str, Any]],
                             similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
        """
        Perform semantic deduplication on chunks using optimized approximate nearest neighbors
        with efficient memory usage and improved processing speed.

        Args:
            chunks: List of document chunks with metadata
            similarity_threshold: Threshold for considering chunks as duplicates (0.0 to 1.0)

        Returns:
            List of deduplicated chunks
        """
        if not chunks:
            return []

        # Early return if only one chunk
        if len(chunks) == 1:
            return chunks

        # Extract text from chunks for embedding
        texts, valid_chunk_indices = self._extract_valid_texts(chunks)

        # Early return if no valid chunks
        if not texts:
            logger.warning("No valid text content found in chunks for deduplication")
            return chunks

        # Track processing time
        start_time = time.time()

        try:
            # Create embeddings in batches
            embeddings_np = self._create_embeddings_in_batches(texts)

            if embeddings_np is None or embeddings_np.shape[0] == 0:
                logger.error("No embeddings were generated for deduplication")
                return chunks

            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings_np)

            # Set of indices to keep (start with all valid chunks)
            indices_to_keep = set(valid_chunk_indices)

            # Choose deduplication strategy based on dataset size
            chunk_count = len(texts)
            self._perform_deduplication_by_size(
                embeddings_np, valid_chunk_indices, chunks,
                indices_to_keep, similarity_threshold, chunk_count
            )

            # Return deduplicated chunks
            deduplicated_chunks = [chunks[i] for i in sorted(indices_to_keep)]
            dedup_time = time.time() - start_time

            logger.info(
                f"Semantic deduplication completed in {dedup_time:.2f}s: "
                f"{len(chunks)} chunks -> {len(deduplicated_chunks)} chunks"
            )

            # Log detailed metrics
            removed_count = len(chunks) - len(deduplicated_chunks)
            removal_percentage = (removed_count / len(chunks)) * 100 if len(chunks) > 0 else 0
            logger.info(f"Deduplication removed {removed_count} chunks ({removal_percentage:.1f}%)")

            return deduplicated_chunks

        except Exception as e:
            logger.error(f"Error during semantic deduplication: {str(e)}")
            log_error(f"Error during semantic deduplication: {str(e)}", include_traceback=True)
            # Return original chunks if deduplication fails
            return chunks

        finally:
            # Log deduplication metrics
            dedup_time = time.time() - start_time
            logger.debug(f"Semantic deduplication processing time: {dedup_time:.2f}s")

    def _extract_valid_texts(self, chunks: List[Dict[str, Any]]) -> tuple[List[str], List[int]]:
        """Extract valid texts from chunks and track their indices"""
        texts = []
        valid_chunk_indices = []

        # First pass: identify valid chunks and their texts
        for i, chunk in enumerate(chunks):
            if "content" in chunk and chunk["content"]:
                texts.append(chunk["content"])
                valid_chunk_indices.append(i)
            elif "text" in chunk and chunk["text"]:
                texts.append(chunk["text"])
                valid_chunk_indices.append(i)
            else:
                # Skip invalid chunks instead of using empty strings
                logger.warning(
                    f"Chunk without text content found during deduplication: {chunk.get('id', 'unknown')}"
                )

        return texts, valid_chunk_indices

    def _create_embeddings_in_batches(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for texts in batches"""
        chunk_count = len(texts)

        # Dynamic batch sizing based on chunk count and available memory
        if chunk_count > 50000:
            batch_size = 32  # Very small batches for extremely large datasets
        elif chunk_count > 20000:
            batch_size = 64
        elif chunk_count > 10000:
            batch_size = 128
        elif chunk_count > 5000:
            batch_size = 256
        else:
            batch_size = 512

        logger.info(f"Deduplicating {chunk_count} chunks with batch size {batch_size}")

        # Create embeddings in batches with progress tracking
        embeddings_list = []
        total_batches = (chunk_count + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, chunk_count)
            batch_texts = texts[start_idx:end_idx]

            try:
                # Time the embedding process
                batch_start_time = time.time()
                batch_embeddings = self.embedding_model.encode(batch_texts)
                batch_time = time.time() - batch_start_time

                # Log progress for large datasets
                if total_batches > 5:
                    logger.info(
                        f"Embedding batch {batch_idx+1}/{total_batches} completed in {batch_time:.2f}s"
                    )

                embeddings_list.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch {batch_idx+1}: {str(e)}")
                # Skip this batch but continue with others
                log_error(f"Skipping embedding batch {batch_idx+1} due to error: {str(e)}")
                self._handle_failed_batch(batch_idx, batch_texts, embeddings_list)

        # Concatenate all embeddings
        if not embeddings_list:
            logger.error("No embeddings were generated for deduplication")
            return None

        return np.vstack(embeddings_list).astype("float32")

    def _handle_failed_batch(self, batch_idx: int, batch_texts: List[str],
                           embeddings_list: List[np.ndarray]) -> None:
        """Handle failed embedding batch"""
        # Create empty embeddings as placeholder to maintain index alignment
        if batch_idx == 0 and not embeddings_list:
            # If this is the first batch, we need to know the embedding dimension
            # Try with a single example to get the dimension
            try:
                sample_embedding = self.embedding_model.encode(["sample text"])
                empty_batch = np.zeros(
                    (len(batch_texts), sample_embedding.shape[1]), dtype=np.float32
                )
                embeddings_list.append(empty_batch)
            except Exception:
                # If even that fails, we can't continue
                raise ValueError("Cannot determine embedding dimension for deduplication")
        elif embeddings_list:
            # Use the dimension from previous batches
            empty_batch = np.zeros(
                (len(batch_texts), embeddings_list[0].shape[1]), dtype=np.float32
            )
            embeddings_list.append(empty_batch)

    def _perform_deduplication_by_size(self, embeddings_np: np.ndarray, valid_indices: List[int],
                                     chunks: List[Dict[str, Any]], indices_to_keep: Set[int],
                                     similarity_threshold: float, chunk_count: int) -> None:
        """Perform deduplication using strategy based on dataset size"""
        if chunk_count > 50000:
            # For extremely large datasets: use hierarchical clustering with HNSW index
            logger.info(
                f"Using hierarchical clustering with HNSW for very large dataset ({chunk_count} chunks)"
            )
            self._dedup_with_hnsw(
                embeddings_np, valid_indices, chunks, indices_to_keep, similarity_threshold
            )
        elif chunk_count > 10000:
            # For large datasets: use optimized IVF index with PQ compression
            logger.info(
                f"Using IVF with PQ compression for large dataset ({chunk_count} chunks)"
            )
            self._dedup_with_ivfpq(
                embeddings_np, valid_indices, chunks, indices_to_keep, similarity_threshold
            )
        else:
            # For smaller datasets: use standard IVF index
            logger.info(f"Using standard IVF index for dataset ({chunk_count} chunks)")
            self._dedup_with_ivf(
                embeddings_np, valid_indices, chunks, indices_to_keep, similarity_threshold
            )

    def _process_search_results(self, batch_start: int, similarities: np.ndarray,
                              neighbors: np.ndarray, valid_indices: List[int],
                              chunks: List[Dict[str, Any]], indices_to_keep: Set[int],
                              similarity_threshold: float) -> None:
        """
        Process the search results from FAISS index to identify and remove duplicate chunks.

        Args:
            batch_start: Starting index of the current batch
            similarities: Matrix of similarity scores
            neighbors: Matrix of neighbor indices
            valid_indices: List mapping embedding indices to chunk indices
            chunks: Original chunks list
            indices_to_keep: Set of indices to keep (modified in-place)
            similarity_threshold: Similarity threshold for deduplication
        """
        try:
            num_queries = similarities.shape[0]

            for i in range(num_queries):
                original_index = valid_indices[batch_start + i]
                if original_index not in indices_to_keep:
                    continue  # Skip if already marked for removal

                for j in range(1, neighbors.shape[1]):  # Start from 1 to skip self-comparison
                    neighbor_index_in_batch = neighbors[i, j]
                    if neighbor_index_in_batch == -1:
                        break  # No more neighbors to process

                    neighbor_index = valid_indices[neighbor_index_in_batch]

                    if neighbor_index in indices_to_keep:
                        similarity = similarities[i, j]
                        if similarity > similarity_threshold:
                            # Mark the neighbor for removal
                            indices_to_keep.discard(neighbor_index)
                            logger.debug(
                                f"Duplicate found: Chunk {neighbor_index} is similar to chunk {original_index} (similarity: {similarity:.4f})"
                            )

        except Exception as e:
            logger.error(f"Error processing search results: {str(e)}")
            log_error(f"Error processing search results: {str(e)}", include_traceback=True)

    def _dedup_with_ivfpq(self, embeddings: np.ndarray, valid_indices: List[int],
                         chunks: List[Dict[str, Any]], indices_to_keep: Set[int],
                         similarity_threshold: float) -> None:
        """
        Deduplicate using an IVF index with Product Quantization (PQ) compression.
        Optimized for large datasets to balance speed and memory usage.
        """
        try:
            dim = embeddings.shape[1]
            num_embeddings = len(embeddings)
            
            # Calculate proper cluster count based on FAISS requirements
            min_training_per_cluster = 39
            nlist_sqrt = int(np.sqrt(num_embeddings))
            max_clusters_safe = num_embeddings // min_training_per_cluster
            nlist = min(nlist_sqrt, max_clusters_safe, 256)  # Cap at 256
            nlist = max(nlist, 4)  # Minimum 4 clusters
            
            logger.debug(f"Using {nlist} clusters for IVF-PQ deduplication ({num_embeddings} embeddings)")
            
            m = 16  # Number of subvectors for PQ
            nbits = 8  # Bits per subvector

            # Define the product quantizer
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

            # Train the index
            index.train(embeddings)
            index.add(embeddings)

            # Process in batches to avoid memory issues
            batch_size = 1000
            k = 10  # Number of neighbors to retrieve

            for batch_start in range(0, len(embeddings), batch_size):
                batch_end = min(batch_start + batch_size, len(embeddings))
                batch_embeddings = embeddings[batch_start:batch_end]

                # Search for similar chunks
                index.nprobe = 10  # Adjust nprobe for better recall
                similarities, neighbors = index.search(batch_embeddings, k=k)

                # Process results
                self._process_search_results(
                    batch_start, similarities, neighbors, valid_indices,
                    chunks, indices_to_keep, similarity_threshold
                )

        except Exception as e:
            logger.error(f"Error in IVF with PQ deduplication: {str(e)}")
            log_error(f"IVF with PQ deduplication failed: {str(e)}", include_traceback=True)

    def _dedup_with_ivf(self, embeddings: np.ndarray, valid_indices: List[int],
                       chunks: List[Dict[str, Any]], indices_to_keep: Set[int],
                       similarity_threshold: float) -> None:
        """
        Deduplicate using an Inverted File (IVF) index.
        Suitable for medium-sized datasets where speed is important.
        """
        try:
            dim = embeddings.shape[1]
            num_embeddings = len(embeddings)
            
            # Calculate proper cluster count based on FAISS requirements
            min_training_per_cluster = 39
            nlist_sqrt = int(np.sqrt(num_embeddings))
            max_clusters_safe = num_embeddings // min_training_per_cluster
            nlist = min(nlist_sqrt, max_clusters_safe, 256)  # Cap at 256
            nlist = max(nlist, 4)  # Minimum 4 clusters
            
            logger.debug(f"Using {nlist} clusters for IVF deduplication ({num_embeddings} embeddings)")

            # Define the IVF quantizer
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train the index
            index.train(embeddings)
            index.add(embeddings)

            # Process in batches to avoid memory issues
            batch_size = 1000
            k = 10  # Number of neighbors to retrieve

            for batch_start in range(0, len(embeddings), batch_size):
                batch_end = min(batch_start + batch_size, len(embeddings))
                batch_embeddings = embeddings[batch_start:batch_end]

                # Search for similar chunks
                index.nprobe = 10  # Adjust nprobe for better recall
                similarities, neighbors = index.search(batch_embeddings, k=k)

                # Process results
                self._process_search_results(
                    batch_start, similarities, neighbors, valid_indices,
                    chunks, indices_to_keep, similarity_threshold
                )

        except Exception as e:
            logger.error(f"Error in IVF deduplication: {str(e)}")
            log_error(f"IVF deduplication failed: {str(e)}", include_traceback=True)

    def _dedup_with_hnsw(self, embeddings: np.ndarray, valid_indices: List[int],
                        chunks: List[Dict[str, Any]], indices_to_keep: Set[int],
                        similarity_threshold: float) -> None:
        """
        Deduplicate using Hierarchical Navigable Small World (HNSW) graph-based index.
        Optimized for very large datasets with millions of vectors.
        """
        try:
            dim = embeddings.shape[1]

            # Create HNSW index with optimized parameters
            # M: number of connections per layer (higher = better recall but more memory)
            # efConstruction: build-time accuracy parameter (higher = better recall but slower build)
            M = 16  # Good balance between accuracy and memory usage
            ef_construction = 128  # Higher values give better recall but slower build

            index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = 64  # Search-time accuracy parameter

            # Add vectors to index
            index.add(embeddings)

            # Process in batches to avoid memory issues
            batch_size = 1000
            k = 10  # Number of neighbors to retrieve

            for batch_start in range(0, len(embeddings), batch_size):
                batch_end = min(batch_start + batch_size, len(embeddings))
                batch_embeddings = embeddings[batch_start:batch_end]

                # Search for similar chunks
                similarities, neighbors = index.search(batch_embeddings, k=k)

                # Process results
                self._process_search_results(
                    batch_start, similarities, neighbors, valid_indices,
                    chunks, indices_to_keep, similarity_threshold
                )

        except Exception as e:
            logger.error(f"Error in HNSW deduplication: {str(e)}")
            log_error(f"HNSW deduplication failed: {str(e)}", include_traceback=True)

    def get_deduplication_stats(self, original_count: int, deduplicated_count: int) -> Dict[str, Any]:
        """
        Get deduplication statistics

        Args:
            original_count: Original number of chunks
            deduplicated_count: Number of chunks after deduplication

        Returns:
            Dictionary with deduplication statistics
        """
        removed_count = original_count - deduplicated_count
        removal_percentage = (removed_count / original_count) * 100 if original_count > 0 else 0

        return {
            "original_chunks": original_count,
            "deduplicated_chunks": deduplicated_count,
            "removed_chunks": removed_count,
            "removal_percentage": removal_percentage,
            "compression_ratio": deduplicated_count / original_count if original_count > 0 else 1.0
        }
