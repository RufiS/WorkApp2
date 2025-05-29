# Unified retrieval system with optimized performance
import logging
import time
import os
from typing import List, Dict, Any, Optional, Tuple, Union

from utils.config import retrieval_config, performance_config
from core.document_processor import DocumentProcessor
from error_handling.enhanced_decorators import with_timing, with_error_tracking

# Setup logging
logger = logging.getLogger(__name__)


class UnifiedRetrievalSystem:
    """Unified system for retrieving relevant context from indexed documents"""

    def __init__(self, document_processor: Optional[DocumentProcessor] = None):
        """
        Initialize the retrieval system

        Args:
            document_processor: Document processor instance (creates new one if None)
        """
        self.document_processor = document_processor or DocumentProcessor()
        self.top_k = retrieval_config.top_k
        self.similarity_threshold = retrieval_config.similarity_threshold
        self.max_context_length = retrieval_config.max_context_length

        # Initialize metrics
        self.total_queries = 0
        self.query_times = []
        self.max_query_times = 100

        logger.info(f"Unified retrieval system initialized with top_k={self.top_k}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get retrieval system metrics

        Returns:
            Dictionary with retrieval metrics
        """
        metrics = {"total_queries": self.total_queries}

        # Calculate average query time
        if self.query_times:
            metrics["avg_query_time"] = sum(self.query_times) / len(self.query_times)
            metrics["min_query_time"] = min(self.query_times)
            metrics["max_query_time"] = max(self.query_times)
        else:
            metrics["avg_query_time"] = 0.0
            metrics["min_query_time"] = 0.0
            metrics["max_query_time"] = 0.0

        # Add document processor metrics
        processor_metrics = self.document_processor.get_metrics()
        metrics.update({f"processor_{k}": v for k, v in processor_metrics.items()})

        return metrics

    def index_documents(self, file_paths: List[str]) -> None:
        """
        Index documents for retrieval

        Args:
            file_paths: List of paths to document files
        """
        self.document_processor.process_documents(file_paths)
        logger.info(f"Indexed {len(file_paths)} documents")

    def save_index(self, index_dir: str) -> None:
        """
        Save the search index to disk

        Args:
            index_dir: Directory to save the index
        """
        os.makedirs(index_dir, exist_ok=True)
        self.document_processor.save_index(index_dir)

    def load_index(self, index_dir: str) -> None:
        """
        Load a search index from disk

        Args:
            index_dir: Directory containing the index
        """
        self.document_processor.load_index(index_dir)

    @with_timing(threshold=0.5)
    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[str, float, int, List[float]]:
        """
        Retrieve relevant context for a query with intelligent routing based on configuration

        Args:
            query: Query string
            top_k: Number of top results to return (None for default)

        Returns:
            Tuple of (formatted context string, retrieval time in seconds, number of chunks, list of retrieval scores)
        """
        # Use default top_k if not specified
        top_k = top_k or self.top_k

        # Log the actual top_k value used and query (truncated for privacy/size)
        query_preview = query[:50] + "..." if len(query) > 50 else query
        
        # Intelligent routing based on configuration settings
        search_method = "basic"
        if performance_config.enable_reranking:
            search_method = "reranking"
            logger.info(f"Using reranking retrieval for query: '{query_preview}' with top_k={top_k}")
            return self.retrieve_with_reranking(query, top_k)
        elif retrieval_config.enhanced_mode:
            search_method = "hybrid"
            logger.info(f"Using hybrid search for query: '{query_preview}' with top_k={top_k}")
            return self.retrieve_with_hybrid_search(query, top_k)
        else:
            logger.info(f"Using basic vector search for query: '{query_preview}' with top_k={top_k}")
        
        # Continue with basic vector search if no enhanced methods are enabled

        # Search for relevant chunks
        start_time = time.time()
        try:
            # First check if index exists and is loaded
            if not self.document_processor.has_index():
                logger.warning("No index has been built. Process documents first.")
                results = []
            else:
                results = self.document_processor.search(query, top_k=top_k)

                # Filter by similarity threshold if enabled
                if self.similarity_threshold > 0:
                    pre_filter_count = len(results)
                    results = [r for r in results if r.get("score", 0) >= self.similarity_threshold]
                    filtered_count = pre_filter_count - len(results)

                    # Log the filtering results
                    logger.info(
                        f"Similarity filtering: {pre_filter_count} chunks → {len(results)} chunks (filtered {filtered_count} below threshold {self.similarity_threshold})"
                    )
                else:
                    # Log the actual number of chunks retrieved
                    logger.info(
                        f"Retrieved {len(results)} chunks (no similarity threshold applied)"
                    )
        except ValueError as e:
            # Handle the case where no index has been built
            logger.warning(f"Search failed: {str(e)}")
            results = []
        except Exception as e:
            # Handle other exceptions
            logger.error(f"Error in search: {str(e)}")
            results = []

        # Update metrics
        self.total_queries += 1
        query_time = time.time() - start_time
        self.query_times.append(query_time)
        if len(self.query_times) > self.max_query_times:
            self.query_times = self.query_times[-self.max_query_times :]

        # Run deduplication after retrieval
        if len(results) > 1:
            try:
                # Simple deduplication based on text similarity
                deduplicated_results = []
                seen_texts = set()
                duplicates_found = 0

                for result in results:
                    text = result.get("text", "")
                    # Create a simplified version for comparison (lowercase, whitespace normalized)
                    simplified_text = " ".join(text.lower().split())

                    # Skip if we've seen a very similar text
                    if any(
                        self._text_similarity(simplified_text, seen) > 0.8 for seen in seen_texts
                    ):
                        logger.debug(f"Skipping duplicate chunk: {text[:50]}...")
                        duplicates_found += 1
                        continue

                    # Add to results and mark as seen
                    deduplicated_results.append(result)
                    seen_texts.add(simplified_text)

                # Update results with deduplicated version
                logger.info(
                    f"Deduplication: {len(results)} chunks u2192 {len(deduplicated_results)} chunks (removed {duplicates_found} duplicates)"
                )
                results = deduplicated_results
            except Exception as e:
                logger.warning(f"Error during deduplication: {str(e)}")
                # Continue with original results if deduplication fails

        # Format context
        context = self._format_context(results)

        # Extract scores for progress tracking
        retrieval_scores = [result.get("score", 0.0) for result in results]

        # Calculate retrieval time
        retrieval_time = time.time() - start_time

        # Log the actual chunks sent to the LLM and performance metrics
        if context:
            logger.info(
                f"Retrieval complete: {len(results)} chunks, {len(context)} chars, {retrieval_time:.3f}s"
            )
            # Log a preview of the context
            if logger.isEnabledFor(logging.DEBUG):
                context_preview = context[:200] + "..." if len(context) > 200 else context
                logger.debug(f"Context preview: {context_preview}")
        else:
            error_msg = f"Retrieved empty context for query: {query[:50]}..."
            logger.warning(error_msg)
            # Log to central error log
            try:
                from utils.error_logging import log_error

                log_error(error_msg, include_traceback=False)
            except ImportError:
                # Fallback to simple file logging if error_logging module is not available
                try:
                    from utils.config import resolve_path

                    fallback_log_path = resolve_path("./logs/workapp_errors.log", create_dir=True)
                except ImportError:
                    fallback_log_path = "./logs/workapp_errors.log"
                    os.makedirs(os.path.dirname(fallback_log_path), exist_ok=True)

                with open(fallback_log_path, "a") as error_log:
                    error_log.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {error_msg}\n"
                    )

        # Return formatted context, retrieval time, number of chunks, and retrieval scores
        return context, query_time, len(results), retrieval_scores

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple similarity between two texts

        Args:
            text1: First text
            text2: Second text

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
        """
        Format search results into a context string

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

    def retrieve_with_reranking(
        self, query: str, top_k: Optional[int] = None, rerank_top_k: Optional[int] = None
    ) -> Tuple[str, float, int, List[float]]:
        """
        Retrieve relevant context with reranking for better results

        Args:
            query: Query string
            top_k: Number of top results to return (None for default)
            rerank_top_k: Number of results to rerank (None for default)

        Returns:
            Tuple of (formatted context string, retrieval time in seconds, number of chunks, list of retrieval scores)
        """
        # Use default values if not specified
        top_k = top_k or self.top_k
        rerank_top_k = rerank_top_k or (top_k * 3)  # Default to 3x top_k for reranking

        # Get initial results (more than needed for reranking)
        start_time = time.time()
        try:
            results = self.document_processor.search(query, top_k=rerank_top_k)
        except ValueError as e:
            # Handle the case where no index has been built
            logger.warning(f"Search failed: {str(e)}")
            return "No relevant information found.", time.time() - start_time, 0, []
        except Exception as e:
            # Handle other exceptions
            logger.error(f"Error in search: {str(e)}")
            return "No relevant information found.", time.time() - start_time, 0, []

        if not results or len(results) <= top_k:
            # If we have fewer results than requested, return them all
            return self.retrieve(query, top_k=top_k)

        # Rerank results if reranking is enabled
        if performance_config.enable_reranking:
            try:
                # Rerank using cross-encoder if available
                reranked_results = self._rerank_results(query, results)
                results = reranked_results[:top_k]
            except Exception as e:
                logger.error(f"Error in reranking: {str(e)}")
                # Fall back to initial results if reranking fails
                results = results[:top_k]
        else:
            # No reranking, just take top_k results
            results = results[:top_k]

        # Filter by similarity threshold if enabled
        if self.similarity_threshold > 0:
            results = [r for r in results if r["score"] >= self.similarity_threshold]

        # Format context
        context = self._format_context(results)

        # Extract scores for progress tracking
        retrieval_scores = [result.get("score", 0.0) for result in results]

        # Calculate retrieval time
        retrieval_time = time.time() - start_time

        # Update metrics
        self.total_queries += 1
        self.query_times.append(retrieval_time)
        if len(self.query_times) > self.max_query_times:
            self.query_times = self.query_times[-self.max_query_times :]

        # Return formatted context, retrieval time, number of chunks, and retrieval scores
        return context, retrieval_time, len(results), retrieval_scores

    def retrieve_with_hybrid_search(
        self, query: str, top_k: Optional[int] = None
    ) -> Tuple[str, float, int, List[float]]:
        """
        Retrieve relevant context using hybrid search (vector + keyword)

        Args:
            query: Query string
            top_k: Number of top results to return (None for default)

        Returns:
            Tuple of (formatted context string, retrieval time in seconds, number of chunks, list of retrieval scores)
        """
        # Use default top_k if not specified
        top_k = top_k or self.top_k
        start_time = time.time()
        
        try:
            # First check if index exists and is loaded
            if not self.document_processor.has_index():
                logger.error("HYBRID DEBUG: No index has been built. Process documents first.")
                return "No relevant information found.", time.time() - start_time, 0, []
            
            # Debug: Check chunk availability
            chunks_available = hasattr(self.document_processor, 'chunks') and self.document_processor.chunks
            logger.info(f"HYBRID DEBUG: Chunks available: {chunks_available}, chunk count: {len(self.document_processor.chunks) if chunks_available else 0}")
            
            # Get vector search results
            logger.info(f"HYBRID DEBUG: Getting vector search results for query: '{query[:50]}...'")
            vector_results = self.document_processor.search(query, top_k=top_k * 2)  # Get more for hybrid combination
            logger.info(f"HYBRID DEBUG: Vector search returned {len(vector_results)} results")
            
            # Get keyword search results
            logger.info(f"HYBRID DEBUG: Getting keyword search results...")
            keyword_results = self._perform_keyword_search(query, top_k * 2)
            logger.info(f"HYBRID DEBUG: Keyword search returned {len(keyword_results)} results")
            
            # Debug: Log sample results
            if vector_results:
                logger.info(f"HYBRID DEBUG: Sample vector result score: {vector_results[0].get('score', 'N/A')}")
            if keyword_results:
                logger.info(f"HYBRID DEBUG: Sample keyword result score: {keyword_results[0].get('score', 'N/A')}")
            
            # Combine and rerank results based on vector weight
            logger.info(f"HYBRID DEBUG: Combining results with vector_weight={retrieval_config.vector_weight}")
            hybrid_results = self._combine_search_results(
                vector_results, keyword_results, retrieval_config.vector_weight
            )
            logger.info(f"HYBRID DEBUG: Combined search returned {len(hybrid_results)} results")
            
            # Take top_k results
            results = hybrid_results[:top_k]
            logger.info(f"HYBRID DEBUG: Taking top {top_k} results: {len(results)} results")
            
            # Filter by similarity threshold if enabled
            if self.similarity_threshold > 0:
                pre_filter_count = len(results)
                results = [r for r in results if r.get("score", 0) >= self.similarity_threshold]
                filtered_count = pre_filter_count - len(results)
                logger.info(
                    f"HYBRID DEBUG: Similarity filtering: {pre_filter_count} chunks → {len(results)} chunks (filtered {filtered_count} below threshold {self.similarity_threshold})"
                )
            else:
                logger.info(f"HYBRID DEBUG: No similarity threshold applied")
            
            # Update metrics
            self.total_queries += 1
            query_time = time.time() - start_time
            self.query_times.append(query_time)
            if len(self.query_times) > self.max_query_times:
                self.query_times = self.query_times[-self.max_query_times :]
            
            # Format context
            context = self._format_context(results)
            logger.info(f"HYBRID DEBUG: Formatted context length: {len(context)} chars")
            
            # Extract scores for progress tracking
            retrieval_scores = [result.get("score", 0.0) for result in results]
            
            # Log results
            logger.info(
                f"Hybrid search complete: {len(results)} chunks, {len(context)} chars, {query_time:.3f}s (vector_weight={retrieval_config.vector_weight})"
            )
            
            return context, query_time, len(results), retrieval_scores
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}", exc_info=True)
            # Fall back to basic vector search
            logger.info("Falling back to basic vector search")
            return self._perform_basic_search(query, top_k)
    
    def _perform_keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search on the document chunks

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of search results with keyword-based scores
        """
        try:
            logger.info(f"KEYWORD DEBUG: Starting keyword search for query: '{query[:50]}...'")
            
            # Get all chunks from the document processor
            if not hasattr(self.document_processor, 'chunks'):
                logger.error(f"KEYWORD DEBUG: document_processor has no 'chunks' attribute")
                return []
                
            if not self.document_processor.chunks:
                logger.error(f"KEYWORD DEBUG: document_processor.chunks is empty or None")
                logger.info(f"KEYWORD DEBUG: chunks type: {type(self.document_processor.chunks)}")
                return []
                
            chunks_count = len(self.document_processor.chunks)
            logger.info(f"KEYWORD DEBUG: Found {chunks_count} chunks to search")
            
            # Log sample chunk for debugging
            if chunks_count > 0:
                sample_chunk = self.document_processor.chunks[0]
                logger.info(f"KEYWORD DEBUG: Sample chunk keys: {list(sample_chunk.keys()) if isinstance(sample_chunk, dict) else 'Not a dict'}")
                if isinstance(sample_chunk, dict) and 'text' in sample_chunk:
                    logger.info(f"KEYWORD DEBUG: Sample chunk text length: {len(sample_chunk['text'])}")
                    logger.info(f"KEYWORD DEBUG: Sample chunk text preview: '{sample_chunk['text'][:100]}...'")
            
            # Simple keyword matching with TF-IDF-like scoring
            query_words = set(query.lower().split())
            logger.info(f"KEYWORD DEBUG: Query words: {query_words}")
            scored_results = []
            
            for i, chunk in enumerate(self.document_processor.chunks):
                if i < 3:  # Log first 3 chunks for debugging
                    logger.info(f"KEYWORD DEBUG: Processing chunk {i}, type: {type(chunk)}")
                    
                chunk_text = chunk.get("text", "").lower()
                chunk_words = set(chunk_text.split())
                
                # Calculate keyword overlap score
                if query_words and chunk_words:
                    intersection = len(query_words.intersection(chunk_words))
                    union = len(query_words.union(chunk_words))
                    keyword_score = intersection / len(query_words) if query_words else 0.0
                    
                    # Boost score if query words appear multiple times
                    for word in query_words:
                        if word in chunk_text:
                            keyword_score += chunk_text.count(word) * 0.1
                    
                    if keyword_score > 0:
                        result = chunk.copy()
                        result["score"] = keyword_score
                        scored_results.append(result)
                        
                        if len(scored_results) <= 3:  # Log first few matches
                            logger.info(f"KEYWORD DEBUG: Found match {len(scored_results)}, score: {keyword_score:.3f}, intersection: {intersection}")
            
            logger.info(f"KEYWORD DEBUG: Found {len(scored_results)} keyword matches")
            
            # Sort by keyword score and return top results
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = scored_results[:top_k]
            
            logger.info(f"KEYWORD DEBUG: Returning top {len(final_results)} keyword results")
            if final_results:
                logger.info(f"KEYWORD DEBUG: Top result score: {final_results[0]['score']:.3f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"KEYWORD DEBUG: Error in keyword search: {str(e)}", exc_info=True)
            return []
    
    def _combine_search_results(
        self, vector_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]], vector_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and keyword search results using weighted scoring

        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            vector_weight: Weight for vector scores (0.0 to 1.0)

        Returns:
            Combined and ranked results
        """
        keyword_weight = 1.0 - vector_weight
        combined_scores = {}
        all_results = {}
        
        # Normalize vector scores (assuming they're cosine similarities between 0 and 1)
        if vector_results:
            max_vector_score = max(r.get("score", 0) for r in vector_results)
            min_vector_score = min(r.get("score", 0) for r in vector_results)
            vector_range = max_vector_score - min_vector_score if max_vector_score != min_vector_score else 1.0
        
        # Process vector results
        for result in vector_results:
            chunk_id = id(result.get("text", ""))  # Use text as unique identifier
            normalized_score = (result.get("score", 0) - min_vector_score) / vector_range if vector_results else 0
            combined_scores[chunk_id] = vector_weight * normalized_score
            all_results[chunk_id] = result.copy()
        
        # Normalize keyword scores
        if keyword_results:
            max_keyword_score = max(r.get("score", 0) for r in keyword_results)
            min_keyword_score = min(r.get("score", 0) for r in keyword_results)
            keyword_range = max_keyword_score - min_keyword_score if max_keyword_score != min_keyword_score else 1.0
        
        # Process keyword results
        for result in keyword_results:
            chunk_id = id(result.get("text", ""))
            normalized_score = (result.get("score", 0) - min_keyword_score) / keyword_range if keyword_results else 0
            
            if chunk_id in combined_scores:
                # Combine with existing vector score
                combined_scores[chunk_id] += keyword_weight * normalized_score
            else:
                # Only keyword score
                combined_scores[chunk_id] = keyword_weight * normalized_score
                all_results[chunk_id] = result.copy()
        
        # Create final results with combined scores
        final_results = []
        for chunk_id, combined_score in combined_scores.items():
            result = all_results[chunk_id]
            result["score"] = combined_score
            result["hybrid_score"] = combined_score  # Keep original for debugging
            final_results.append(result)
        
        # Sort by combined score
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.debug(f"Combined {len(vector_results)} vector + {len(keyword_results)} keyword results into {len(final_results)} hybrid results")
        
        return final_results
    
    def _perform_basic_search(self, query: str, top_k: int) -> Tuple[str, float, int, List[float]]:
        """
        Perform basic vector search (fallback method)

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            Tuple of (formatted context string, retrieval time in seconds, number of chunks, list of retrieval scores)
        """
        start_time = time.time()
        try:
            results = self.document_processor.search(query, top_k=top_k)
            
            # Filter by similarity threshold if enabled
            if self.similarity_threshold > 0:
                results = [r for r in results if r.get("score", 0) >= self.similarity_threshold]
            
            context = self._format_context(results)
            retrieval_scores = [result.get("score", 0.0) for result in results]
            query_time = time.time() - start_time
            
            return context, query_time, len(results), retrieval_scores
            
        except Exception as e:
            logger.error(f"Error in basic search: {str(e)}")
            return "No relevant information found.", time.time() - start_time, 0, []

    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using a cross-encoder model

        Args:
            query: Query string
            results: Initial search results

        Returns:
            Reranked results
        """
        try:
            from sentence_transformers import CrossEncoder

            # Initialize cross-encoder
            reranker_model = getattr(
                retrieval_config, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            cross_encoder = CrossEncoder(reranker_model)

            # Prepare pairs for reranking
            pairs = [(query, result["text"]) for result in results]

            # Get scores from cross-encoder
            scores = cross_encoder.predict(pairs)

            # Update scores in results
            for i, score in enumerate(scores):
                results[i]["score"] = float(score)

            # Sort by new scores
            reranked_results = sorted(results, key=lambda x: x["score"], reverse=True)

            return reranked_results
        except ImportError:
            logger.warning("CrossEncoder not available for reranking")
            return results
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            return results

    def query(
        self,
        query: str,
        filtered_results: Optional[List[Dict[str, Any]]] = None,
        use_enhanced_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a query and return an answer with context

        Args:
            query: The user's question
            filtered_results: Pre-filtered search results (if None, will perform search)
            use_enhanced_context: Whether to use enhanced context processing

        Returns:
            Dictionary with answer, context, and metadata
        """
        try:
            from utils.text_processing.context_enhancement.context_processor import (
                EnhancedContextProcessor,
            )
            from llm.llm_service import LLMService
        except ImportError:
            logger.error("Failed to import required modules for query processing")
            return {
                "answer": "Error: Required modules not available",
                "context": "",
                "metadata": {"error": "Import error"},
            }

        # Get search results if not provided
        if filtered_results is None:
            try:
                results = self.document_processor.search(query, top_k=self.top_k)
                # Filter by similarity threshold
                filtered_results = [r for r in results if r["score"] >= self.similarity_threshold]
            except Exception as e:
                logger.error(f"Error in search: {str(e)}")
                return {
                    "answer": f"Error retrieving information: {str(e)}",
                    "context": "",
                    "metadata": {"error": str(e)},
                }

        # Process context
        if use_enhanced_context and filtered_results:
            try:
                # Use enhanced context processing if available
                processor = EnhancedContextProcessor()
                if hasattr(processor, "process_with_metadata"):
                    result = processor.process_with_metadata(query, filtered_results)
                    context = str(result["context"])
                    metadata = result["metadata"]
                elif hasattr(processor, "process"):
                    context = processor.process(query, filtered_results)
                    metadata = {"context_processing": "enhanced"}
                else:
                    # Fall back to basic context processing
                    context = "\n\n".join([result["text"] for result in filtered_results])
                    metadata = {
                        "context_processing": "basic",
                        "error": "EnhancedContextProcessor missing required methods",
                    }
            except Exception as e:
                logger.error(f"Error in enhanced context processing: {str(e)}")
                # Fall back to basic context processing
                context = "\n\n".join([result["text"] for result in filtered_results])
                metadata = {"context_processing": "basic", "error": str(e)}
        else:
            # Basic context processing
            context = "\n\n".join([result["text"] for result in filtered_results])
            metadata = {"context_processing": "basic"}

        # Generate answer
        try:
            # Create LLM service if not available
            from utils.config import app_config
            from llm.llm_service_enhanced import LLMService
            from utils.llm_service_enhanced import LLMService

            llm_service = LLMService(app_config.api_keys.get("openai", ""))
            answer_data = llm_service.generate_answer(query, str(context))

            return {
                "answer": answer_data.get("content", "No answer generated"),
                "context": context,
                "metadata": {
                    **metadata,
                    "model": answer_data.get("model", ""),
                    "tokens": answer_data.get("usage", {}).get("total_tokens", 0),
                },
            }
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "context": context,
                "metadata": {**metadata, "error": str(e)},
            }
