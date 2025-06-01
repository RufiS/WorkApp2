"""Main document ingestion manager - orchestrates all ingestion components"""

import logging
from typing import List, Dict, Any, Set, Optional
from sentence_transformers import SentenceTransformer

from core.config import retrieval_config, performance_config
from utils.common.error_handler import CommonErrorHandler, with_error_context
from utils.common.metrics_collector import metrics_collector
from utils.error_handling.enhanced_decorators import with_advanced_retry, with_timing

from .chunk_cache import ChunkCache
from .enhanced_file_processor import EnhancedFileProcessor
from .chunk_optimizer import ChunkOptimizer
from .metadata_handler import MetadataHandler
from .deduplication_engine import DeduplicationEngine

logger = logging.getLogger(__name__)


class DocumentIngestion:
    """Main document ingestion orchestrator combining all ingestion capabilities"""
    
    def __init__(self, embedding_model_name: str = None):
        """
        Initialize the document ingestion processor
        
        Args:
            embedding_model_name: Name of the embedding model to use (defaults to config value)
        """
        self.embedding_model_name = embedding_model_name or retrieval_config.embedding_model
        self.chunk_size = retrieval_config.chunk_size
        self.chunk_overlap = retrieval_config.chunk_overlap
        
        # Initialize embedding model for dimension verification
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize component modules
        self.cache = ChunkCache()
        self.file_processor = EnhancedFileProcessor(self.chunk_size, self.chunk_overlap)
        self.chunk_optimizer = ChunkOptimizer(min_chunk_size=50)
        self.metadata_handler = MetadataHandler()
        self.deduplication_engine = DeduplicationEngine(self.embedding_model)
        
        # Initialize metrics
        self.total_documents = 0
        self.total_chunks = 0
        self.processed_files: Set[str] = set()
        
        # Store chunk parameters for backward compatibility
        self.saved_chunk_params = (self.chunk_size, self.chunk_overlap)
        
        logger.info(f"Document ingestion initialized with model {self.embedding_model_name}")
    
    @property
    def chunk_cache(self) -> Dict[str, Any]:
        """Get chunk cache dictionary for backward compatibility"""
        return self.cache.cache
    
    @property
    def cache_hits(self) -> int:
        """Get cache hits for backward compatibility"""
        return self.cache.cache_hits
    
    @property
    def cache_misses(self) -> int:
        """Get cache misses for backward compatibility"""
        return self.cache.cache_misses
    
    @with_timing(threshold=1.0)
    @with_advanced_retry(max_attempts=3, backoff_factor=2)
    def process_file(self, file) -> List[Dict[str, Any]]:
        """
        Process a file object (from streamlit file uploader)
        
        Args:
            file: File object from streamlit file uploader
            
        Returns:
            List of document chunks with metadata
        """
        try:
            chunks = self.file_processor.process_file_upload(file)
            
            # Update metrics
            if chunks:
                metrics_collector.increment_counter("files_processed")
                metrics_collector.record_metric("chunks_per_file", len(chunks))
            
            return chunks
        except Exception as e:
            error_msg = CommonErrorHandler.handle_processing_error(
                "DocumentIngestion", "file processing", e
            )
            raise RuntimeError(error_msg) from e
    
    @with_timing(threshold=1.0)
    @with_advanced_retry(max_attempts=3, backoff_factor=2)
    def load_and_chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a document and split it into chunks with caching
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks with metadata
        """
        try:
            # Check cache first
            cache_key = self.cache.get_cache_key(file_path, self.chunk_size, self.chunk_overlap)
            cached_chunks = self.cache.get_from_cache(cache_key)
            if cached_chunks is not None:
                logger.info(f"Cache hit for document {file_path}")
                return cached_chunks
            
            # Process the file
            chunks = self.file_processor.load_and_chunk_document(file_path)
            
            if not chunks:
                return []
            
            # Compute document hash for metadata
            doc_hash = self.metadata_handler.compute_document_hash(file_path)
            
            # Handle small chunks
            if any(len(chunk.get("text", "")) < 50 for chunk in chunks):
                chunks = self.chunk_optimizer.handle_small_chunks(chunks, file_path)
            
            # Add document hash to chunk metadata
            for chunk in chunks:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                chunk["metadata"]["doc_hash"] = doc_hash
            
            # Update metrics
            self.total_documents += 1
            self.total_chunks += len(chunks)
            
            # Add to cache
            self.cache.add_to_cache(cache_key, chunks)
            
            # Update metadata with document hash
            self.metadata_handler.update_metadata_with_hash(file_path, doc_hash, len(chunks))
            
            # Record metrics
            metrics_collector.increment_counter("documents_processed")
            metrics_collector.record_metric("chunks_per_document", len(chunks))
            
            logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            error_msg = CommonErrorHandler.handle_processing_error(
                "DocumentIngestion", "document loading and chunking", e
            )
            raise
    
    def process_documents(self, file_paths: List[str], dry_run: bool = False) -> List[Dict[str, Any]]:
        """
        Process multiple documents and extract chunks
        
        Args:
            file_paths: List of paths to document files
            dry_run: If True, preview changes without processing
            
        Returns:
            List of all chunks from processed documents
        """
        all_chunks = []
        processed_files_count = 0
        skipped_files_count = 0
        failed_files_count = 0
        
        if dry_run:
            logger.info("[DRY RUN] Running in dry run mode - no changes will be made")
        
        try:
            # Process each document
            for file_path in file_paths:
                try:
                    # Skip already processed files
                    if file_path in self.processed_files:
                        logger.info(
                            f"{'[DRY RUN] ' if dry_run else ''}Skipping already processed file: {file_path}"
                        )
                        skipped_files_count += 1
                        continue
                    
                    # Load and chunk document
                    chunks = self.load_and_chunk_document(file_path)
                    all_chunks.extend(chunks)
                    processed_files_count += 1
                    
                    # Mark file as processed (only if not in dry run mode)
                    if not dry_run:
                        self.processed_files.add(file_path)
                    
                    logger.info(
                        f"{'[DRY RUN] ' if dry_run else ''}Processed {file_path}: {len(chunks)} chunks"
                    )
                except Exception as e:
                    logger.error(
                        f"{'[DRY RUN] ' if dry_run else ''}Error processing document {file_path}: {str(e)}"
                    )
                    failed_files_count += 1
            
            # Perform semantic deduplication if there are enough chunks
            if len(all_chunks) > 1:
                original_chunk_count = len(all_chunks)
                all_chunks = self.deduplication_engine.semantic_deduplication(all_chunks)
                logger.info(
                    f"{'[DRY RUN] ' if dry_run else ''}After semantic deduplication: "
                    f"{len(all_chunks)} chunks (removed {original_chunk_count - len(all_chunks)} duplicates)"
                )
                
                # Record deduplication metrics
                metrics_collector.record_metric("deduplication_ratio", 
                                               len(all_chunks) / original_chunk_count if original_chunk_count > 0 else 1.0)
            
            if dry_run:
                logger.info(
                    f"[DRY RUN] Would process {processed_files_count} new files, "
                    f"skipped {skipped_files_count} already processed files, "
                    f"failed {failed_files_count} files"
                )
                logger.info(f"[DRY RUN] Would extract {len(all_chunks)} chunks total")
            
            return all_chunks
            
        except Exception as e:
            error_msg = CommonErrorHandler.handle_processing_error(
                "DocumentIngestion", "multiple document processing", e
            )
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get ingestion metrics
        
        Returns:
            Dictionary with ingestion metrics
        """
        # Get metrics from individual components
        cache_stats = self.cache.get_cache_stats()
        
        # Combine with overall metrics
        metrics = {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "processed_files_count": len(self.processed_files),
            **cache_stats
        }
        
        # Add component-specific metrics
        if hasattr(self.chunk_optimizer, 'get_metrics'):
            optimizer_metrics = self.chunk_optimizer.get_metrics()
            metrics.update({f"optimizer_{k}": v for k, v in optimizer_metrics.items()})
        
        # Add processing statistics from metadata handler
        processing_stats = self.metadata_handler.get_processing_stats()
        metrics.update({f"metadata_{k}": v for k, v in processing_stats.items()})
        
        return metrics
    
    @with_error_context("cache management")
    def clear_cache(self) -> None:
        """Clear the document cache"""
        self.cache.clear_cache()
        logger.info("Document ingestion cache cleared")
    
    def get_processed_files(self) -> Set[str]:
        """Get set of processed files"""
        return self.processed_files.copy()
    
    def is_file_processed(self, file_path: str) -> bool:
        """
        Check if a file has been processed
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file has been processed
        """
        return file_path in self.processed_files or self.metadata_handler.is_document_processed(file_path)
    
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
        """
        Validate chunk structure and content
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        return self.chunk_optimizer.validate_chunks(chunks)
    
    def analyze_chunk_distribution(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the distribution of chunk sizes
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Dictionary with chunk size statistics
        """
        return self.chunk_optimizer.analyze_chunk_distribution(chunks)
    
    def optimize_chunk_boundaries(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize chunk boundaries to avoid splitting sentences
        
        Args:
            chunks: List of chunks to optimize
            
        Returns:
            List of optimized chunks
        """
        return self.chunk_optimizer.optimize_chunk_boundaries(chunks)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        return self.cache.get_cache_stats()
    
    def cleanup_expired_cache_entries(self) -> int:
        """
        Remove expired cache entries
        
        Returns:
            Number of entries removed
        """
        return self.cache.cleanup_expired_entries()
    
    def get_deduplication_stats(self, original_count: int, deduplicated_count: int) -> Dict[str, Any]:
        """Get deduplication statistics"""
        return self.deduplication_engine.get_deduplication_stats(original_count, deduplicated_count)
    
    # Backward compatibility methods
    def _get_cache_key(self, file_path: str, chunk_size: int, chunk_overlap: int) -> str:
        """Generate cache key (backward compatibility)"""
        return self.cache.get_cache_key(file_path, chunk_size, chunk_overlap)
    
    def _add_to_cache(self, key: str, chunks: List[Dict[str, Any]], ttl: Optional[float] = None) -> None:
        """Add to cache (backward compatibility)"""
        self.cache.add_to_cache(key, chunks, ttl)
    
    def _get_from_cache(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get from cache (backward compatibility)"""
        return self.cache.get_from_cache(key)
    
    def _handle_small_chunks(self, chunks: List[Dict[str, Any]], min_size: int = 50, 
                           file_path: str = None) -> List[Dict[str, Any]]:
        """Handle small chunks (backward compatibility)"""
        self.chunk_optimizer.min_chunk_size = min_size
        return self.chunk_optimizer.handle_small_chunks(chunks, file_path)
    
    def _semantic_deduplication(self, chunks: List[Dict[str, Any]], 
                              similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
        """Perform semantic deduplication (backward compatibility)"""
        return self.deduplication_engine.semantic_deduplication(chunks, similarity_threshold)
    
    def _compute_document_hash(self, file_path: str) -> str:
        """Compute document hash (backward compatibility)"""
        return self.metadata_handler.compute_document_hash(file_path)
    
    def _update_metadata_with_hash(self, file_path: str, doc_hash: str, chunk_count: int) -> None:
        """Update metadata with hash (backward compatibility)"""
        self.metadata_handler.update_metadata_with_hash(file_path, doc_hash, chunk_count)
