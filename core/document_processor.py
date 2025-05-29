"""Document Processor Facade

Facade combining document ingestion and index management.
Provides a unified interface for document processing operations.
"""

import logging
from typing import List, Dict, Tuple, Any, Optional, Union

import faiss
import numpy as np

from core.document_ingestion import DocumentIngestion
from core.vector_index_engine import IndexManager
from utils.config import retrieval_config
from utils.error_logging import log_error, log_warning

# Setup logging
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Unified document processor combining ingestion and indexing capabilities"""

    def __init__(self, embedding_model_name: str = None):
        """
        Initialize the document processor facade

        Args:
            embedding_model_name: Name of the embedding model to use (defaults to config value)
        """
        self.embedding_model_name = embedding_model_name or retrieval_config.embedding_model

        # Initialize the two main components
        self.ingestion = DocumentIngestion(embedding_model_name)
        self.index_manager = IndexManager(embedding_model_name)

        # Sync some properties for backward compatibility
        self.embedding_dim = self.index_manager.embedding_dim
        self.gpu_available = self.index_manager.gpu_available

        logger.info(f"Document processor facade initialized with model {self.embedding_model_name}")

    @property
    def index(self):
        """Get the current FAISS index"""
        return self.index_manager.index

    @property
    def texts(self):
        """Get the current texts/chunks"""
        return self.index_manager.texts

    @property
    def chunks(self):
        """Get the current chunks (alias for texts)"""
        return self.index_manager.chunks

    @property
    def processed_files(self):
        """Get the set of processed files"""
        return self.ingestion.processed_files

    def has_index(self, index_dir: Optional[str] = None) -> bool:
        """
        Check if an index exists either in memory or on disk

        Args:
            index_dir: Directory to check for index files (defaults to config value)

        Returns:
            True if an index exists, False otherwise
        """
        return self.index_manager.has_index(index_dir)

    def create_empty_index(self) -> None:
        """Create a new empty FAISS index"""
        self.index_manager.create_empty_index()

    def process_file(self, file) -> List[Dict[str, Any]]:
        """
        Process a file object (from streamlit file uploader)

        Args:
            file: File object from streamlit file uploader

        Returns:
            List of document chunks with metadata
        """
        return self.ingestion.process_file(file)

    def load_and_chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a document and split it into chunks with caching

        Args:
            file_path: Path to the document file

        Returns:
            List of document chunks with metadata
        """
        return self.ingestion.load_and_chunk_document(file_path)

    def process_documents(
        self, file_paths: List[str], dry_run: bool = False
    ) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """
        Process multiple documents and build a search index

        Args:
            file_paths: List of paths to document files
            dry_run: If True, preview changes without writing to disk

        Returns:
            Tuple of (FAISS index, list of chunks)
        """
        # Use ingestion to process documents and extract chunks
        all_chunks = self.ingestion.process_documents(file_paths, dry_run)

        if not dry_run and all_chunks:
            # Use index manager to create index from chunks
            index, chunks = self.index_manager.create_index_from_chunks(all_chunks)
            return index, chunks
        else:
            # For dry run or empty chunks, return what we would create
            if dry_run and all_chunks:
                # Create a temporary index for preview
                temp_index_manager = IndexManager(self.embedding_model_name)
                temp_index, temp_chunks = temp_index_manager.create_index_from_chunks(all_chunks)
                return temp_index, temp_chunks
            else:
                return None, all_chunks

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using the query

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            List of relevant chunks with scores
        """
        return self.index_manager.search(query, top_k)

    def clear_index(self, index_dir: Optional[str] = None) -> None:
        """
        Clear the index and all cached data

        Args:
            index_dir: Directory containing the index files to clear (defaults to config value)
        """
        self.index_manager.clear_index(index_dir)
        # Also clear ingestion cache and processed files
        self.ingestion.clear_cache()  # Use the proper method instead of direct assignment
        self.ingestion.processed_files = set()

    def save_index(self, index_dir: Optional[str] = None, dry_run: bool = False) -> None:
        """
        Save the FAISS index and chunks to disk using atomic file operations

        Args:
            index_dir: Directory to save the index and chunks (defaults to config value)
            dry_run: If True, skip saving to disk (preview only)
        """
        self.index_manager.save_index(index_dir, dry_run)

    def load_index(self, index_dir: Optional[str] = None) -> None:
        """
        Load a FAISS index and chunks from disk

        Args:
            index_dir: Directory containing the index and chunks (defaults to config value)
        """
        self.index_manager.load_index(index_dir)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string

        Args:
            query: Query string to embed

        Returns:
            NumPy array of query embedding
        """
        return self.index_manager.embed_query(query)

    def batch_embed_chunks(self, chunks, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Embed document chunks in batches

        Args:
            chunks: List of document chunks (either dictionaries with 'text' key or strings)
            batch_size: Size of batches for embedding (None for default)

        Returns:
            NumPy array of embeddings
        """
        return self.index_manager.batch_embed_chunks(chunks, batch_size)

    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index from embeddings

        Args:
            embeddings: NumPy array of embeddings

        Returns:
            FAISS index
        """
        return self.index_manager.build_index(embeddings)

    def create_index_from_chunks(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """
        Create a FAISS index from document chunks

        Args:
            chunks: List of document chunks

        Returns:
            Tuple of (FAISS index, list of chunks)
        """
        return self.index_manager.create_index_from_chunks(chunks)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive processor metrics

        Returns:
            Dictionary with processor metrics from both ingestion and index manager
        """
        ingestion_metrics = self.ingestion.get_metrics()
        index_metrics = self.index_manager.get_metrics()

        # Combine metrics with prefixes to avoid conflicts
        combined_metrics = {}

        # Add ingestion metrics with prefix
        for key, value in ingestion_metrics.items():
            combined_metrics[f"ingestion_{key}"] = value

        # Add index metrics with prefix
        for key, value in index_metrics.items():
            combined_metrics[f"index_{key}"] = value

        # Add some combined metrics
        combined_metrics["total_documents"] = ingestion_metrics.get("total_documents", 0)
        combined_metrics["total_chunks"] = max(
            ingestion_metrics.get("total_chunks", 0), index_metrics.get("total_chunks", 0)
        )
        combined_metrics["embedding_model"] = self.embedding_model_name
        combined_metrics["embedding_dim"] = self.embedding_dim
        combined_metrics["gpu_available"] = self.gpu_available

        return combined_metrics

    # Backward compatibility methods
    def create_embeddings(
        self, texts: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Create embeddings for a list of text chunks (backward compatibility)

        Args:
            texts: List of text chunks (dictionaries with 'text' key)

        Returns:
            Tuple of (list of text chunks, numpy array of embeddings)
        """
        embeddings = self.batch_embed_chunks(texts)
        return texts, embeddings

    def _build_faiss_index(self, texts: List[str]) -> Tuple[faiss.Index, List[str]]:
        """
        Build a FAISS index from text chunks (backward compatibility)

        Args:
            texts: List of text chunks

        Returns:
            Tuple of (FAISS index, list of texts)
        """
        # Convert strings to chunk format
        chunks = [{"text": text} for text in texts]
        embeddings = self.batch_embed_chunks(chunks)
        index = self.build_index(embeddings)
        return index, texts
