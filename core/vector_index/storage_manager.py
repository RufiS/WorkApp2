"""Index Storage Manager

Handles atomic save/load operations for FAISS indices with metadata.
Extracted from core/vector_index_engine.py
"""

import os
import time
import json
import logging
from typing import Optional, List, Dict, Any

import numpy as np
import faiss

from core.config import resolve_path
from core.index_management.gpu_manager import gpu_manager
from utils.logging.error_logging import log_error

# Setup logging
logger = logging.getLogger(__name__)


class StorageManager:
    """Handles atomic storage operations for FAISS indices"""

    def __init__(self, embedding_model_name: str, embedding_dim: int):
        """
        Initialize storage manager

        Args:
            embedding_model_name: Name of the embedding model
            embedding_dim: Dimension of embeddings
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = embedding_dim
        logger.info(f"Storage manager initialized for model {embedding_model_name}, dim {embedding_dim}")

    def save_index(
        self,
        index: faiss.Index,
        texts: List[Any],
        index_dir: Optional[str] = None,
        dry_run: bool = False
    ) -> None:
        """
        Save FAISS index and texts to disk using atomic operations

        Args:
            index: FAISS index to save
            texts: List of text chunks
            index_dir: Directory to save to (defaults to config value)
            dry_run: If True, skip actual saving (preview only)

        Raises:
            ValueError: If no index provided
            IOError: If there's an I/O error during file writing
            OSError: If there's an OS error during atomic operations
        """
        if index is None or not texts:
            raise ValueError("No index or texts provided for saving")

        # Resolve directory path
        index_dir = resolve_path(index_dir or "data/index", create_dir=True)

        # Handle dry run
        if dry_run:
            logger.info(f"[DRY RUN] Would save index with {len(texts)} chunks to {index_dir}")
            return

        logger.info(f"Saving index with {len(texts)} chunks to {index_dir}")

        # Create directory if needed
        os.makedirs(index_dir, exist_ok=True)

        # Create lock file to indicate update in progress
        lock_path = os.path.join(index_dir, "index.lock")
        with open(lock_path, "w") as lock_file:
            lock_file.write(f"Update in progress: {time.time()}")

        try:
            # Define file paths
            paths = self._get_file_paths(index_dir)

            # Convert GPU index to CPU for saving
            cpu_index = self._ensure_cpu_index(index)

            # Save components atomically
            self._save_index_file(cpu_index, paths["temp_index"], paths["index"])
            self._save_texts_file(texts, paths["temp_texts"], paths["texts"])
            self._save_metadata_file(index, paths["temp_metadata"], paths["metadata"])

            logger.info(f"Successfully saved index with {len(texts)} chunks to {index_dir}")

        except Exception as e:
            error_msg = f"Error during index save operation: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg, include_traceback=True)
            raise
        finally:
            # Remove lock file
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except Exception as e:
                    logger.warning(f"Failed to remove lock file: {str(e)}")

    def load_index(self, index_dir: Optional[str] = None) -> tuple[faiss.Index, List[Any]]:
        """
        Load FAISS index and texts from disk

        Args:
            index_dir: Directory to load from (defaults to config value)

        Returns:
            Tuple of (FAISS index, list of texts)

        Raises:
            FileNotFoundError: If index files are not found
            ValueError: If index parameters don't match
        """
        # Resolve directory path
        index_dir = resolve_path(index_dir or "data/index", create_dir=False)

        # Get file paths
        paths = self._get_file_paths(index_dir)

        # Check required files exist
        self._verify_files_exist(paths)

        # Load metadata first
        metadata = self._load_metadata(paths["metadata"])

        # Load index with dimension checking
        index = self._load_index_file(paths["index"], metadata)

        # Load texts
        texts = self._load_texts_file(paths["texts"])

        # Move to GPU if configured
        if gpu_manager.should_use_gpu() and index is not None:
            index, success = gpu_manager.move_index_to_gpu(index)
            if success:
                logger.info("Loaded index moved to GPU successfully")
            else:
                logger.warning("Failed to move loaded index to GPU")
        elif gpu_manager.should_use_gpu() and index is None:
            logger.warning("Cannot move None index to GPU - index loading may have failed")

        logger.info(f"Successfully loaded index with {len(texts)} chunks from {index_dir}")
        return index, texts

    def index_exists(self, index_dir: Optional[str] = None) -> bool:
        """
        Check if index files exist on disk

        Args:
            index_dir: Directory to check (defaults to config value)

        Returns:
            True if index files exist, False otherwise
        """
        index_dir = resolve_path(index_dir or "data/index", create_dir=False)
        paths = self._get_file_paths(index_dir)

        exists = os.path.exists(paths["index"]) and os.path.exists(paths["texts"])

        if exists:
            logger.debug(f"Index files found at {index_dir}")
        else:
            logger.debug(f"Index files not found at {index_dir}")

        return exists

    def clear_index(self, index_dir: Optional[str] = None) -> None:
        """
        Clear index files from disk

        Args:
            index_dir: Directory to clear (defaults to config value)
        """
        index_dir = resolve_path(index_dir or "data/index", create_dir=True)
        paths = self._get_file_paths(index_dir)

        # Create directory if it doesn't exist
        os.makedirs(index_dir, exist_ok=True)

        # Remove files if they exist
        for file_path in [paths["index"], paths["texts"], paths["metadata"]]:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed index file: {file_path}")

        logger.info(f"Index files cleared from {index_dir}")

    def _get_file_paths(self, index_dir: str) -> Dict[str, str]:
        """Get file paths for index components - FIXED to match cache system expectations"""
        return {
            "index": os.path.join(index_dir, "faiss_index.bin"),  # Match cache system
            "temp_index": os.path.join(index_dir, "faiss_index.bin.tmp"),
            "texts": os.path.join(index_dir, "chunks.json"),  # Match cache system  
            "temp_texts": os.path.join(index_dir, "chunks.json.tmp"),
            "metadata": os.path.join(index_dir, "metadata.json"),
            "temp_metadata": os.path.join(index_dir, "metadata.json.tmp"),
        }

    def _ensure_cpu_index(self, index: faiss.Index) -> faiss.Index:
        """Ensure index is on CPU for saving"""
        # Check if index is on GPU before converting
        if hasattr(index, "getDevice") and index.getDevice() >= 0:
            cpu_index, success = gpu_manager.move_index_to_cpu(index)
            if success:
                return cpu_index
            else:
                logger.warning("Failed to move index to CPU for saving, using original")
                return index
        return index

    def _save_index_file(self, index: faiss.Index, temp_path: str, final_path: str) -> None:
        """Save FAISS index with atomic operation"""
        try:
            faiss.write_index(index, temp_path)
            os.replace(temp_path, final_path)
            logger.debug(f"Index saved to {final_path}")
        except IOError as e:
            raise IOError(f"Failed to save index file: {str(e)}") from e

    def _save_texts_file(self, texts: List[Any], temp_path: str, final_path: str) -> None:
        """Save texts with atomic operation - FIXED to handle both numpy and JSON formats"""
        try:
            # Validate texts before saving
            if not texts or len(texts) == 0:
                logger.error("Attempting to save empty texts list!")
                raise ValueError("Cannot save empty texts list")

            # Ensure directory exists
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)

            # CRITICAL FIX: Determine format based on file extension
            if final_path.endswith('.json'):
                # Save as JSON format for chunks.json (cache system expects this)
                logger.debug(f"Saving {len(texts)} chunks as JSON to {temp_path}")
                
                # Preserve full chunk dictionaries for JSON format  
                chunk_data = []
                for item in texts:
                    if isinstance(item, dict):
                        chunk_data.append(item)  # Keep full dictionary
                    elif isinstance(item, str):
                        chunk_data.append({"text": item})  # Wrap string in dict
                    else:
                        chunk_data.append({"text": str(item)})  # Convert to string and wrap
                
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)
                    
            else:
                # Save as numpy format for legacy compatibility
                logger.debug(f"Saving {len(texts)} texts as numpy to {temp_path}")
                
                # Extract text content for numpy format
                text_content = []
                for item in texts:
                    if isinstance(item, dict) and 'text' in item:
                        text_content.append(item['text'])
                    elif isinstance(item, str):
                        text_content.append(item)
                    else:
                        text_content.append(str(item))

                text_array = np.array(text_content, dtype=object)
                with open(temp_path, 'wb') as f:
                    np.save(f, text_array)

            # Verify temp file was created and has content
            if not os.path.exists(temp_path):
                raise IOError(f"Temp file was not created: {temp_path}")

            file_size = os.path.getsize(temp_path)
            if file_size == 0:
                raise IOError(f"Temp file is empty: {temp_path}")

            logger.debug(f"Successfully created temp file: {temp_path} ({file_size} bytes)")

            # Atomic move
            os.replace(temp_path, final_path)
            logger.debug(f"Texts saved to {final_path}")

        except Exception as e:
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise IOError(f"Failed to save texts file: {str(e)}") from e

    def _save_metadata_file(self, index: faiss.Index, temp_path: str, final_path: str) -> None:
        """Save metadata with atomic operation"""
        try:
            metadata = {
                "embedding_model": self.embedding_model_name,
                "embedding_dim": self.embedding_dim,
                "last_updated": time.time(),
                "index_size": getattr(index, "ntotal", 0),
                "index_type": type(index).__name__,
            }

            with open(temp_path, "w") as f:
                json.dump(metadata, f, indent=2)
            os.replace(temp_path, final_path)
            logger.debug(f"Metadata saved to {final_path}")
        except IOError as e:
            raise IOError(f"Failed to save metadata file: {str(e)}") from e

    def _verify_files_exist(self, paths: Dict[str, str]) -> None:
        """Verify required files exist"""
        if not os.path.exists(paths["index"]):
            raise FileNotFoundError(f"Index file not found at {paths['index']}")
        if not os.path.exists(paths["texts"]):
            raise FileNotFoundError(f"Texts file not found at {paths['texts']}")

    def _load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """Load metadata if available"""
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                logger.debug(f"Loaded metadata from {metadata_path}")
            except Exception as e:
                logger.warning(f"Error reading metadata: {str(e)}")
        return metadata

    def _load_index_file(self, index_path: str, metadata: Dict[str, Any]) -> faiss.Index:
        """Load FAISS index with dimension validation"""
        try:
            index = faiss.read_index(index_path)
            logger.debug(f"Successfully loaded FAISS index from {index_path}")

            # Verify dimensions match
            if hasattr(index, "d") and index.d != self.embedding_dim:
                logger.warning(
                    f"Index dimension ({index.d}) doesn't match embedding model dimension ({self.embedding_dim})"
                )
                # Create new index with correct dimensions
                logger.info(f"Creating new index with dimension {self.embedding_dim}")
                return faiss.IndexFlatL2(self.embedding_dim)

            return index

        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            # Create new index on load error
            logger.info(f"Creating new index with dimension {self.embedding_dim} due to load error")
            return faiss.IndexFlatL2(self.embedding_dim)

    def _load_texts_file(self, texts_path: str) -> List[Any]:
        """Load texts array - FIXED to handle both numpy and JSON formats"""
        try:
            # CRITICAL FIX: Determine format based on file extension
            if texts_path.endswith('.json'):
                # Load JSON format (chunks.json from cache system)
                logger.debug(f"Loading chunks as JSON from {texts_path}")
                with open(texts_path, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                # Ensure we have a list
                if not isinstance(chunk_data, list):
                    logger.warning(f"JSON file contains non-list data: {type(chunk_data)}")
                    return []
                
                logger.debug(f"Loaded {len(chunk_data)} chunks from JSON file")
                return chunk_data
                
            else:
                # Load numpy format (legacy compatibility)
                logger.debug(f"Loading texts as numpy from {texts_path}")
                texts_array = np.load(texts_path, allow_pickle=True)
                if texts_array is not None and hasattr(texts_array, "tolist"):
                    texts = texts_array.tolist()
                    if texts is None:
                        texts = []
                    logger.debug(f"Loaded {len(texts)} texts from numpy file")
                    return texts
                else:
                    logger.warning(f"Invalid texts array loaded from {texts_path}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error loading texts from {texts_path}: {str(e)}")
            raise ValueError(f"Failed to load texts: {str(e)}")

    def get_storage_info(self, index_dir: Optional[str] = None) -> Dict[str, Any]:
        """Get information about stored index"""
        index_dir = resolve_path(index_dir or "data/index", create_dir=False)
        paths = self._get_file_paths(index_dir)

        info = {
            "index_directory": index_dir,
            "index_exists": self.index_exists(index_dir),
            "files": {}
        }

        # Get file information
        for name, path in paths.items():
            if name.startswith("temp_"):
                continue

            if os.path.exists(path):
                stat = os.stat(path)
                info["files"][name] = {
                    "exists": True,
                    "size_bytes": stat.st_size,
                    "modified_time": stat.st_mtime,
                }
            else:
                info["files"][name] = {"exists": False}

        # Load metadata if available
        if os.path.exists(paths["metadata"]):
            info["metadata"] = self._load_metadata(paths["metadata"])

        return info
