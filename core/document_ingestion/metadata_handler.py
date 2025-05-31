"""Metadata handling functionality for document ingestion"""

import os
import time
import json
import logging
import hashlib
import threading
import fcntl
import errno
from typing import Dict, Any, Optional

from core.config import retrieval_config, resolve_path
from utils.common.error_handler import CommonErrorHandler, with_error_context
from utils.error_handling.decorators import with_retry
from utils.logging.error_logging import log_error

logger = logging.getLogger(__name__)


class MetadataHandler:
    """Handles document metadata and hash management"""
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize the metadata handler
        
        Args:
            index_path: Path to index directory (None for config default)
        """
        self.index_path = index_path or retrieval_config.index_path
        logger.info(f"Metadata handler initialized for path: {self.index_path}")
    
    @with_error_context("document hash computation")
    def compute_document_hash(self, file_path: str) -> str:
        """
        Compute a hash for a document file
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Hash string for the document
        """
        try:
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"Permission denied: {file_path}")
            
            # Use a buffer to handle large files efficiently
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read and update hash in chunks of 4K
                for byte_block in iter(lambda: f.read(4096), b""):
                    hasher.update(byte_block)
            
            # Get the hexadecimal digest
            hash_value = hasher.hexdigest()
            logger.debug(f"Computed hash for {os.path.basename(file_path)}: {hash_value}")
            return hash_value
            
        except (FileNotFoundError, PermissionError) as e:
            # Re-raise these specific exceptions
            logger.error(f"Error accessing file for hashing: {str(e)}")
            raise
        except Exception as e:
            logger.warning(f"Error computing hash for {file_path}: {str(e)}")
            # Return a fallback hash based on filename and modification time
            return self._compute_fallback_hash(file_path)
    
    def _compute_fallback_hash(self, file_path: str) -> str:
        """
        Compute a fallback hash based on filename and modification time
        when the standard hashing method fails
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Fallback hash string
        """
        try:
            # Get file metadata
            mtime = os.path.getmtime(file_path)
            file_size = os.path.getsize(file_path)
            
            # Create a string with file info
            fallback = f"{file_path}:{mtime}:{file_size}"
            
            # Hash the string
            hash_value = hashlib.md5(fallback.encode()).hexdigest()
            logger.info(f"Using fallback hash for {os.path.basename(file_path)}: {hash_value}")
            return hash_value
        except Exception as e:
            # Last resort fallback
            logger.error(f"Error computing fallback hash: {str(e)}")
            return hashlib.md5(file_path.encode()).hexdigest()
    
    @with_retry(max_attempts=3, backoff_factor=1.0, exception_types=(IOError, OSError))
    def update_metadata_with_hash(self, file_path: str, doc_hash: str, chunk_count: int) -> None:
        """
        Update metadata.json with document hash information
        
        Args:
            file_path: Path to the document file
            doc_hash: Hash of the document
            chunk_count: Number of chunks extracted from the document
        """
        # Create paths for metadata files
        metadata_dir = resolve_path(self.index_path, create_dir=True)
        metadata_path = os.path.join(metadata_dir, "metadata.json")
        temp_path = os.path.join(metadata_dir, f"metadata.json.tmp.{threading.get_ident()}")
        lock_path = os.path.join(metadata_dir, "metadata.lock")
        
        # Log resolved paths for debugging
        logger.debug(f"Resolved metadata directory: {metadata_dir}")
        logger.debug(f"Metadata file path: {metadata_path}")
        
        # Create metadata directory if it doesn't exist
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Use proper file locking to prevent race conditions
        metadata_file = None
        lock_acquired = False
        
        try:
            # Create a lock file to indicate an update is in progress (additional safety)
            with open(lock_path, "w") as lock_file:
                lock_file.write(
                    f"Update in progress: {time.time()} by thread {threading.get_ident()}"
                )
            
            # Open or create the metadata file with explicit mode for better compatibility
            metadata_file = open(metadata_path, "a+")
            
            # Acquire an exclusive lock on the file with timeout
            lock_acquired = self._acquire_file_lock(metadata_file)
            
            # Read and update metadata
            metadata = self._read_existing_metadata(metadata_file)
            self._update_metadata_content(metadata, file_path, doc_hash, chunk_count)
            
            # Write updated metadata atomically
            self._write_metadata_atomically(metadata, temp_path, metadata_path)
            
            logger.info(f"Updated metadata for {os.path.basename(file_path)} with hash {doc_hash}")
        
        except json.JSONDecodeError as e:
            error_msg = f"JSON decode error in metadata file: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg, include_traceback=True)
            raise
        
        except (IOError, OSError) as e:
            error_msg = f"Error accessing metadata file: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg, include_traceback=True)
            raise
        
        except Exception as e:
            error_msg = f"Unexpected error updating metadata with hash: {str(e)}"
            logger.error(error_msg)
            log_error(error_msg, include_traceback=True)
            raise
        
        finally:
            self._cleanup_metadata_resources(metadata_file, lock_acquired, temp_path, lock_path)
    
    def _acquire_file_lock(self, metadata_file) -> bool:
        """Acquire file lock with timeout"""
        start_time = time.time()
        max_wait_time = 10.0  # Maximum time to wait for lock in seconds
        
        while time.time() - start_time < max_wait_time:
            try:
                # Try non-blocking lock first
                fcntl.flock(metadata_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except IOError as e:
                # If lock is held by another process, wait a bit and retry
                if e.errno == errno.EAGAIN or e.errno == errno.EACCES:
                    logger.debug(
                        f"Waiting for metadata file lock (thread {threading.get_ident()})"
                    )
                    time.sleep(0.1)
                else:
                    # Re-raise for other IOErrors
                    raise
        
        # If we couldn't acquire the lock after waiting, use blocking lock as last resort
        logger.warning(
            f"Using blocking lock for metadata file after waiting {max_wait_time} seconds"
        )
        fcntl.flock(metadata_file.fileno(), fcntl.LOCK_EX)
        return True
    
    def _read_existing_metadata(self, metadata_file) -> Dict[str, Any]:
        """Read existing metadata from file"""
        # Seek to the beginning of the file to read its content
        metadata_file.seek(0)
        content = metadata_file.read()
        
        # Parse existing metadata or create new if empty/invalid
        if content.strip():
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted metadata file: {str(e)}")
                # Backup the corrupted file for debugging
                self._backup_corrupted_metadata(content)
                # Start with fresh metadata
                return {}
        else:
            return {}
    
    def _backup_corrupted_metadata(self, content: str) -> None:
        """Backup corrupted metadata file"""
        try:
            metadata_dir = resolve_path(self.index_path, create_dir=True)
            backup_path = os.path.join(metadata_dir, f"metadata.json.corrupted.{int(time.time())}")
            with open(backup_path, "w") as backup_file:
                backup_file.write(content)
            logger.info(f"Backed up corrupted metadata to {backup_path}")
        except Exception as backup_error:
            logger.warning(f"Failed to backup corrupted metadata: {str(backup_error)}")
    
    def _update_metadata_content(self, metadata: Dict[str, Any], file_path: str, 
                               doc_hash: str, chunk_count: int) -> None:
        """Update metadata dictionary with new document information"""
        # Initialize document_hashes if not present
        if "document_hashes" not in metadata:
            metadata["document_hashes"] = {}
        
        # Add or update document hash with enhanced metadata
        metadata["document_hashes"][file_path] = {
            "hash": doc_hash,
            "last_processed": time.time(),
            "chunk_count": chunk_count,
            "file_size": os.path.getsize(file_path),
            "file_name": os.path.basename(file_path),
            "file_extension": os.path.splitext(file_path)[1].lower(),
            "last_modified": os.path.getmtime(file_path),
            "processing_timestamp": time.time(),
            "updated_by": f"thread-{threading.get_ident()}",
        }
    
    def _write_metadata_atomically(self, metadata: Dict[str, Any], temp_path: str, 
                                 metadata_path: str) -> None:
        """Write metadata to file atomically"""
        # Write to a thread-specific temporary file first
        with open(temp_path, "w") as temp_file:
            json.dump(metadata, temp_file, indent=2)
            temp_file.flush()
            # Ensure the temporary file is fully written to disk
            os.fsync(temp_file.fileno())
        
        # Use atomic replace to update the original file
        # This is done while still holding the lock on the original file
        os.replace(temp_path, metadata_path)
    
    def _cleanup_metadata_resources(self, metadata_file, lock_acquired: bool, 
                                  temp_path: str, lock_path: str) -> None:
        """Clean up metadata file resources"""
        # Clean up resources
        if metadata_file is not None:
            if lock_acquired:
                try:
                    # Release the lock explicitly
                    fcntl.flock(metadata_file.fileno(), fcntl.LOCK_UN)
                    logger.debug(
                        f"Released metadata file lock (thread {threading.get_ident()})"
                    )
                except Exception as unlock_error:
                    logger.warning(f"Error releasing file lock: {str(unlock_error)}")
            
            # Close the file
            try:
                metadata_file.close()
            except Exception as close_error:
                logger.warning(f"Error closing metadata file: {str(close_error)}")
        
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp file: {str(cleanup_error)}")
        
        # Remove lock file
        if os.path.exists(lock_path):
            try:
                os.remove(lock_path)
            except Exception as lock_cleanup_error:
                logger.warning(f"Failed to remove lock file: {str(lock_cleanup_error)}")
    
    def get_document_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific document
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document metadata or None if not found
        """
        try:
            metadata_dir = resolve_path(self.index_path, create_dir=False)
            metadata_path = os.path.join(metadata_dir, "metadata.json")
            
            if not os.path.exists(metadata_path):
                return None
            
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                return metadata.get("document_hashes", {}).get(file_path)
        
        except Exception as e:
            error_msg = CommonErrorHandler.handle_processing_error(
                "MetadataHandler", "metadata retrieval", e
            )
            return None
    
    def get_all_metadata(self) -> Dict[str, Any]:
        """
        Get all metadata
        
        Returns:
            All metadata or empty dict if not found
        """
        try:
            metadata_dir = resolve_path(self.index_path, create_dir=False)
            metadata_path = os.path.join(metadata_dir, "metadata.json")
            
            if not os.path.exists(metadata_path):
                return {}
            
            with open(metadata_path, "r") as f:
                return json.load(f)
        
        except Exception as e:
            error_msg = CommonErrorHandler.handle_processing_error(
                "MetadataHandler", "all metadata retrieval", e
            )
            return {}
    
    def is_document_processed(self, file_path: str) -> bool:
        """
        Check if a document has been processed
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if document has been processed, False otherwise
        """
        try:
            doc_metadata = self.get_document_metadata(file_path)
            if not doc_metadata:
                return False
            
            # Check if file has been modified since last processing
            current_mtime = os.path.getmtime(file_path)
            last_modified = doc_metadata.get("last_modified", 0)
            
            # If file was modified after last processing, it needs reprocessing
            return current_mtime <= last_modified
        
        except Exception as e:
            logger.warning(f"Error checking document processing status: {str(e)}")
            return False
    
    def clear_metadata(self) -> None:
        """Clear all metadata"""
        try:
            metadata_dir = resolve_path(self.index_path, create_dir=False)
            metadata_path = os.path.join(metadata_dir, "metadata.json")
            
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                logger.info("Metadata cleared successfully")
        
        except Exception as e:
            error_msg = CommonErrorHandler.handle_processing_error(
                "MetadataHandler", "metadata clearing", e
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics from metadata
        
        Returns:
            Dictionary with processing statistics
        """
        try:
            metadata = self.get_all_metadata()
            document_hashes = metadata.get("document_hashes", {})
            
            if not document_hashes:
                return {"total_documents": 0, "total_chunks": 0}
            
            total_chunks = sum(doc.get("chunk_count", 0) for doc in document_hashes.values())
            total_size = sum(doc.get("file_size", 0) for doc in document_hashes.values())
            
            # Get file type distribution
            extensions = {}
            for doc in document_hashes.values():
                ext = doc.get("file_extension", "unknown")
                extensions[ext] = extensions.get(ext, 0) + 1
            
            return {
                "total_documents": len(document_hashes),
                "total_chunks": total_chunks,
                "total_size_bytes": total_size,
                "file_types": extensions,
                "avg_chunks_per_doc": total_chunks / len(document_hashes) if document_hashes else 0,
                "avg_size_per_doc": total_size / len(document_hashes) if document_hashes else 0,
            }
        
        except Exception as e:
            error_msg = CommonErrorHandler.handle_processing_error(
                "MetadataHandler", "processing stats", e
            )
            return {"error": error_msg}
