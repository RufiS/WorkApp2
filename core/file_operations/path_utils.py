"""Modern Path Utilities for WorkApp2.

Replaces os.path usage with pathlib.Path for cross-platform compatibility.
"""

import logging
from pathlib import Path
from typing import Union, Optional, List
import os

logger = logging.getLogger(__name__)


class PathUtils:
    """Modern path utilities using pathlib.Path."""
    
    @staticmethod
    def ensure_path(path_input: Union[str, Path]) -> Path:
        """Convert string path to Path object.
        
        Args:
            path_input: String path or Path object
            
        Returns:
            Path object
        """
        return Path(path_input) if isinstance(path_input, str) else path_input
    
    @staticmethod
    def resolve_path(path_input: Union[str, Path], create_dir: bool = False) -> Path:
        """Resolve a path and optionally create parent directories.
        
        Args:
            path_input: String path or Path object
            create_dir: Whether to create parent directories
            
        Returns:
            Resolved Path object
        """
        path = PathUtils.ensure_path(path_input).resolve()
        
        if create_dir:
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path.parent}")
            
        return path
    
    @staticmethod
    def safe_file_write(file_path: Union[str, Path], content: str, encoding: str = "utf-8") -> bool:
        """Safely write content to a file with proper error handling.
        
        Args:
            file_path: Path to the file
            content: Content to write
            encoding: File encoding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            path = PathUtils.resolve_path(file_path, create_dir=True)
            path.write_text(content, encoding=encoding)
            logger.debug(f"Successfully wrote to file: {path}")
            return True
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {str(e)}")
            return False
    
    @staticmethod
    def safe_file_read(file_path: Union[str, Path], encoding: str = "utf-8") -> Optional[str]:
        """Safely read content from a file.
        
        Args:
            file_path: Path to the file
            encoding: File encoding
            
        Returns:
            File content if successful, None otherwise
        """
        try:
            path = PathUtils.ensure_path(file_path)
            if not path.exists():
                logger.warning(f"File does not exist: {path}")
                return None
                
            content = path.read_text(encoding=encoding)
            logger.debug(f"Successfully read file: {path}")
            return content
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
        """Get file size in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes, None if error
        """
        try:
            path = PathUtils.ensure_path(file_path)
            return path.stat().st_size if path.exists() else None
        except Exception as e:
            logger.error(f"Error getting file size for {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def list_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = False) -> List[Path]:
        """List files in a directory with optional pattern matching.
        
        Args:
            directory: Directory to search
            pattern: File pattern (e.g., "*.py", "*.txt")
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        try:
            dir_path = PathUtils.ensure_path(directory)
            if not dir_path.is_dir():
                logger.warning(f"Directory does not exist: {dir_path}")
                return []
            
            if recursive:
                files = list(dir_path.rglob(pattern))
            else:
                files = list(dir_path.glob(pattern))
            
            # Filter to only return files (not directories)
            files = [f for f in files if f.is_file()]
            
            logger.debug(f"Found {len(files)} files matching pattern '{pattern}' in {dir_path}")
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {str(e)}")
            return []
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> bool:
        """Ensure a directory exists, creating it if necessary.
        
        Args:
            directory: Directory path
            
        Returns:
            True if directory exists or was created successfully
        """
        try:
            dir_path = PathUtils.ensure_path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {str(e)}")
            return False
    
    @staticmethod
    def clean_filename(filename: str) -> str:
        """Clean a filename for cross-platform compatibility.
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned filename safe for filesystem use
        """
        # Remove or replace problematic characters
        invalid_chars = '<>:"/\\|?*'
        cleaned = filename
        
        for char in invalid_chars:
            cleaned = cleaned.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        cleaned = cleaned.strip('. ')
        
        # Limit length to avoid filesystem issues
        if len(cleaned) > 255:
            name, ext = os.path.splitext(cleaned)
            max_name_len = 255 - len(ext)
            cleaned = name[:max_name_len] + ext
        
        return cleaned or "unnamed_file"
    
    @staticmethod
    def get_relative_path(file_path: Union[str, Path], base_path: Union[str, Path]) -> Path:
        """Get relative path from base path.
        
        Args:
            file_path: Target file path
            base_path: Base directory path
            
        Returns:
            Relative path
        """
        try:
            file_p = PathUtils.ensure_path(file_path).resolve()
            base_p = PathUtils.ensure_path(base_path).resolve()
            return file_p.relative_to(base_p)
        except ValueError:
            # Paths are not relative to each other
            return PathUtils.ensure_path(file_path)
