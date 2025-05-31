"""Modern File Operations package for WorkApp2.

This package provides modern, async-capable file operations using pathlib.Path
and aiofiles, replacing deprecated os.path patterns.
"""

__version__ = "0.1.0"

from .async_file_ops import AsyncFileHandler
from .path_utils import PathUtils
from .json_ops import JsonHandler

__all__ = ["AsyncFileHandler", "PathUtils", "JsonHandler"]
