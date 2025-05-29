# Index freshness utilities
import os
import time
import logging
from typing import Optional, Tuple

from utils.config import app_config, retrieval_config, resolve_path
from utils.index_management.index_operations import get_index_modified_time

# Setup logging
logger = logging.getLogger(__name__)

# Global variable to track last check time
_last_check_time = 0
_check_interval = 5  # seconds


def is_index_fresh(
    index_path: Optional[str] = None, last_known_mtime: float = 0
) -> Tuple[bool, float]:
    """
    Check if the index is fresh (hasn't been modified since last check)

    Args:
        index_path: Path to the index directory, defaults to app_config.index_path
        last_known_mtime: Last known modification time of the index

    Returns:
        Tuple of (is_fresh, current_mtime)
    """
    global _last_check_time

    # Throttle checks to avoid excessive file system operations
    current_time = time.time()
    if current_time - _last_check_time < _check_interval:
        # Skip the check if we've checked recently
        return True, last_known_mtime

    _last_check_time = current_time

    if index_path is None:
        index_path = retrieval_config.index_path

    # Resolve the path
    resolved_index_path = resolve_path(index_path)

    current_mtime = get_index_modified_time(index_path)

    # If last_known_mtime is 0, we haven't loaded the index yet
    if last_known_mtime == 0:
        return False, current_mtime

    # Check if the index has been modified since we last loaded it
    is_fresh = current_mtime <= last_known_mtime

    if not is_fresh:
        logger.info(
            f"Index has been modified since last load (mtime: {current_mtime} vs {last_known_mtime})"
        )

    return is_fresh, current_mtime
