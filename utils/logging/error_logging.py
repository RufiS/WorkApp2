# Error and metrics logging module
import os
import time
import json
import logging
import traceback
import threading
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Create a lock for thread-safe file access
log_lock = threading.RLock()

"""
WorkApp2 Logging Standards

This section documents the standardized logging approach for WorkApp2.
It provides examples and best practices for consistent error logging across the codebase.
...
"""


def _log_to_file(log_entry: Dict[str, Any], log_path: str) -> None:
    """
    Internal function to write a log entry to a file with proper error handling
    """
    try:
        from utils.config import resolve_path

        resolved_log_path = resolve_path(log_path, create_dir=True)
    except ImportError:
        resolved_log_path = log_path
        os.makedirs(os.path.dirname(resolved_log_path), exist_ok=True)

    with log_lock:
        try:
            with open(resolved_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            import tempfile

            fallback_path = os.path.join(tempfile.gettempdir(), "workapp2_errors.log")
            with open(fallback_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                f.write(f"Meta-error: Failed to write to primary log: {str(e)}\n")


def log_error(
    error_message: str,
    include_traceback: bool = False,
    error_type: str = "ERROR",
    source: Optional[str] = None,
    additional_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an error to the central error log file and application logger
    """
    try:
        from utils.config import app_config, resolve_path

        error_log_path = resolve_path(app_config.error_log_path, create_dir=True)
    except ImportError:
        error_log_path = os.path.join(".", "logs", "workapp_errors.log")
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)

    timestamp = datetime.now().isoformat()
    log_entry: Dict[str, Any] = {
        "timestamp": timestamp,
        "type": error_type,
        "message": error_message,
    }

    if source:
        log_entry["source"] = source
    if include_traceback:
        log_entry["traceback"] = traceback.format_exc()
    if additional_data:
        log_entry["data"] = additional_data

    _log_to_file(log_entry, error_log_path)
    log_message = f"{error_message} [source: {source}]" if source else error_message
    logger.error(log_message, exc_info=include_traceback)


def log_warning(
    warning_message: str,
    include_traceback: bool = False,
    error_type: str = "WARNING",
    source: Optional[str] = None,
    additional_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a warning to the central error log file and application logger
    """
    try:
        from utils.config import app_config, resolve_path

        error_log_path = resolve_path(app_config.error_log_path, create_dir=True)
    except ImportError:
        error_log_path = os.path.join(".", "logs", "workapp_errors.log")
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)

    timestamp = datetime.now().isoformat()
    log_entry: Dict[str, Any] = {
        "timestamp": timestamp,
        "type": error_type,
        "message": warning_message,
    }

    if source:
        log_entry["source"] = source
    if include_traceback:
        log_entry["traceback"] = traceback.format_exc()
    if additional_data:
        log_entry["data"] = additional_data

    _log_to_file(log_entry, error_log_path)
    log_message = f"{warning_message} [source: {source}]" if source else warning_message
    logger.warning(log_message, exc_info=include_traceback)


class QueryLogger:
    """
    Logger for query metrics and zero-hit queries
    """

    def __init__(self, log_dir: str = "./logs"):
        try:
            from utils.config import resolve_path

            self.log_dir = resolve_path(log_dir, create_dir=True)
        except ImportError:
            self.log_dir = log_dir
            os.makedirs(self.log_dir, exist_ok=True)
        self.query_log_path = os.path.join(self.log_dir, "query_metrics.log")
        self.zero_hit_log_path = os.path.join(self.log_dir, "zero_hit_queries.log")

    def log_query(
        self, query: str, latency: float, hit_count: int, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log query metrics
        """
        try:
            timestamp = datetime.now().isoformat()
            log_entry: Dict[str, Any] = {
                "timestamp": timestamp,
                "query": query,
                "latency": round(latency, 4),
                "hit_count": hit_count,
            }
            if metadata:
                log_entry["metadata"] = metadata
            _log_to_file(log_entry, self.query_log_path)
            if hit_count == 0:
                self.log_zero_hit_query(query, metadata)
        except Exception as e:
            logger.error(f"Error logging query metrics: {str(e)}")
            log_error(
                f"Error logging query metrics: {str(e)}",
                include_traceback=True,
                source="QueryLogger.log_query",
            )

    def log_zero_hit_query(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log zero-hit queries for analysis
        """
        try:
            timestamp = datetime.now().isoformat()
            log_entry: Dict[str, Any] = {"timestamp": timestamp, "query": query}
            if metadata:
                log_entry["metadata"] = metadata
            _log_to_file(log_entry, self.zero_hit_log_path)
        except Exception as e:
            logger.error(f"Error logging zero-hit query: {str(e)}")
            log_error(
                f"Error logging zero-hit query: {str(e)}",
                include_traceback=True,
                source="QueryLogger.log_zero_hit_query",
            )

    def log_warning(self, warning_message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a warning related to query processing
        """
        try:
            timestamp = datetime.now().isoformat()
            log_entry: Dict[str, Any] = {
                "timestamp": timestamp,
                "type": "WARNING",
                "message": warning_message,
            }
            if metadata:
                log_entry["metadata"] = metadata
            warning_log_path = os.path.join(self.log_dir, "query_warnings.log")
            _log_to_file(log_entry, warning_log_path)
            logger.warning(f"Query warning: {warning_message}")
            log_warning(warning_message, source="QueryLogger", additional_data=metadata)
        except Exception as e:
            logger.error(f"Error logging query warning: {str(e)}")
            log_error(
                f"Error logging query warning: {str(e)}",
                include_traceback=True,
                source="QueryLogger.log_warning",
            )

    def log_error(self, error_message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error related to query processing
        """
        try:
            timestamp = datetime.now().isoformat()
            log_entry: Dict[str, Any] = {
                "timestamp": timestamp,
                "type": "ERROR",
                "message": error_message,
            }
            if metadata:
                log_entry["metadata"] = metadata
            error_log_path = os.path.join(self.log_dir, "query_errors.log")
            _log_to_file(log_entry, error_log_path)
            logger.error(f"Query error: {error_message}")
            log_error(
                error_message,
                include_traceback=False,
                error_type="QUERY_ERROR",
                source="QueryLogger",
                additional_data=metadata,
            )
        except Exception as e:
            logger.error(f"Error logging query error: {str(e)}")
            log_error(
                f"Error logging query error: {str(e)}",
                include_traceback=True,
                source="QueryLogger.log_error",
            )

    def analyze_zero_hit_queries(self, limit: int = 100) -> List[Dict[str, Any]]:
        # unchanged...
        ...

    def get_query_metrics(self, days: int = 7) -> Dict[str, Any]:
        # unchanged...
        ...


# Create global query logger instance
try:
    from utils.config import resolve_path

    query_logger = QueryLogger(log_dir=resolve_path(os.path.join(".", "logs"), create_dir=True))
except ImportError:
    query_logger = QueryLogger(log_dir=os.path.join(".", "logs"))
