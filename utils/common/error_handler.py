"""Common error handling utilities"""

import logging
import traceback
import functools
from typing import Optional, Type, Union, Tuple, Callable, Any
from utils.error_logging import log_error, log_warning

logger = logging.getLogger(__name__)


class CommonErrorHandler:
    """Common error handling patterns"""
    
    @staticmethod
    def handle_file_error(operation: str, file_path: str, error: Exception) -> str:
        """
        Handle file operation errors consistently
        
        Args:
            operation: The operation being performed
            file_path: Path to the file
            error: The exception that occurred
            
        Returns:
            Standardized error message
        """
        error_msg = f"Error {operation} file {file_path}: {str(error)}"
        
        if isinstance(error, FileNotFoundError):
            error_msg = f"File not found: {file_path}"
        elif isinstance(error, PermissionError):
            error_msg = f"Permission denied: {file_path}"
        elif isinstance(error, OSError):
            error_msg = f"OS error {operation} {file_path}: {str(error)}"
        
        logger.error(error_msg)
        log_error(error_msg, include_traceback=True)
        return error_msg
    
    @staticmethod
    def handle_validation_error(field: str, value: Any, expected: str) -> str:
        """
        Handle validation errors consistently
        
        Args:
            field: Field name being validated
            value: Value that failed validation
            expected: Description of expected value
            
        Returns:
            Standardized error message
        """
        error_msg = f"Invalid {field}: {value}. Expected {expected}"
        logger.error(error_msg)
        log_error(error_msg, include_traceback=False)
        return error_msg
    
    @staticmethod
    def handle_processing_error(component: str, operation: str, error: Exception) -> str:
        """
        Handle processing errors consistently
        
        Args:
            component: Component where error occurred
            operation: Operation being performed
            error: The exception that occurred
            
        Returns:
            Standardized error message
        """
        error_msg = f"Error in {component} during {operation}: {str(error)}"
        logger.error(error_msg, exc_info=True)
        log_error(error_msg, include_traceback=True)
        return error_msg
    
    @staticmethod
    def safe_execute(func: Callable, *args, default=None, log_errors: bool = True, **kwargs) -> Any:
        """
        Safely execute a function with error handling
        
        Args:
            func: Function to execute
            *args: Function arguments
            default: Default value to return on error
            log_errors: Whether to log errors
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or default value on error
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                error_msg = f"Error executing {func.__name__}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                log_error(error_msg, include_traceback=True)
            return default


def with_error_context(operation: str, component: str = ""):
    """
    Decorator to add error context to functions
    
    Args:
        operation: Description of the operation
        component: Component name (optional)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = f"{component} - " if component else ""
                error_msg = f"{context}Error during {operation}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                log_error(error_msg, include_traceback=True)
                raise RuntimeError(error_msg) from e
        return wrapper
    return decorator


def suppress_errors(default=None, log_errors: bool = True):
    """
    Decorator to suppress errors and return a default value
    
    Args:
        default: Default value to return on error
        log_errors: Whether to log errors
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    error_msg = f"Suppressed error in {func.__name__}: {str(e)}"
                    logger.warning(error_msg)
                    log_warning(error_msg, include_traceback=True)
                return default
        return wrapper
    return decorator
