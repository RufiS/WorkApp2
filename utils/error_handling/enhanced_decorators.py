# Enhanced error handling decorators
import logging
import time
import functools
import traceback
import random
from typing import Callable, Any, Dict, Optional, Type, List, Union

# Setup logging
logger = logging.getLogger(__name__)

# Error registry to track errors
error_registry = {
    "errors": [],
    "error_counts": {},
    "max_errors": 100  # Maximum number of errors to store
}

def register_error(error: Exception, source: str, attempt: int = 1, include_traceback: bool = True):
    """
    Register an error in the error registry
    
    Args:
        error: The exception that occurred
        source: Source of the error (usually function name)
        attempt: Attempt number (for retries)
        include_traceback: Whether to include traceback in error details
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Update error count
    if error_type not in error_registry["error_counts"]:
        error_registry["error_counts"][error_type] = 1
    else:
        error_registry["error_counts"][error_type] += 1
    
    # Create error entry
    error_entry = {
        "timestamp": time.time(),
        "type": error_type,
        "message": error_msg,
        "source": source,
        "attempt": attempt
    }
    
    # Add traceback if requested
    if include_traceback:
        error_entry["traceback"] = traceback.format_exc()
    
    # Add to errors list, maintaining max size
    error_registry["errors"].append(error_entry)
    if len(error_registry["errors"]) > error_registry["max_errors"]:
        error_registry["errors"].pop(0)  # Remove oldest error


def with_advanced_retry(max_attempts: int = 3, 
                       backoff_factor: float = 2.0,
                       exception_types: Optional[List[Type[Exception]]] = None,
                       on_retry: Optional[Callable[[Exception, int], None]] = None,
                       on_failure: Optional[Callable[[Exception, int], None]] = None,
                       jitter: bool = True,
                       max_backoff: float = 60.0,
                       propagate_final_exception: bool = True,
                       transform_exception: Optional[Callable[[Exception, int], Exception]] = None):
    """
    Advanced retry decorator with configurable backoff, exception filtering, and improved error propagation
    
    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Exponential backoff factor
        exception_types: List of exception types to retry on (None for all)
        on_retry: Callback function to execute on retry
        on_failure: Callback function to execute on final failure
        jitter: Whether to add random jitter to backoff times to prevent thundering herd
        max_backoff: Maximum backoff time in seconds
        propagate_final_exception: Whether to propagate the final exception after all retries fail
        transform_exception: Optional function to transform the exception before re-raising
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry this exception type
                    if exception_types is not None and not any(isinstance(e, exc_type) for exc_type in exception_types):
                        # Not a retryable exception
                        logger.warning(f"Non-retryable exception in {func.__name__}: {str(e)}")
                        break
                    
                    # Log the error
                    logger.warning(f"Attempt {attempt+1}/{max_attempts} failed in {func.__name__}: {str(e)}")
                    
                    # Register the error
                    register_error(e, func.__name__, attempt+1)
                    
                    # Call on_retry callback if provided
                    if on_retry is not None:
                        try:
                            on_retry(e, attempt+1)
                        except Exception as callback_error:
                            logger.error(f"Error in on_retry callback: {str(callback_error)}")
                    
                    # Check if we should retry
                    if attempt < max_attempts - 1:
                        # Calculate sleep time with exponential backoff
                        sleep_time = min(backoff_factor ** attempt, max_backoff)
                        
                        # Add jitter if enabled (±20% randomness)
                        if jitter:
                            jitter_factor = 1.0 + random.uniform(-0.2, 0.2)
                            sleep_time = sleep_time * jitter_factor
                        
                        logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        # Last attempt failed
                        logger.error(f"All {max_attempts} attempts failed in {func.__name__}")
                        
                        # Call on_failure callback if provided
                        if on_failure is not None:
                            try:
                                on_failure(e, max_attempts)
                            except Exception as callback_error:
                                logger.error(f"Error in on_failure callback: {str(callback_error)}")
            
            # If we get here, all attempts failed
            if last_exception is not None:
                # Transform exception if requested
                if transform_exception is not None:
                    try:
                        transformed_exception = transform_exception(last_exception, max_attempts)
                        if transformed_exception is not None and isinstance(transformed_exception, Exception):
                            if propagate_final_exception:
                                raise transformed_exception from last_exception
                    except Exception as transform_error:
                        logger.error(f"Error transforming exception: {str(transform_error)}")
                        # Fall back to original exception
                        if propagate_final_exception:
                            raise last_exception
                elif propagate_final_exception:
                    # Re-raise the original exception
                    raise last_exception
            
            # If we get here and propagate_final_exception is False, return None
            return None
            
        return wrapper
    return decorator

def with_error_tracking(error_types: Optional[List[Type[Exception]]] = None,
                       include_traceback: bool = True,
                       log_level: int = logging.ERROR,
                       propagate: bool = True,
                       transform_exception: Optional[Callable[[Exception], Exception]] = None):
    """
    Decorator to track errors in the error registry with improved error propagation
    
    Args:
        error_types: List of exception types to track (None for all)
        include_traceback: Whether to include traceback in error details
        log_level: Logging level for errors
        propagate: Whether to propagate the exception after tracking
        transform_exception: Optional function to transform the exception before re-raising
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if we should track this exception type
                if error_types is None or any(isinstance(e, exc_type) for exc_type in error_types):
                    # Track the error
                    register_error(e, func.__name__, include_traceback=include_traceback)
                    
                    # Log the error
                    logger.log(log_level, f"Error in {func.__name__}: {str(e)}")
                    if include_traceback:
                        logger.log(log_level, traceback.format_exc())
                
                # Transform exception if requested
                if transform_exception is not None:
                    try:
                        transformed_exception = transform_exception(e)
                        if transformed_exception is not None and isinstance(transformed_exception, Exception):
                            if propagate:
                                raise transformed_exception from e
                    except Exception as transform_error:
                        logger.error(f"Error transforming exception: {str(transform_error)}")
                        # Fall back to original exception
                        if propagate:
                            raise e
                elif propagate:
                    # Re-raise the original exception
                    raise
                    
                # If we get here, we're not propagating the exception
                return None
                
        return wrapper
    return decorator

def with_fallback(fallback_function: Optional[Callable] = None,
                 fallback_value: Any = None,
                 log_error: bool = True,
                 propagate_exceptions: Optional[List[Type[Exception]]] = None,
                 raise_on_fallback_error: bool = False):
    """
    Decorator to provide a fallback on error with improved error propagation
    
    Args:
        fallback_function: Function to call on error
        fallback_value: Value to return on error
        log_error: Whether to log the error
        propagate_exceptions: List of exception types that should be re-raised instead of using fallback
        raise_on_fallback_error: Whether to raise exceptions that occur in the fallback function
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Register the error
                register_error(e, func.__name__)
                
                # Check if we should propagate this exception type
                if propagate_exceptions and any(isinstance(e, exc_type) for exc_type in propagate_exceptions):
                    if log_error:
                        logger.error(f"Error in {func.__name__}, propagating exception: {str(e)}")
                    raise
                
                # Log the error
                if log_error:
                    logger.error(f"Error in {func.__name__}, using fallback: {str(e)}")
                
                # Use fallback function if provided
                if fallback_function is not None:
                    try:
                        return fallback_function(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Error in fallback function: {str(fallback_error)}")
                        if raise_on_fallback_error:
                            # Propagate the fallback error if configured to do so
                            raise fallback_error from e
                
                # Return fallback value
                return fallback_value
                
        return wrapper
    return decorator

def with_timing(log_level: int = logging.INFO,
               threshold: Optional[float] = None):
    """
    Decorator to log function execution time
    
    Args:
        log_level: Logging level for timing logs
        threshold: Only log if execution time exceeds threshold (seconds)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log execution time if it exceeds threshold
            if threshold is None or execution_time >= threshold:
                logger.log(log_level, f"{func.__name__} executed in {execution_time:.4f} seconds")
            
            return result
                
        return wrapper
    return decorator

def with_robust_error_handling(log_to_central: bool = True,
                              include_traceback: bool = True,
                              reraise: bool = True,
                              error_return_value: Any = None,
                              propagate_exceptions: Optional[List[Type[Exception]]] = None,
                              suppress_exceptions: Optional[List[Type[Exception]]] = None):
    """
    Comprehensive error handling decorator that logs to central error log
    with improved error propagation control
    
    Args:
        log_to_central: Whether to log to the central error log
        include_traceback: Whether to include traceback in error logs
        reraise: Whether to re-raise the exception after logging
        error_return_value: Value to return on error if not re-raising
        propagate_exceptions: List of exception types that should always be re-raised
                             regardless of reraise setting
        suppress_exceptions: List of exception types that should never be re-raised
                            regardless of reraise setting
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                error_msg = f"Error in {func.__name__}: {str(e)}"
                logger.error(error_msg)
                
                # Include traceback in logs if requested
                if include_traceback:
                    tb = traceback.format_exc()
                    logger.error(f"Traceback:\n{tb}")
                
                # Register the error in the registry
                register_error(e, func.__name__, include_traceback=include_traceback)
                
                # Log to central error log if requested
                if log_to_central:
                    try:
                        from utils.error_logging import log_error
                        log_error(error_msg, 
                                 include_traceback=include_traceback, 
                                 source=func.__name__,
                                 additional_data={"args": str(args), "kwargs": str(kwargs)})
                    except ImportError:
                        logger.warning("Could not import log_error function from utils.error_logging")
                        # Fallback to simple file logging
                        try:
                            with open('error_log.txt', 'a') as f:
                                f.write(f"{time.time()}: {error_msg}\n")
                                if include_traceback:
                                    f.write(f"Traceback:\n{traceback.format_exc()}\n")
                        except Exception as log_error:
                            logger.error(f"Failed to write to error log file: {str(log_error)}")
                
                # Determine whether to re-raise based on exception type and configuration
                should_reraise = reraise
                
                # Always propagate specific exception types if configured
                if propagate_exceptions and any(isinstance(e, exc_type) for exc_type in propagate_exceptions):
                    should_reraise = True
                    logger.info(f"Propagating exception of type {type(e).__name__} as configured")
                
                # Never propagate specific exception types if configured
                if suppress_exceptions and any(isinstance(e, exc_type) for exc_type in suppress_exceptions):
                    should_reraise = False
                    logger.info(f"Suppressing exception of type {type(e).__name__} as configured")
                
                # Re-raise or return error value based on configuration
                if should_reraise:
                    raise
                return error_return_value
                
        return wrapper
    return decorator