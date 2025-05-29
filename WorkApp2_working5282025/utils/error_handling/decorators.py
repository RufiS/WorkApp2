import functools
import logging
import time
from typing import Callable

# Setup logging
logger = logging.getLogger(__name__)

# Define specific exception types for better error handling
class RetryableError(Exception):
    """Base class for errors that should be retried"""
    pass

class ConfigurationError(Exception):
    """Error related to configuration issues"""
    pass

class IndexOperationError(Exception):
    """Error related to index operations"""
    pass

class APIError(RetryableError):
    """Error related to external API calls"""
    pass

class NetworkError(RetryableError):
    """Error related to network connectivity"""
    pass

# Type alias for better readability
ExceptionTypes = (
    type[BaseException]
    | tuple[type[BaseException], ...]
    | list[type[BaseException]]
)

def _ensure_tuple_ex_types(
    exception_types: ExceptionTypes | None,
    default: tuple[type[BaseException], ...] | None = (RetryableError,)
) -> tuple[type[BaseException], ...]:
    # Accept None or empty list/tuple as default
    if exception_types is None or not exception_types:
        return default if default is not None else (Exception,)
    if isinstance(exception_types, type):
        return (exception_types,)
    # By this point, exception_types should be a tuple or list
    return tuple(exception_types)

def with_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exception_types: ExceptionTypes | None = None,
    context: str = "",
    propagate_final_exception: bool = True,
    add_context_to_exception: bool = True
) -> Callable[..., object]:
    """
    Decorator to retry a function on specified exceptions.
    """
    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            exceptions_to_catch = _ensure_tuple_ex_types(exception_types)
            last_exception = None
            context_info = f" [{context}]" if context else ""
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions_to_catch as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = backoff_factor ** attempt
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}{context_info}: {str(e)}. Retrying in {wait_time:.2f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}{context_info}: {str(e)}"
                        )

            # All attempts failed
            if last_exception:
                if propagate_final_exception:
                    if add_context_to_exception:
                        try:
                            raise last_exception.__class__(
                                f"Failed after {max_attempts} attempts{context_info}: {str(last_exception)}"
                            ) from last_exception
                        except Exception:
                            raise last_exception
                    else:
                        raise last_exception
                else:
                    logger.error(
                        f"Suppressing exception after {max_attempts} failed attempts: {str(last_exception)}"
                    )
                    return None
            else:
                error_msg = f"Failed after {max_attempts} attempts{context_info} with unknown error"
                if propagate_final_exception:
                    raise RuntimeError(error_msg)
                else:
                    logger.error(error_msg)
                    return None
        return wrapper
    return decorator

def with_error_handling(
    default_return_value: object = None,
    log_level: str = "error",
    exception_types: ExceptionTypes | None = None,
    include_traceback: bool = False,
    context: str = "",
    propagate_exceptions: ExceptionTypes | None = None
) -> Callable[..., object]:
    """
    Decorator for error-handling and fallback return values.
    """
    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            exceptions_to_catch = _ensure_tuple_ex_types(exception_types, default=(Exception,))
            exceptions_to_propagate = _ensure_tuple_ex_types(propagate_exceptions, default=())
            context_info = f" [{context}]" if context else ""
            try:
                return func(*args, **kwargs)
            except exceptions_to_propagate as e:
                log_method = getattr(logger, log_level.lower(), logger.error)
                log_method(f"Propagating exception in {func.__name__}{context_info}: {str(e)}")
                raise
            except exceptions_to_catch as e:
                log_method = getattr(logger, log_level.lower(), logger.error)
                error_msg = f"Error in {func.__name__}{context_info}: {str(e)}"
                if include_traceback:
                    log_method(error_msg, exc_info=True)
                else:
                    log_method(error_msg)
                return default_return_value
        return wrapper
    return decorator

def with_recovery_strategy(
    strategies: dict[type[BaseException], Callable[..., object]],
    default_strategy: Callable[..., object] | None = None,
    log_level: str = "error",
    context: str = "",
    propagate_unhandled: bool = True,
    propagate_strategy_errors: bool = False
) -> Callable[..., object]:
    """
    Decorator that applies custom recovery strategies for exceptions.
    """
    def decorator(func: Callable[..., object]) -> Callable[..., object]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            context_info = f" [{context}]" if context else ""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_method = getattr(logger, log_level.lower(), logger.error)
                log_method(f"Error in {func.__name__}{context_info}: {str(e)}")

                for exc_type, strategy in strategies.items():
                    if isinstance(e, exc_type):
                        log_method(f"Applying recovery strategy for {exc_type.__name__}")
                        try:
                            return strategy(e, *args, **kwargs)
                        except Exception as strategy_error:
                            log_method(
                                f"Error in recovery strategy for {exc_type.__name__}: {str(strategy_error)}"
                            )
                            if propagate_strategy_errors:
                                raise strategy_error from e
                            break

                if default_strategy:
                    log_method("Applying default recovery strategy")
                    try:
                        return default_strategy(e, *args, **kwargs)
                    except Exception as default_error:
                        log_method(
                            f"Error in default recovery strategy: {str(default_error)}"
                        )
                        if propagate_strategy_errors:
                            raise default_error from e
                        if propagate_unhandled:
                            raise e
                        return None

                if propagate_unhandled:
                    raise e
                else:
                    return None
        return wrapper
    return decorator
