"""Unified error handling package."""

# Basic decorators
from utils.error_handling.decorators import (
    with_retry,
    with_error_handling,
    with_recovery_strategy,
    RetryableError,
    ConfigurationError,
    IndexOperationError,
    APIError,
    NetworkError
)

# Enhanced decorators
from utils.error_handling.enhanced_decorators import (
    with_advanced_retry,
    with_error_tracking,
    with_fallback,
    with_timing,
    with_robust_error_handling,
    register_error,
    error_registry
)

__all__ = [
    # Basic decorators
    'with_retry',
    'with_error_handling',
    'with_recovery_strategy',
    'RetryableError',
    'ConfigurationError',
    'IndexOperationError',
    'APIError',
    'NetworkError',
    # Enhanced decorators
    'with_advanced_retry',
    'with_error_tracking',
    'with_fallback',
    'with_timing',
    'with_robust_error_handling',
    'register_error',
    'error_registry'
]
