# WorkApp2 Logging Standards
"""
This module documents the standardized logging approach for WorkApp2.
It provides examples and best practices for consistent error logging across the codebase.
"""

import logging
from typing import Dict, Any, Optional

# Import standard logging functions
from utils.error_logging import log_error, log_warning

# Setup module logger
logger = logging.getLogger(__name__)


def example_function_with_standard_logging():
    """
    Example function demonstrating the standardized logging approach
    """
    try:
        # Function logic here
        result = 1 / 0  # This will raise a ZeroDivisionError
    except Exception as e:
        # 1. Log to application logger for immediate visibility in console/logs
        logger.error(f"Error in example_function: {str(e)}")

        # 2. Log to central error log for persistent storage and analysis
        log_error(
            error_message=f"Error in example function: {str(e)}",
            include_traceback=True,  # Include stack trace for debugging
            source="example_module.example_function",  # Identify the source
            additional_data={"context": "example operation"},  # Add relevant context
        )

        # 3. Re-raise or handle the exception as appropriate
        raise


def logging_best_practices():
    """
    Document best practices for logging in WorkApp2
    """
    return {
        "general_guidelines": [
            "Always use both logger.* and log_* functions for complete coverage",
            "Include source information in central logs for easier debugging",
            "Use appropriate log levels (error, warning, info) consistently",
            "Include stack traces for unexpected errors",
            "Add contextual data to help with debugging",
        ],
        "error_logging": [
            "Use log_error() for all errors that should be centrally tracked",
            "Include traceback for unexpected errors",
            "Always specify the source module/function",
            "Add relevant context data to help with debugging",
        ],
        "warning_logging": [
            "Use log_warning() for potential issues that don't prevent operation",
            "Be consistent with warning levels across the application",
            "Include enough context to understand the warning",
        ],
        "application_logging": [
            "Use logger.error(), logger.warning(), etc. for application logs",
            "Keep messages concise but informative",
            "Use consistent formatting across the application",
        ],
    }


def when_to_use_which_logging():
    """
    Guidelines for when to use different logging approaches
    """
    return {
        "log_error": [
            "Critical errors that prevent normal operation",
            "Unexpected exceptions that should be investigated",
            "Security-related issues",
            "Data integrity problems",
            "Configuration errors that prevent proper functioning",
        ],
        "log_warning": [
            "Potential issues that don't prevent operation",
            "Performance degradation scenarios",
            "Deprecated feature usage",
            "Recoverable errors with fallback behavior",
            "Edge cases that were handled but should be noted",
        ],
        "logger.error": [
            "Use alongside log_error() for immediate visibility",
            "For errors that should appear in application logs",
            "When detailed context isn't needed",
        ],
        "logger.warning": [
            "Use alongside log_warning() for immediate visibility",
            "For warnings that should appear in application logs",
            "When detailed context isn't needed",
        ],
        "logger.info": [
            "Normal operational events",
            "Startup/shutdown information",
            "Configuration loading",
            "Major state transitions",
        ],
        "logger.debug": [
            "Detailed information for debugging",
            "Function entry/exit points during troubleshooting",
            "Variable values during development",
            "Temporary logging during problem investigation",
        ],
    }
