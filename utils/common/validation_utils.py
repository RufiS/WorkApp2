"""Common validation utilities"""

import os
import re
import logging
from typing import Any, Optional, List, Dict, Union, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationUtils:
    """Common validation patterns and utilities"""

    @staticmethod
    def validate_file_path(file_path: str, must_exist: bool = True,
                          allowed_extensions: Optional[List[str]] = None) -> bool:
        """
        Validate a file path

        Args:
            file_path: Path to validate
            must_exist: Whether the file must exist
            allowed_extensions: List of allowed file extensions

        Returns:
            True if valid, False otherwise
        """
        if not file_path or not isinstance(file_path, str):
            return False

        path = Path(file_path)

        # Check if file exists if required
        if must_exist and not path.exists():
            return False

        # Check if it's a file (not directory)
        if must_exist and not path.is_file():
            return False

        # Check allowed extensions
        if allowed_extensions:
            file_ext = path.suffix.lower()
            if file_ext not in allowed_extensions:
                return False

        return True

    @staticmethod
    def validate_positive_number(value: Any, allow_zero: bool = False) -> bool:
        """
        Validate that a value is a positive number

        Args:
            value: Value to validate
            allow_zero: Whether zero is allowed

        Returns:
            True if valid, False otherwise
        """
        try:
            num = float(value)
            return num > 0 or (allow_zero and num >= 0)
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_string_length(value: str, min_length: int = 0,
                              max_length: Optional[int] = None) -> bool:
        """
        Validate string length

        Args:
            value: String to validate
            min_length: Minimum length
            max_length: Maximum length (None for no limit)

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(value, str):
            return False

        length = len(value)

        if length < min_length:
            return False

        if max_length is not None and length > max_length:
            return False

        return True

    @staticmethod
    def validate_dict_keys(data: Dict[str, Any], required_keys: List[str],
                          optional_keys: Optional[List[str]] = None) -> bool:
        """
        Validate dictionary has required keys and no unexpected keys

        Args:
            data: Dictionary to validate
            required_keys: List of required keys
            optional_keys: List of optional keys

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, dict):
            return False

        # Check required keys
        for key in required_keys:
            if key not in data:
                return False

        # Check for unexpected keys
        if optional_keys is not None:
            allowed_keys = set(required_keys + optional_keys)
            if not set(data.keys()).issubset(allowed_keys):
                return False

        return True

    @staticmethod
    def validate_range(value: Union[int, float], min_val: Optional[Union[int, float]] = None,
                      max_val: Optional[Union[int, float]] = None) -> bool:
        """
        Validate that a value is within a range

        Args:
            value: Value to validate
            min_val: Minimum value (None for no minimum)
            max_val: Maximum value (None for no maximum)

        Returns:
            True if valid, False otherwise
        """
        try:
            num = float(value)

            if min_val is not None and num < min_val:
                return False

            if max_val is not None and num > max_val:
                return False

            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email format

        Args:
            email: Email to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(email, str):
            return False

        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format

        Args:
            url: URL to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(url, str):
            return False

        pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$'
        return bool(re.match(pattern, url))

    @staticmethod
    def sanitize_filename(filename: str, replacement: str = "_") -> str:
        """
        Sanitize a filename by removing/replacing invalid characters

        Args:
            filename: Filename to sanitize
            replacement: Character to replace invalid characters with

        Returns:
            Sanitized filename
        """
        if not isinstance(filename, str):
            return ""

        # Remove or replace invalid characters
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized = re.sub(invalid_chars, replacement, filename)

        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(' .')

        # Ensure it's not empty
        if not sanitized:
            sanitized = "untitled"

        return sanitized

    @staticmethod
    def validate_with_schema(data: Any, schema: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate data against a simple schema

        Args:
            data: Data to validate
            schema: Schema definition

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            # Simple type checking
            if 'type' in schema:
                expected_type = schema['type']
                if expected_type == 'dict' and not isinstance(data, dict):
                    errors.append(f"Expected dict, got {type(data).__name__}")
                elif expected_type == 'list' and not isinstance(data, list):
                    errors.append(f"Expected list, got {type(data).__name__}")
                elif expected_type == 'str' and not isinstance(data, str):
                    errors.append(f"Expected str, got {type(data).__name__}")
                elif expected_type == 'int' and not isinstance(data, int):
                    errors.append(f"Expected int, got {type(data).__name__}")
                elif expected_type == 'float' and not isinstance(data, (int, float)):
                    errors.append(f"Expected float, got {type(data).__name__}")

            # Required fields for dicts
            if isinstance(data, dict) and 'required' in schema:
                for field in schema['required']:
                    if field not in data:
                        errors.append(f"Missing required field: {field}")

            # String length validation
            if isinstance(data, str) and 'min_length' in schema:
                if len(data) < schema['min_length']:
                    errors.append(f"String too short: {len(data)} < {schema['min_length']}")

            if isinstance(data, str) and 'max_length' in schema:
                if len(data) > schema['max_length']:
                    errors.append(f"String too long: {len(data)} > {schema['max_length']}")

            # Numeric range validation
            if isinstance(data, (int, float)):
                if 'minimum' in schema and data < schema['minimum']:
                    errors.append(f"Value too small: {data} < {schema['minimum']}")
                if 'maximum' in schema and data > schema['maximum']:
                    errors.append(f"Value too large: {data} > {schema['maximum']}")

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return len(errors) == 0, errors


def validate_required_fields(required_fields: List[str]):
    """
    Decorator to validate required fields in function arguments

    Args:
        required_fields: List of required field names
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function argument names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Check required fields
            for field in required_fields:
                if field not in bound_args.arguments:
                    raise ValueError(f"Missing required field: {field}")
                if bound_args.arguments[field] is None:
                    raise ValueError(f"Required field cannot be None: {field}")

            return func(*args, **kwargs)
        return wrapper
    return decorator
