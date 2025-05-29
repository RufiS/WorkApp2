"""Sanitizer utilities for LLM prompts."""


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to help prevent prompt injection.
    Escapes triple backticks and strips dangerous characters.

    Args:
        text: The user input string.

    Returns:
        The sanitized string.
    """
    # Escape triple backticks
    sanitized = text.replace("```", r"\`\`\`")
    # Remove null bytes
    sanitized = sanitized.replace("\x00", "")
    # Optionally, add more sanitization here
    return sanitized
