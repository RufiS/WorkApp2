"""Enhanced formatting prompt for LLM service - LLM-first approach without regex."""

import textwrap
from llm.prompts.sanitizer import sanitize_input


def generate_formatting_prompt(raw_answer: str) -> str:
    """
    Generate an enhanced prompt for formatting the raw answer with proper dollar sign handling and bullet points.

    Args:
        raw_answer: The raw answer from the extraction model

    Returns:
        A formatted prompt string
    """
    raw_answer = sanitize_input(raw_answer)
    
    # Handle special case in Python code to avoid LLM confusion
    if raw_answer.strip() == "Answer not found. Please contact a manager or fellow dispatcher.":
        return raw_answer
    
    return textwrap.dedent(
        f"""
System: You are a professional formatting assistant for Karls Technology dispatchers. 
Format answers to be clear, readable, and professional while preserving all technical information.

CRITICAL FORMATTING RULES:

**Currency & Dollar Signs:**
- ALWAYS escape dollar signs: $125 → \\$125, $4 → \\$4
- Use format: \\$XXX.XX for all monetary amounts
- Never leave bare $ symbols (they break display)

**Bullet Points & Lists:**
- Use proper markdown bullets with line breaks:
  - Item 1
  - Item 2
  - Item 3
- Each bullet point MUST be on its own line
- Use "- " (dash space) for consistent formatting
- Leave blank lines between sections

**General Formatting:**
- Use **bold** for important terms and headings
- Format phone numbers as XXX-XXX-XXXX
- Preserve ALL technical details and confidence scores
- Use clear paragraph breaks for readability
- Maintain professional dispatcher terminology

**Example Format:**
**Main Answer:**
The service costs \\$125 for standard installation.

**Additional Details:**
- Equipment fee: \\$45
- Labor charge: \\$80
- Contact: 480-555-0123

Raw answer to format:
{raw_answer}

Formatted answer (follow all rules above):
"""
    )


def check_formatting_quality(formatted_answer: str, raw_answer: str) -> bool:
    """
    Check the quality of the formatted answer using simple string comparisons.

    Args:
        formatted_answer: The formatted answer
        raw_answer: The raw answer

    Returns:
        True if the formatting quality is acceptable, False otherwise
    """
    # Basic length check - formatted should not be significantly shorter
    if len(formatted_answer) < len(raw_answer) * 0.7:
        return False

    # Check if key terms from raw answer are preserved
    # Extract important words (simplified approach without regex)
    raw_words = set(word.lower() for word in raw_answer.split() if len(word) > 4)
    formatted_words = set(word.lower() for word in formatted_answer.split() if len(word) > 4)
    
    # At least 80% of important words should be preserved
    if len(raw_words) > 0:
        preserved_ratio = len(raw_words & formatted_words) / len(raw_words)
        if preserved_ratio < 0.8:
            return False

    # Check if confidence information is preserved (simple string check)
    if "confidence:" in raw_answer.lower() and "confidence" not in formatted_answer.lower():
        return False

    # Check if uncertain tags are preserved
    raw_uncertain_count = raw_answer.count("<uncertain>")
    formatted_uncertain_count = formatted_answer.count("<uncertain>")
    if formatted_uncertain_count < raw_uncertain_count:
        return False

    return True
