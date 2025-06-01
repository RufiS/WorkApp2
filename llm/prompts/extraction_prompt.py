"""Enhanced extraction prompt for LLM service with improved comprehensiveness and JSON reliability."""

import textwrap
from llm.prompts.sanitizer import sanitize_input


def generate_extraction_prompt(query: str, context: str, version: str = "v2") -> str:
    """
    Generate a prompt for extracting an answer from context with version selection.

    Args:
        query: The user's question
        context: The retrieved context
        version: Prompt version ("v2" for enhanced, "simple" for fallback)

    Returns:
        A formatted prompt string
    """
    if version == "simple":
        return generate_extraction_prompt_simple(query, context)
    elif version == "v2":
        return generate_extraction_prompt_v2(query, context)
    else:
        return generate_extraction_prompt_v2(query, context)  # Default to v2


def generate_extraction_prompt_v2(query: str, context: str) -> str:
    """
    Enhanced comprehensive extraction prompt with improved JSON reliability.

    Args:
        query: The user's question
        context: The retrieved context

    Returns:
        A formatted prompt string optimized for comprehensiveness and JSON compliance
    """
    query = sanitize_input(query)
    context = sanitize_input(context)

    return textwrap.dedent(
        f"""
        CRITICAL: RESPOND WITH ONLY VALID JSON - NO OTHER TEXT BEFORE OR AFTER

        You are a Karls Technology dispatcher assistant providing comprehensive answers.

        CONTEXT:
        {context}

        QUESTION: {query}

        COMPREHENSIVE ANSWER REQUIREMENTS:
        • Include ALL information related to the question from ANY part of the context
        • Connect related information across different sections of the context
        • Synthesize information from multiple sources when they relate to the same topic
        • For procedures: include ALL steps in exact order with complete details
        • For metro/city phone queries: map cities to specific phone numbers exactly as shown
        • Cross-reference related topics mentioned anywhere in the context
        • Include background information that supports or explains the main answer
        • Preserve exact terminology and classifications from the context

        SPECIAL HANDLING:
        • Metro areas: Find ALL phone numbers in the metro section and map them to their specific cities
        • Procedures: Include introductory text AND all procedural steps in original order
        • Multiple scenarios: If context contains multiple scenarios, include ALL relevant ones
        • Section headers: Use them to organize comprehensive answers
        • Contradictory info: Present both versions with clear distinctions

        If no relevant information exists in context: "Answer not found. Please contact a manager or fellow dispatcher."

        REQUIRED JSON FORMAT (ONLY THIS - NO OTHER TEXT):
        {{
          "answer": "comprehensive answer connecting all relevant information from context",
          "sources": ["relevant source/section references from context"],
          "confidence": 0.95
        }}

        JSON RESPONSE:
        """
    )





def generate_extraction_prompt_simple(query: str, context: str) -> str:
    """
    Simplified extraction prompt for fallback scenarios.

    Args:
        query: The user's question
        context: The retrieved context

    Returns:
        Simple, reliable prompt for basic extraction
    """
    query = sanitize_input(query)
    context = sanitize_input(context)

    return textwrap.dedent(
        f"""
        RESPOND WITH ONLY JSON - NO OTHER TEXT

        You are a Karls Technology dispatcher assistant. Answer using only the provided context.

        CONTEXT:
        {context}

        QUESTION: {query}

        If information isn't in context: "Answer not found. Please contact a manager or fellow dispatcher."

        JSON FORMAT:
        {{
          "answer": "comprehensive answer here",
          "sources": [],
          "confidence": 0.8
        }}

        JSON:
        """
    )
