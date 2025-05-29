"""Enhanced extraction prompt for LLM service with improved comprehensiveness and JSON reliability."""

import textwrap
from utils.prompts.sanitizer import sanitize_input


def generate_extraction_prompt(query: str, context: str, version: str = "v2") -> str:
    """
    Generate a prompt for extracting an answer from context with version selection.

    Args:
        query: The user's question
        context: The retrieved context
        version: Prompt version ("v2" for enhanced, "legacy" for original)

    Returns:
        A formatted prompt string
    """
    if version == "legacy":
        return generate_extraction_prompt_legacy(query, context)
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


def generate_extraction_prompt_legacy(query: str, context: str) -> str:
    """
    Legacy extraction prompt for comparison and fallback.

    Args:
        query: The user's question
        context: The retrieved context

    Returns:
        Original formatted prompt string
    """
    query = sanitize_input(query)
    context = sanitize_input(context)
    
    return textwrap.dedent(
        f"""
        System: You are an expert Karls Technology dispatcher assistant.
                Answer strictly using provided context. If not found, reply: Not found in document.
                If the answer is not in CONTEXT, respond exactly:
                "Answer not found. Please contact a manager or fellow dispatcher."

        CONTEXT:
        
        {context}
        
        QUESTION:
        {query}

        INSTRUCTIONS:
          - CRITICAL REQUIREMENT: Provide an EXHAUSTIVELY comprehensive answer with ALL relevant information from the context. NEVER omit contextual information.
          - ALWAYS begin your answer with any introductory or general text that appears before procedural steps in the context.
          - Maintain section order exactly as in the context: include general info then procedural steps in their original sequence.
          - Preserve the EXACT order of all procedural steps without rearrangement.
          - If the context contains SECTION headers, use them to organize your answer.
          - If information about a topic appears in multiple sections, synthesize it coherently while maintaining the original meaning.
          - If you find contradictory information about the same topic, note the contradiction and present both versions.
          - Pay attention to cross-references between sections and ensure your answer reflects these relationships.

          - Metro & City Association:
            1. Identify each "Metro" heading (e.g. "Tampa Metro", "St Petersburg / Tampa Metro").
            2. Define a Metro block as everything from that heading up to (but not including) the next Metro heading.
            3. Within each Metro block:
               a. Find all instances of phone mappings by scanning for one of these exact patterns:
                  - "<MetroName> Phone Number: ###-###-####"
                  - "###-###-#### – City1, City2, …"
                  - "CityName – ###-###-####"
               b. Also locate an "Areas Serviced:" line (case-sensitive) anywhere in the block and extract its comma-separated list of cities.
               c. For each sub-record (i.e. each Phone Number instance plus its associated cities from either the "Areas Serviced:" list or the same line), map **only** those cities to that specific phone number.
               d. Ignore any intervening "Table of Contents" or URL lines when pairing a Phone Number with its "Areas Serviced:" list.
            4. When multiple phone numbers appear in one Metro block, treat each separately—do **not** mix cities across different sub-records.
            5. If asked about a **city**, return the number from the sub-record whose city list contains that exact name (match full words only).
            6. If asked about the **Metro** itself, return **all** unique phone numbers found in its entire block.
            

          - Terminology & Disambiguation:
            - Use ONLY the exact terminology, headings, and acronyms from CONTEXT.
            - If ambiguity or multiple definitions arise, mark uncertain parts with <uncertain>…</uncertain> tags and explain briefly.
            - For terms with multiple meanings in the context, explicitly state which meaning you are using.
            - For appointments and services (e.g., "Revisit", "Appointment", "Service"), specify their classification (e.g., "complementary", "billable") exactly as in the context.

          - Uncertainty marking:
            - Use <uncertain>…</uncertain> tags when you infer information or face ambiguity.

          - Confidence scoring:
            - Append a confidence score (0–100%) based solely on how explicitly the answer is supported by the context.
            - Use 90–100% only for fully explicit answers; lower scores for any inference or ambiguity.

         IMPORTANT: You must respond with valid JSON in exactly this format:
         {{
           "answer": "your comprehensive answer here",
           "sources": ["relevant source references if any"],
           "confidence": 0.95
         }}

         JSON Response:
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
