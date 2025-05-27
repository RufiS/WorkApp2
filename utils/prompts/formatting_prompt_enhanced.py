"""Enhanced formatting prompt for LLM service."""

import re
import textwrap
from utils.prompts.sanitizer import sanitize_input


def generate_formatting_prompt(raw_answer: str) -> str:
    """
    Generate an enhanced prompt for formatting the raw answer.
    
    Args:
        raw_answer: The raw answer from the extraction model
        
    Returns:
        A formatted prompt string
    """
    raw_answer = sanitize_input(raw_answer)
    return textwrap.dedent(f"""
        System: You are a formatting assistant for Karls Technology dispatchers.
                If the draft reads exactly "Answer not found. Please contact a manager or fellow dispatcher.",
                return it unchanged. Otherwise, format the answer as follows:

        FORMATTING RULES:
        - Preserve ALL technical details, explanations, and procedural steps from the draft - DO NOT remove ANY information.
        - CRITICAL: NEVER change the order of procedural steps or instructions - maintain the EXACT sequence as presented in the draft.
        - IMPORTANT: Preserve ALL <uncertain>...</uncertain> tags in the original text - do not remove or modify these tags.
        - IMPORTANT: Preserve the exact confidence score at the end of the answer - do not increase or decrease it.
        - CRITICAL: Do not change entity types, classifications, or terminology - maintain exactly what was in the draft.
        - CRITICAL: Verify that terms like "Revisit", "Appointment", "Service", etc. maintain their EXACT classification (e.g., "complementary", "billable", etc.) from the draft.
        - CRITICAL: NEVER suggest websites or URLs that don't appear in the context or the draft answer.
        - CRITICAL: Do NOT substitute any terms with synonyms - use the EXACT terminology from the draft.
        - If you notice entity confusion or terminology mixing in the draft, add more <uncertain> tags around those sections.
        - If you notice inconsistent use of terms, add a note in <uncertain> tags explaining the inconsistency.
        - If the draft uses different terms for what appears to be the same concept, do not harmonize them - preserve the distinction and add an <uncertain> tag noting the potential confusion.
        - Ensure no line of text exceeds 80 characters in width. If a line would be longer, start a new line. Preserve all original formatting, such as indentation, bullet points, and numbered lists, when wrapping lines.
        - Break long lines into shorter segments for better readability.
        - Use short paragraphs with line breaks between them.
        - For numbered lists, maintain all detailed explanations for each step.
        - For bullet points, preserve all details while formatting for better readability.
        - Maintain all specific parameters, settings, configurations, and technical details.
        - Organize information in a logical flow.
        - Highlight important warnings or prerequisites.
        - Do not summarize or condense the information - keep ALL details intact.
        - Do not add new information not present in the draft.
        - Do not remove any uncertainty expressed in the draft.
        - When terms are disambiguated in the draft, ensure this disambiguation is clearly presented and emphasized.

        ENHANCED FORMATTING RULES:
        - PHONE NUMBERS: Always format phone numbers consistently as XXX-XXX-XXXX with hyphens.
        - CITY NAMES: Preserve exact city name spelling and capitalization as found in the draft.
        - METRO AREAS: Format metro area names in bold (e.g., **Tampa Metro**).
        - HEADERS: Format section headers in bold and on their own line.
        - LISTS: For lists of cities or services, use bullet points (•) with one item per line.
        - TABLES: If information is presented in a table-like format, preserve the alignment and structure.
        - WARNINGS: Format warnings or critical notes in bold and preceded by "⚠️ WARNING: ".
        - STEPS: For procedural steps, use numbered lists with clear step numbers (1., 2., etc.).
        - MARKDOWN: Use markdown formatting for better readability:
          * Bold for emphasis (**important**).
          * Italics for technical terms (*term*).
          * Code blocks for commands or inputs (`command`).
        - QUALITY CHECK: Verify that the final answer includes all required sections from the draft.

        DRAFT:
        
        {raw_answer}
        

        FINAL ANSWER:
    """)


def check_formatting_quality(formatted_answer: str, raw_answer: str) -> bool:
    """
    Check the quality of the formatted answer against the raw answer.
    
    Args:
        formatted_answer: The formatted answer
        raw_answer: The raw answer
        
    Returns:
        True if the formatting quality is acceptable, False otherwise
    """
    # Check if the formatted answer is too short compared to the raw answer
    if len(formatted_answer) < len(raw_answer) * 0.8:
        return False

    # Extract section headers from raw answer (assuming they're on their own lines)
    raw_sections: list[str] = re.findall(r'^([A-Z][A-Za-z\s]+):$', raw_answer, re.MULTILINE)

    # Check if each section header appears in the formatted answer
    for section in raw_sections:
        if section not in formatted_answer and f"**{section}**" not in formatted_answer:
            return False

    # Check if any <uncertain> tags were removed
    raw_uncertain_count = raw_answer.count('<uncertain>')
    formatted_uncertain_count = formatted_answer.count('<uncertain>')
    if formatted_uncertain_count < raw_uncertain_count:
        return False

    # Check if confidence score is preserved
    confidence_match_raw = re.search(r'Confidence: (\d+)%', raw_answer)
    confidence_match_formatted = re.search(r'Confidence: (\d+)%', formatted_answer)

    if confidence_match_raw and not confidence_match_formatted:
        return False

    if confidence_match_raw and confidence_match_formatted:
        if confidence_match_raw.group(1) != confidence_match_formatted.group(1):
            return False

    return True