# Text processing utilities for UI display
import re
from typing import Dict, List, Any, Tuple

def smart_wrap(text: str, width: int = 80) -> str:
    """Intelligently wrap text while preserving formatting
    
    Args:
        text: The text to wrap
        width: The maximum width of each line
        
    Returns:
        The wrapped text
    """
    # Note: We're not using Python's textwrap module anymore as it caused issues with
    # extra hyphens and poor indentation. Instead, we're using CSS-based wrapping
    # in the display_answer function in utils/ui/components.py.
    
    # We're no longer modifying the text here to avoid potential issues
    # The double spacing will be handled in safe_highlight_for_streamlit
    return text

def extract_confidence_score(text: str) -> int:
    """Extract confidence score from the answer
    
    Args:
        text: The text to extract the confidence score from
        
    Returns:
        The confidence score as an integer, or None if not found
    """
    match = re.search(r'Confidence score: (\d+)%', text)
    if match:
        return int(match.group(1))
    return None

def get_confidence_description(score: int) -> str:
    """Get a description of what the confidence score means
    
    Args:
        score: The confidence score
        
    Returns:
        A description of the confidence level
    """
    if score > 90:
        return "Answer is directly stated in the context with high certainty"
    elif score > 80:
        return "Answer is well-supported by the context"
    elif score > 70:
        return "Answer is supported by the context with some inference"
    elif score > 60:
        return "Answer requires moderate inference from context"
    elif score > 50:
        return "Answer is partially supported with significant uncertainty"
    elif score > 30:
        return "Answer involves substantial inference and uncertainty"
    else:
        return "Answer has very limited support from the context"

def safe_highlight_for_streamlit(text: str) -> Dict[str, Any]:
    """Create a version of the text with annotations for Streamlit display
    
    Args:
        text: The text to process
        
    Returns:
        Dictionary with processed text and highlighting information
    """
    result = {
        "text": text,
        "uncertain_sections": [],
        "term_distinctions": [],
        "warnings": []
    }
    
    # Extract uncertain sections
    uncertain_count = text.count('<uncertain>')
    if uncertain_count > 0:
        result["warnings"].append({
            "type": "uncertain",
            "message": f"This answer contains {uncertain_count} uncertain {'section' if uncertain_count == 1 else 'sections'} (highlighted in pink)."
        })
        
        # Find all uncertain sections
        for match in re.finditer(r'<uncertain>(.*?)</uncertain>', text, re.DOTALL):
            result["uncertain_sections"].append({
                "text": match.group(1),
                "start": match.start(),
                "end": match.end()
            })
    
    # Remove uncertainty tags for clean text
    #clean_text = re.sub(r'<uncertain>(.*?)</uncertain>', r'\1', text)
    clean_text = text
    # Replace any double newlines with single newlines
    #clean_text = re.sub(r'\n\n+', '\n', clean_text)
    
    # Find term distinctions
    disambiguation_patterns = [
        r'(different|distinct) (types?|categories|classifications) of ([\w\s]+)',
        r'(not to be confused with|distinguished from|different from) ([\w\s]+)',
        r'(\w+) (refers to|means|is defined as) ([^,.]+), while (\w+) (refers to|means|is defined as) ([^,.]+)',
        r'(\w+) and (\w+) are (different|distinct) ([\w\s]+)',
        r'(complementary|billable|free|paid) (service|appointment|revisit)',
        r'(Revisit|Service|Appointment) is (a|an) (complementary|billable|free|paid)',
        r'(\w+) (is classified as|is categorized as|falls under) ([\w\s]+)'
    ]
    
    found_distinctions = False
    for pattern in disambiguation_patterns:
        for match in re.finditer(pattern, clean_text, re.IGNORECASE):
            found_distinctions = True
            result["term_distinctions"].append({
                "text": match.group(0),
                "start": match.start(),
                "end": match.end()
            })
    
    if found_distinctions:
        result["warnings"].append({
            "type": "distinction",
            "message": "This answer contains important term distinctions (highlighted in red)."
        })
    
    # Check for term confusion
    confused_term_pairs = [
        (r'\b(Revisit)\b.*?\b(billable|paid)\b', 'Revisit is a complementary service, not billable'),
        (r'\b(complementary)\b.*?\b(charge|cost|fee|price)\b', 'Complementary services should not have charges'),
        (r'\b(free)\b.*?\b(charge|cost|fee|price)\b', 'Free services should not have charges'),
        (r'\b(billable)\b.*?\b(free|no charge|no cost)\b', 'Billable services are not free'),
        (r'\b(Appointment)\b.*?\b(complementary)\b', 'Be careful about appointment classifications'),
        (r'\b(Service)\b.*?\b(appointment)\b.*?\b(same|equivalent|equal)\b', 'Services and appointments are distinct concepts'),
    ]
    
    for pattern, correction in confused_term_pairs:
        if re.search(pattern, clean_text, re.IGNORECASE):
            result["warnings"].append({
                "type": "confusion",
                "message": f"POTENTIAL TERM CONFUSION DETECTED: {correction}. Please verify this information carefully and refer to the official documentation for clarification."
            })
            break
    
    result["text"] = clean_text
    return result

def get_confidence_color(score: int) -> Tuple[str, str]:
    """Get color and message for confidence score
    
    Args:
        score: The confidence score
        
    Returns:
        Tuple of (color_hex, confidence_message)
    """
    if score > 90:
        return "#4CAF50", "Very High Confidence"  # Green for very high confidence
    elif score > 80:
        return "#8BC34A", "High Confidence"  # Light green for high confidence
    elif score > 70:
        return "#CDDC39", "Good Confidence"  # Lime for good confidence
    elif score > 60:
        return "#FFC107", "Moderate Confidence"  # Amber for moderate confidence
    elif score > 50:
        return "#FF9800", "Fair Confidence"  # Orange for fair confidence
    elif score > 30:
        return "#FF5722", "Low Confidence"  # Deep orange for low confidence
    else:
        return "#F44336", "Very Low Confidence"  # Red for very low confidence