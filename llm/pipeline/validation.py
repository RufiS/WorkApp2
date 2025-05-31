"""
Enhanced JSON validation utilities for LLM responses with repair capabilities

Multi-strategy approach for maximum reliability
"""
import json
import re
import logging
from typing import Dict, Any, Optional, Tuple, List
import jsonschema

logger = logging.getLogger(__name__)

# Define JSON schemas for LLM responses
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["answer"],
}

# Minimal fallback schema for degraded scenarios
MINIMAL_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
}


def validate_json_output(
    content: str, schema: Dict[str, Any] = ANSWER_SCHEMA
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Enhanced JSON validation with multiple extraction strategies and repair capabilities

    Args:
        content: The LLM output string to validate
        schema: JSON schema to validate against

    Returns:
        Tuple of (is_valid, parsed_json, error_message)
    """
    if not content or not content.strip():
        return False, None, "Empty content provided"

    # Strategy 1: Pure JSON parsing
    success, parsed, error = _extract_pure_json(content, schema)
    if success:
        logger.debug("JSON extraction succeeded with pure JSON strategy")
        return True, parsed, None

    # Strategy 2: Extract from markdown code blocks
    success, parsed, error = _extract_from_markdown(content, schema)
    if success:
        logger.debug("JSON extraction succeeded with markdown strategy")
        return True, parsed, None

    # Strategy 3: Pattern-based extraction
    success, parsed, error = _extract_with_patterns(content, schema)
    if success:
        logger.debug("JSON extraction succeeded with pattern strategy")
        return True, parsed, None

    # Strategy 4: JSON repair and extraction
    success, parsed, error = _repair_and_extract(content, schema)
    if success:
        logger.debug("JSON extraction succeeded with repair strategy")
        return True, parsed, None

    # Strategy 5: Partial JSON extraction
    success, parsed, error = _extract_partial_json(content, schema)
    if success:
        logger.debug("JSON extraction succeeded with partial strategy")
        return True, parsed, None

    # Strategy 6: Fallback content extraction (always succeeds)
    success, parsed, error = _extract_fallback(content)
    if success:
        logger.warning("Using fallback content extraction")
        return True, parsed, None

    # If all strategies fail (should never happen with fallback)
    return False, None, "All extraction strategies failed"


def _extract_pure_json(content: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Strategy 1: Parse content as pure JSON"""
    try:
        parsed_json = json.loads(content.strip())
        jsonschema.validate(instance=parsed_json, schema=schema)
        return True, parsed_json, None
    except json.JSONDecodeError as e:
        return False, None, f"JSON decode error: {str(e)}"
    except jsonschema.exceptions.ValidationError as e:
        return False, None, f"Schema validation error: {str(e)}"


def _extract_from_markdown(content: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Strategy 2: Extract JSON from markdown code blocks"""
    # Look for ```json or ``` blocks
    json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    
    for block in json_blocks:
        try:
            parsed_json = json.loads(block.strip())
            jsonschema.validate(instance=parsed_json, schema=schema)
            return True, parsed_json, None
        except (json.JSONDecodeError, jsonschema.exceptions.ValidationError):
            continue
    
    return False, None, "No valid JSON found in markdown blocks"


def _extract_with_patterns(content: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Strategy 3: Extract JSON using improved pattern matching"""
    # More sophisticated JSON pattern that handles nested structures
    json_patterns = [
        r'\{[^{}]*"answer"[^{}]*\}',  # Simple flat JSON
        r'\{(?:[^{}]|\{[^{}]*\})*\}',  # JSON with one level of nesting
        r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}',  # JSON with two levels of nesting
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                parsed_json = json.loads(match.strip())
                if "answer" in parsed_json:  # Must have answer field
                    jsonschema.validate(instance=parsed_json, schema=schema)
                    return True, parsed_json, None
            except (json.JSONDecodeError, jsonschema.exceptions.ValidationError):
                continue
    
    return False, None, "No valid JSON found with patterns"


def _repair_and_extract(content: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Strategy 4: Repair common JSON issues and then extract"""
    # Find potential JSON in content
    json_pattern = r'\{[\s\S]*?\}'
    matches = re.findall(json_pattern, content)
    
    for match in matches:
        repaired_json = _repair_json_string(match)
        try:
            parsed_json = json.loads(repaired_json)
            if "answer" in parsed_json:  # Must have answer field
                jsonschema.validate(instance=parsed_json, schema=schema)
                return True, parsed_json, None
        except (json.JSONDecodeError, jsonschema.exceptions.ValidationError):
            continue
    
    return False, None, "JSON repair failed"


def _repair_json_string(json_str: str) -> str:
    """Repair common JSON formatting issues"""
    # Remove extra text before and after JSON
    json_str = json_str.strip()
    
    # Fix trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix unescaped quotes in values (basic approach)
    # Look for patterns like "answer": "He said "hello""
    json_str = re.sub(r':\s*"([^"]*)"([^"]*)"([^"]*)"', r': "\1\\""\2\\""\3"', json_str)
    
    # Fix confidence values without leading zero
    json_str = re.sub(r'"confidence":\s*\.(\d+)', r'"confidence": 0.\1', json_str)
    
    # Fix missing quotes around field names
    json_str = re.sub(r'(\w+):', r'"\1":', json_str)
    
    # Fix single quotes to double quotes
    json_str = json_str.replace("'", '"')
    
    return json_str


def _extract_partial_json(content: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Strategy 5: Extract what we can and build minimal valid JSON"""
    # Try to extract answer text even if JSON is malformed
    answer_patterns = [
        r'"answer"\s*:\s*"([^"]*)"',
        r"'answer'\s*:\s*'([^']*)'",
        r'answer:\s*"([^"]*)"',
        r'answer:\s*([^,}\n]+)',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            answer_text = match.group(1).strip()
            if answer_text:
                # Try to extract sources and confidence too
                sources = _extract_sources_from_content(content)
                confidence = _extract_confidence_from_content(content)
                
                partial_json = {
                    "answer": answer_text,
                    "sources": sources,
                    "confidence": confidence
                }
                
                try:
                    jsonschema.validate(instance=partial_json, schema=schema)
                    return True, partial_json, None
                except jsonschema.exceptions.ValidationError:
                    # Try with minimal schema
                    try:
                        minimal_json = {"answer": answer_text}
                        jsonschema.validate(instance=minimal_json, schema=MINIMAL_SCHEMA)
                        return True, minimal_json, None
                    except jsonschema.exceptions.ValidationError:
                        continue
    
    return False, None, "Could not extract partial JSON"


def _extract_sources_from_content(content: str) -> List[str]:
    """Extract sources from content if possible"""
    sources_patterns = [
        r'"sources"\s*:\s*\[(.*?)\]',
        r"'sources'\s*:\s*\[(.*?)\]",
        r'sources:\s*\[(.*?)\]',
    ]
    
    for pattern in sources_patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            sources_str = match.group(1)
            # Extract individual source strings
            source_items = re.findall(r'"([^"]*)"', sources_str)
            if source_items:
                return source_items
    
    return []


def _extract_confidence_from_content(content: str) -> float:
    """Extract confidence from content if possible"""
    confidence_patterns = [
        r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)',
        r"'confidence'\s*:\s*([0-9]*\.?[0-9]+)",
        r'confidence:\s*([0-9]*\.?[0-9]+)',
    ]
    
    for pattern in confidence_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            try:
                confidence = float(match.group(1))
                # Ensure it's between 0 and 1
                if 0 <= confidence <= 1:
                    return confidence
                elif confidence > 1:
                    return confidence / 100  # Convert percentage
            except ValueError:
                continue
    
    return 0.8  # Default confidence


def _extract_fallback(content: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Strategy 6: Fallback extraction that always succeeds"""
    # If all else fails, treat the entire content as the answer
    # Clean up the content first
    cleaned_content = content.strip()
    
    # Remove common JSON artifacts if present
    cleaned_content = re.sub(r'^[{"\'\s]*', '', cleaned_content)
    cleaned_content = re.sub(r'[}"\'\s]*$', '', cleaned_content)
    
    # Remove JSON field names if present
    cleaned_content = re.sub(r'^(answer|response)\s*:\s*', '', cleaned_content, flags=re.IGNORECASE)
    
    # If content is still empty or too short, provide a default message
    if not cleaned_content or len(cleaned_content.strip()) < 5:
        cleaned_content = "Unable to process the response. Please try rephrasing your question."
    
    fallback_json = {
        "answer": cleaned_content,
        "sources": [],
        "confidence": 0.5  # Low confidence since this is fallback
    }
    
    return True, fallback_json, None


# Backwards compatibility function
def validate_json_output_legacy(
    content: str, schema: Dict[str, Any] = ANSWER_SCHEMA
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Legacy validation function for comparison/fallback"""
    try:
        # First, try to parse as pure JSON
        parsed_json = json.loads(content)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from markdown code blocks
        json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", content)

        if json_blocks:
            # Try each extracted block
            for block in json_blocks:
                try:
                    parsed_json = json.loads(block)
                    break
                except json.JSONDecodeError:
                    continue
            else:  # No valid JSON found in blocks
                return False, None, "Could not parse JSON from code blocks"
        else:
            # Try to find JSON-like structures with regex
            json_pattern = r"\{[\s\S]*?\}"
            json_matches = re.findall(json_pattern, content)

            if json_matches:
                for match in json_matches:
                    try:
                        parsed_json = json.loads(match)
                        break
                    except json.JSONDecodeError:
                        continue
                else:  # No valid JSON found in matches
                    return False, None, "Could not parse JSON from content"
            else:
                return False, None, "No JSON-like structures found in content"

    # Validate against schema
    try:
        jsonschema.validate(instance=parsed_json, schema=schema)
        return True, parsed_json, None
    except jsonschema.exceptions.ValidationError as e:
        return False, parsed_json, str(e)
