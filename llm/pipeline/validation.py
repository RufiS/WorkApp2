"""
LLM-First JSON validation utilities for LLM responses

Pure LLM approach without regex dependencies
"""
import json
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
    content: str, 
    schema: Dict[str, Any] = ANSWER_SCHEMA,
    llm_service: Optional[Any] = None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    LLM-First JSON validation without regex
    
    Args:
        content: The LLM output string to validate
        schema: JSON schema to validate against
        llm_service: Optional LLM service for JSON fixing
        
    Returns:
        Tuple of (is_valid, parsed_json, error_message)
    """
    if not content or not content.strip():
        logger.error("Empty content provided for JSON validation")
        return False, None, "Empty content provided"
    
    # Strategy 1: Direct JSON parsing (no regex)
    success, parsed, error = _try_direct_json_parse(content, schema)
    if success:
        logger.debug("JSON extraction succeeded with direct parsing")
        return True, parsed, None
    
    # Strategy 2: LLM-based JSON extraction
    if llm_service:
        success, parsed, error = _llm_extract_json(content, schema, llm_service)
        if success:
            logger.debug("JSON extraction succeeded with LLM assistance")
            return True, parsed, None
    
    # Strategy 3: LLM-based JSON repair
    if llm_service:
        success, parsed, error = _llm_repair_json(content, schema, llm_service)
        if success:
            logger.debug("JSON extraction succeeded with LLM repair")
            return True, parsed, None
    
    # Strategy 4: Content-based fallback (always succeeds)
    success, parsed, error = _content_fallback(content, llm_service)
    if success:
        logger.warning(f"Using content fallback for: {content[:100]}...")
        return True, parsed, None
    
    # This should never happen with fallback
    logger.error(f"All JSON extraction strategies failed for: {content[:200]}...")
    return False, None, "All extraction strategies failed"


def _try_direct_json_parse(content: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Try to parse content as direct JSON without any regex preprocessing"""
    content_stripped = content.strip()
    
    # Check if content looks like it starts and ends with JSON delimiters
    if content_stripped.startswith('{') and content_stripped.endswith('}'):
        try:
            parsed_json = json.loads(content_stripped)
            jsonschema.validate(instance=parsed_json, schema=schema)
            return True, parsed_json, None
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {str(e)}")
        except jsonschema.exceptions.ValidationError as e:
            logger.debug(f"Schema validation failed: {str(e)}")
    
    return False, None, "Not valid JSON format"


def _llm_extract_json(
    content: str, 
    schema: Dict[str, Any], 
    llm_service: Any
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Use LLM to extract JSON from content"""
    
    extraction_prompt = f"""
Extract the JSON object from the following content. The JSON should have these fields:
- "answer": The main answer text
- "sources": Array of source references (optional)
- "confidence": Number between 0 and 1 (optional)

Content to extract from:
{content}

Respond with ONLY the JSON object, starting with {{ and ending with }}:
"""
    
    try:
        llm_response = llm_service.generate(
            extraction_prompt,
            temperature=0.0,
            max_tokens=2000
        )
        
        if llm_response:
            parsed_json = json.loads(llm_response.strip())
            jsonschema.validate(instance=parsed_json, schema=schema)
            logger.info(f"LLM successfully extracted JSON from malformed content")
            return True, parsed_json, None
            
    except Exception as e:
        logger.warning(f"LLM JSON extraction failed: {str(e)}")
    
    return False, None, "LLM extraction failed"


def _llm_repair_json(
    content: str, 
    schema: Dict[str, Any], 
    llm_service: Any
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Use LLM to repair malformed JSON"""
    
    repair_prompt = f"""
The following content appears to be malformed JSON. Please fix it and return valid JSON.

Requirements:
- Must be valid JSON starting with {{ and ending with }}
- Must have an "answer" field with the main response
- Should have "sources" array if sources are mentioned
- Should have "confidence" between 0-1 if confidence is mentioned

Content to repair:
{content}

Return ONLY the corrected JSON:
"""
    
    try:
        llm_response = llm_service.generate(
            repair_prompt,
            temperature=0.0,
            max_tokens=2000
        )
        
        if llm_response:
            parsed_json = json.loads(llm_response.strip())
            jsonschema.validate(instance=parsed_json, schema=schema)
            logger.info(f"LLM successfully repaired malformed JSON")
            return True, parsed_json, None
            
    except Exception as e:
        logger.warning(f"LLM JSON repair failed: {str(e)}")
    
    return False, None, "LLM repair failed"


def _content_fallback(
    content: str,
    llm_service: Optional[Any] = None
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Fallback that treats content as answer text"""
    
    # If we have LLM service, use it to clean the content
    if llm_service:
        cleanup_prompt = f"""
Extract just the answer text from the following content, removing any JSON formatting or field names:

{content}

Return only the clean answer text:
"""
        
        try:
            cleaned_answer = llm_service.generate(
                cleanup_prompt,
                temperature=0.0,
                max_tokens=2000
            )
            
            if cleaned_answer and cleaned_answer.strip():
                content = cleaned_answer.strip()
                
        except Exception as e:
            logger.warning(f"LLM content cleanup failed: {str(e)}")
    
    # Basic cleanup without regex
    cleaned_content = content.strip()
    
    # Remove common JSON artifacts by checking start/end
    if cleaned_content.startswith('{') and not cleaned_content.endswith('}'):
        cleaned_content = cleaned_content[1:]
    if cleaned_content.endswith('}') and not cleaned_content.startswith('{'):
        cleaned_content = cleaned_content[:-1]
    
    # Remove quotes at start/end
    if cleaned_content.startswith('"') and cleaned_content.endswith('"'):
        cleaned_content = cleaned_content[1:-1]
    
    # Check for minimum content
    if not cleaned_content or len(cleaned_content) < 5:
        cleaned_content = "Unable to process the response. Please try rephrasing your question."
    
    fallback_json = {
        "answer": cleaned_content,
        "sources": [],
        "confidence": 0.5  # Low confidence since this is fallback
    }
    
    return True, fallback_json, None
