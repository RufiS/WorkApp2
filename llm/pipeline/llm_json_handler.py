"""
Pure LLM-based JSON handling utilities

Provides robust JSON generation and validation using LLM reasoning
"""
import json
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class LLMJSONHandler:
    """Handles JSON generation and validation using pure LLM approaches"""
    
    def __init__(self, llm_service):
        """
        Initialize with LLM service
        
        Args:
            llm_service: The LLM service instance for generation
        """
        self.llm_service = llm_service
        self.max_retries = 3
        
    def ensure_valid_json(self, content: str, expected_schema: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Ensure content is valid JSON, using LLM to fix if needed
        
        Args:
            content: The content that should be JSON
            expected_schema: Expected JSON schema
            
        Returns:
            Tuple of (success, parsed_json, error_message)
        """
        # First try direct parsing
        try:
            parsed = json.loads(content.strip())
            logger.debug("Direct JSON parsing succeeded")
            return True, parsed, None
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {e}")
            
        # Use LLM to fix the JSON
        for attempt in range(self.max_retries):
            logger.info(f"Attempting LLM JSON repair (attempt {attempt + 1}/{self.max_retries})")
            
            repair_prompt = self._generate_repair_prompt(content, expected_schema, attempt)
            
            try:
                response = self.llm_service.generate(
                    repair_prompt,
                    temperature=0.0,
                    max_tokens=2000
                )
                
                if response:
                    cleaned_response = response.strip()
                    parsed = json.loads(cleaned_response)
                    logger.info("LLM successfully repaired JSON")
                    return True, parsed, None
                    
            except Exception as e:
                logger.warning(f"LLM JSON repair attempt {attempt + 1} failed: {e}")
                
        # If all repairs fail, create a fallback
        logger.warning("All JSON repair attempts failed, using fallback")
        fallback = self._create_fallback_json(content)
        return True, fallback, None
        
    def generate_json_response(self, prompt: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Generate a JSON response using LLM with the given schema
        
        Args:
            prompt: The prompt to send to LLM
            schema: Expected JSON schema
            
        Returns:
            Tuple of (success, json_response, error_message)
        """
        enhanced_prompt = self._enhance_prompt_for_json(prompt, schema)
        
        try:
            response = self.llm_service.generate(
                enhanced_prompt,
                temperature=0.0,
                max_tokens=2000
            )
            
            if response:
                return self.ensure_valid_json(response, schema)
                
        except Exception as e:
            logger.error(f"Failed to generate JSON response: {e}")
            
        return False, None, "Failed to generate JSON response"
        
    def _generate_repair_prompt(self, content: str, schema: Dict[str, Any], attempt: int) -> str:
        """Generate a repair prompt based on attempt number"""
        
        if attempt == 0:
            return f"""
Fix the following malformed JSON. Return ONLY valid JSON:

{content}

Expected format:
{{
  "answer": "the answer text",
  "sources": ["source1", "source2"],
  "confidence": 0.95
}}

Valid JSON:
"""
        elif attempt == 1:
            return f"""
The following text contains JSON but it's malformed. Extract and fix it.
RESPOND WITH ONLY THE JSON, NOTHING ELSE:

{content}

Return ONLY this structure:
{json.dumps({"answer": "...", "sources": ["..."], "confidence": 0.95}, indent=2)}
"""
        else:
            return f"""
CRITICAL: Extract any answer from this text and return as JSON:

{content}

If you can find an answer, return:
{{"answer": "the answer you found", "sources": [], "confidence": 0.5}}

If no answer found, return:
{{"answer": "Unable to process response", "sources": [], "confidence": 0.0}}

JSON ONLY:
"""
            
    def _enhance_prompt_for_json(self, prompt: str, schema: Dict[str, Any]) -> str:
        """Enhance a prompt to ensure JSON output"""
        
        example = {
            "answer": "Example answer text here",
            "sources": ["Example source 1", "Example source 2"],
            "confidence": 0.95
        }
        
        return f"""{prompt}

CRITICAL: Respond with ONLY valid JSON in this exact format:
{json.dumps(example, indent=2)}

Your JSON response:
"""
        
    def _create_fallback_json(self, content: str) -> Dict[str, Any]:
        """Create a fallback JSON response from any content"""
        
        # Try to extract something meaningful from the content
        cleaned = content.strip()
        
        # Remove common JSON artifacts
        for artifact in ['{', '}', '"', "'", 'answer:', 'response:']:
            cleaned = cleaned.replace(artifact, ' ')
            
        # Collapse multiple spaces
        cleaned = ' '.join(cleaned.split())
        
        if len(cleaned) < 10:
            cleaned = "Unable to process the response. Please try rephrasing your question."
            
        return {
            "answer": cleaned,
            "sources": [],
            "confidence": 0.3
        }
        
    def validate_response_format(self, response: Dict[str, Any]) -> bool:
        """
        Validate that response has required fields
        
        Args:
            response: The response dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["answer"]
        
        for field in required_fields:
            if field not in response:
                logger.warning(f"Response missing required field: {field}")
                return False
                
        # Validate types
        if not isinstance(response.get("answer"), str):
            logger.warning("Answer field is not a string")
            return False
            
        if "sources" in response and not isinstance(response["sources"], list):
            logger.warning("Sources field is not a list")
            return False
            
        if "confidence" in response:
            conf = response["confidence"]
            if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                logger.warning(f"Invalid confidence value: {conf}")
                return False
                
        return True
        
    def extract_answer_text(self, json_response: Dict[str, Any]) -> str:
        """
        Extract answer text from JSON response
        
        Args:
            json_response: The JSON response
            
        Returns:
            The answer text
        """
        return json_response.get("answer", "No answer found")
