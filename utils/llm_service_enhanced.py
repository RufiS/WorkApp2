# Enhanced LLM service for chat completions with caching and optimized batch processing
import asyncio
import logging
import time
import hashlib
import json
import jsonschema
import os
from typing import Dict, Any, Optional, List, Union, Tuple, Literal
from dataclasses import dataclass, field

import openai
from openai import AsyncOpenAI

from utils.config_unified import model_config, performance_config

# Setup logging
logger = logging.getLogger(__name__)

# Import prompt utilities
try:
    from utils.prompts.extraction_prompt import generate_extraction_prompt
    from utils.prompts.formatting_prompt_enhanced import generate_formatting_prompt, check_formatting_quality
    from utils.prompts.system_message import get_system_message
except ImportError:
    # Define fallback functions if prompt modules are not available
    logger.warning("Prompt modules not found, using fallback prompts")
    
    def generate_extraction_prompt(query, context):
        return f"Question: {query}\n\nContext: {context}\n\nAnswer the question based on the context provided."
    
    def generate_formatting_prompt(raw_answer):
        return f"Format the following answer to be clear and well-structured:\n{raw_answer}"
    
    def check_formatting_quality(formatted_answer, raw_answer) -> bool:
        return True  # Fallback always returns True
    
    def get_system_message() -> str:
        return "You are a helpful assistant that provides accurate information based on the context provided."
from utils.error_handling.enhanced_decorators import with_advanced_retry, with_timing, with_error_tracking

# Define JSON schema for LLM responses
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {"type": "string"}
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["answer"]
}

def validate_json_output(content: str, schema: Dict[str, Any] = ANSWER_SCHEMA) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Validate LLM output against a JSON schema
    
    Args:
        content: The LLM output string to validate
        schema: JSON schema to validate against
        
    Returns:
        Tuple of (is_valid, parsed_json, error_message)
    """
    # Try to extract JSON from the content if it's not pure JSON
    try:
        # First, try to parse as pure JSON
        parsed_json = json.loads(content)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from markdown code blocks
        import re
        json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        
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
            json_pattern = r'\{[\s\S]*?\}'
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

@dataclass
class LLMCacheEntry:
    """Cache entry for LLM responses"""
    response: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    ttl: float = 3600  # Time to live in seconds (default: 1 hour)
    
    def is_expired(self) -> bool:
        """Check if the cache entry is expired"""
        return time.time() - self.timestamp > self.ttl



class LLMService:
    """Enhanced service for interacting with LLM models"""
    
    def __init__(self, api_key: str):
        """
        Initialize the LLM service
        
        Args:
            api_key: OpenAI API key
            
        Raises:
            ValueError: If the API key is invalid
        """
        if not api_key or not api_key.strip():
            raise ValueError("Invalid API key provided")
            
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.system_message = get_system_message()
        
        # Initialize cache
        self.response_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size = performance_config.query_cache_size
        self.enable_cache = performance_config.enable_query_cache
        
        # Initialize request tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.request_times = []
        self.max_request_times = 100  # Maximum number of request times to store
        
        logger.info(f"LLM service initialized with cache size {self.cache_size}")
    
    def generate_extraction_prompt(self, query: str, context: str) -> str:
        """
        Generate a prompt for extracting an answer from context.
        
        Args:
            query: The user's question
            context: The retrieved context
            
        Returns:
            A formatted prompt string
        """
        # Check if context is too large (approximate token count)
        approx_token_count = len(context.split()) * 1.3  # Rough estimate: 1 token ≈ 0.75 words
        max_context_tokens = 4096
        
        if approx_token_count > max_context_tokens:
            # Log truncation event
            truncation_msg = f"Context too large ({int(approx_token_count)} estimated tokens), truncating to ~{max_context_tokens} tokens"
            logger.warning(truncation_msg)
            # Log to central error log
            try:
                from utils.config_unified import app_config
                from utils.error_logging import log_error
                log_error(truncation_msg, error_type="WARNING", source="LLMService.generate_extraction_prompt")
            except ImportError:
                # Fallback to simple file logging
                try:
                    from utils.config_unified import resolve_path
                    fallback_log_path = resolve_path("./logs/workapp_errors.log", create_dir=True)
                except ImportError:
                    fallback_log_path = "./logs/workapp_errors.log"
                    os.makedirs(os.path.dirname(fallback_log_path), exist_ok=True)
                
                with open(fallback_log_path, "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {truncation_msg}\n")
            
            # Truncate context without splitting sentences
            # First, split by sentences
            import re
            sentences = re.split(r'(?<=[.!?]) +', context)
            
            # Calculate approximate tokens per sentence
            tokens_per_sentence = [len(s.split()) * 1.3 for s in sentences]
            
            # Add sentences until we reach the limit
            truncated_context = []
            current_tokens = 0
            
            for i, (sentence, tokens) in enumerate(zip(sentences, tokens_per_sentence)):
                if current_tokens + tokens <= max_context_tokens:
                    truncated_context.append(sentence)
                    current_tokens += tokens
                else:
                    break
            
            # Join sentences back together
            context = ' '.join(truncated_context)
            
            # Add truncation notice
            context += "\n\n[Note: Context has been truncated due to length limitations.]"
        
        return generate_extraction_prompt(query, context)
        
    def generate_formatting_prompt(self, raw_answer: str) -> str:
        """
        Generate a prompt for formatting the raw answer.
        
        Args:
            raw_answer: The raw answer from the extraction model
            
        Returns:
            A formatted prompt string
        """
        return generate_formatting_prompt(raw_answer)
    
    def _get_cache_key(self, prompt: str, model: str, max_tokens: int, temperature: Optional[float]) -> str:
        """
        Generate a cache key for a request
        
        Args:
            prompt: The prompt string
            model: The model name
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Cache key string
        """
        # Create a dictionary of request parameters
        request_dict = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Convert to JSON and hash
        request_json = json.dumps(request_dict, sort_keys=True)
        return hashlib.md5(request_json.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, response: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """
        Add a response to the cache
        
        Args:
            key: Cache key
            response: Response to cache
            ttl: Time to live in seconds (None for default)
        """
        if not self.enable_cache:
            return
            
        # Create cache entry
        entry = LLMCacheEntry(response=response, ttl=ttl or 3600)
        
        # Add to cache
        self.response_cache[key] = entry
        
        # Trim cache if needed
        if len(self.response_cache) > self.cache_size:
            # Remove oldest entry
            if self.response_cache:
                try:
                    oldest_key = min(self.response_cache.keys(), key=lambda k: self.response_cache[k].timestamp)
                    del self.response_cache[oldest_key]
                except ValueError:
                    # This would happen if the cache became empty during processing
                    # Log the issue and ensure we don't have a memory leak
                    logger.warning("Cache became empty during trimming - possible race condition")
                    
                    # Force trim the cache to prevent memory leaks
                    while len(self.response_cache) > self.cache_size and self.response_cache:
                        # Just remove any key if we can't determine the oldest
                        try:
                            some_key = next(iter(self.response_cache.keys()))
                            del self.response_cache[some_key]
                            logger.info(f"Removed random cache entry to prevent memory leak")
                        except (StopIteration, RuntimeError) as e:
                            # If we still can't trim, clear the cache as a last resort
                            logger.warning(f"Failed to trim cache properly: {str(e)}. Clearing cache.")
                            self.response_cache.clear()
                            break
                    
                    # Log to central error log
                    try:
                        from utils.config_unified import app_config
                        from utils.error_logging import log_error
                        log_error("Cache became empty during trimming - possible race condition", 
                                 error_type="WARNING", 
                                 source="LLMService._add_to_cache")
                    except ImportError:
                        # Fallback to simple file logging
                        try:
                            from utils.config_unified import resolve_path
                            fallback_log_path = resolve_path("./logs/workapp_errors.log", create_dir=True)
                        except ImportError:
                            fallback_log_path = "./logs/workapp_errors.log"
                            os.makedirs(os.path.dirname(fallback_log_path), exist_ok=True)
                        
                        with open(fallback_log_path, "a") as error_log:
                            error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - Cache became empty during trimming - possible race condition\n")
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a response from the cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached response or None if not found
        """
        if not self.enable_cache:
            return None
            
        # Check if key exists in cache
        if key in self.response_cache:
            entry = self.response_cache[key]
            
            # Check if entry is expired
            if entry.is_expired():
                # Remove expired entry
                del self.response_cache[key]
                self.cache_misses += 1
                return None
            
            # Update timestamp to keep entry fresh
            entry.timestamp = time.time()
            self.cache_hits += 1
            return entry.response
        
        self.cache_misses += 1
        return None
    
    def _update_metrics(self, start_time: float, response: Dict[str, Any]) -> None:
        """
        Update request metrics
        
        Args:
            start_time: Request start time
            response: Response from the API
        """
        # Update request count
        self.total_requests += 1
        
        # Update token count
        if "usage" in response and "total_tokens" in response["usage"]:
            self.total_tokens += response["usage"]["total_tokens"]
        
        # Update request times
        request_time = time.time() - start_time
        self.request_times.append(request_time)
        
        # Trim request times if needed
        if len(self.request_times) > self.max_request_times:
            self.request_times = self.request_times[-self.max_request_times:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics
        
        Returns:
            Dictionary with service metrics
        """
        metrics = {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.response_cache),
            "cache_max_size": self.cache_size
        }
        
        # Calculate cache hit rate
        total_cache_accesses = self.cache_hits + self.cache_misses
        if total_cache_accesses > 0:
            metrics["cache_hit_rate"] = self.cache_hits / total_cache_accesses
        else:
            metrics["cache_hit_rate"] = 0.0
        
        # Calculate average request time
        if self.request_times:
            metrics["avg_request_time"] = sum(self.request_times) / len(self.request_times)
            metrics["min_request_time"] = min(self.request_times)
            metrics["max_request_time"] = max(self.request_times)
        else:
            metrics["avg_request_time"] = 0.0
            metrics["min_request_time"] = 0.0
            metrics["max_request_time"] = 0.0
        
        return metrics
    
    @with_timing(threshold=1.0)
    @with_advanced_retry(max_attempts=model_config.retry_attempts, backoff_factor=model_config.retry_backoff)
    def chat_completion(self, 
                       prompt: str, 
                       model: Optional[str] = None, 
                       max_tokens: Optional[int] = None, 
                       temperature: Optional[float] = None,
                       validate_json: bool = False,
                       json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call the specified LLM model for a chat completion with caching.

        Args:
            prompt (str): The prompt to send to the model.
            model (Optional[str]): The model to use (defaults to extraction_model).
            max_tokens (Optional[int]): Maximum tokens to generate.
            temperature (Optional[float]): Temperature for generation.
            validate_json (bool): Whether to validate the response as JSON.
            json_schema (Optional[Dict[str, Any]]): JSON schema to validate against (defaults to ANSWER_SCHEMA).
        Returns:
            Dict[str, Any]: Dictionary containing the model's response and metadata.
        Raises:
            Exception: If the OpenAI API call fails.
        """
        # Sanity check: Validate inputs
        if not prompt or not isinstance(prompt, str):
            error_msg = f"Invalid prompt: {type(prompt)}"
            logger.error(error_msg)
            return {"content": "Error: Invalid prompt", "error": error_msg, "usage": {"total_tokens": 0}}
            
        # Set default values
        model = model or model_config.extraction_model
        max_tokens = max_tokens or model_config.max_tokens
        temperature = temperature if temperature is not None else model_config.temperature
        json_schema = json_schema or ANSWER_SCHEMA
        
        # Sanity check: Validate model configuration
        if not model:
            error_msg = "No model specified and no default model configured"
            logger.error(error_msg)
            return {"content": "Error: No model specified", "error": error_msg, "usage": {"total_tokens": 0}}
            
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            error_msg = f"Invalid max_tokens value: {max_tokens}"
            logger.error(error_msg)
            max_tokens = 1024  # Use a reasonable default
            logger.info(f"Using default max_tokens value: {max_tokens}")
            
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            error_msg = f"Invalid temperature value: {temperature}"
            logger.error(error_msg)
            temperature = 0.7  # Use a reasonable default
            logger.info(f"Using default temperature value: {temperature}")
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, model, max_tokens, float(temperature))
        cached_response = self._get_from_cache(cache_key)
        if cached_response is not None:
            logger.info(f"Cache hit for prompt hash {cache_key[:8]}")
            return cached_response
        
        # Not in cache, make API call
        start_time = time.time()
        try:
            # Sanity check: Verify API key and client
            if not self.api_key or not hasattr(self, 'client') or self.client is None:
                error_msg = "API key not set or client not initialized"
                logger.error(error_msg)
                return {"content": "Error: API configuration issue", "error": error_msg, "usage": {"total_tokens": 0}}
                
            # Sanity check: Verify system message
            if not hasattr(self, 'system_message') or not self.system_message:
                logger.warning("System message not set, using default")
                self.system_message = "You are a helpful assistant that provides accurate information based on the context provided."
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=model_config.timeout
            )
            usage = response.usage if hasattr(response, "usage") else {}
            
            # Sanity check: Verify response structure
            if not hasattr(response, 'choices') or not response.choices:
                error_msg = "Invalid response structure: missing choices"
                logger.error(error_msg)
                return {"content": "Error: Invalid API response", "error": error_msg, "usage": {"total_tokens": 0}}
                
            if not hasattr(response.choices[0], 'message') or not hasattr(response.choices[0].message, 'content'):
                error_msg = "Invalid response structure: missing message content"
                logger.error(error_msg)
                return {"content": "Error: Invalid API response", "error": error_msg, "usage": {"total_tokens": 0}}
            
            # Format response
            formatted_response = {
                "content": (response.choices[0].message.content or "").strip(),
                "model": response.model,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0)
                },
                "id": response.id,
                "created": response.created
            }
            
            # Sanity check: Verify content
            if not formatted_response["content"]:
                logger.warning("Empty content in API response")
                formatted_response["warning"] = "Empty content in API response"
            
            # Validate JSON if requested
            if validate_json:
                is_valid, parsed_json, error_msg = validate_json_output(
                    formatted_response["content"], 
                    schema=json_schema
                )
                
                if is_valid:
                    formatted_response["parsed_json"] = parsed_json
                else:
                    logger.warning(f"JSON validation failed: {error_msg}")
                    formatted_response["json_validation_error"] = error_msg
                    
                    # If validation failed, we might want to retry with a more structured prompt
                    # This would be handled by the caller
            
            # Update metrics
            self._update_metrics(start_time, formatted_response)
            
            # Add to cache
            self._add_to_cache(cache_key, formatted_response)
            
            return formatted_response
        except Exception as e:
            error_msg = f"Error in chat_completion: {str(e)}"
            logger.error(error_msg)
            # Log to central error log
            try:
                from utils.error_logging import log_error
                log_error(error_msg, include_traceback=True)
            except ImportError:
                # Fallback to simple file logging
                try:
                    from utils.config_unified import resolve_path
                    fallback_log_path = resolve_path("./logs/workapp_errors.log", create_dir=True)
                except ImportError:
                    fallback_log_path = "./logs/workapp_errors.log"
                    os.makedirs(os.path.dirname(fallback_log_path), exist_ok=True)
                
                with open(fallback_log_path, "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
            return {"content": f"Error: {str(e)}", "error": str(e), "usage": {"total_tokens": 0}}
    
    @with_timing(threshold=1.0)
    @with_advanced_retry(max_attempts=model_config.retry_attempts, backoff_factor=model_config.retry_backoff)
    async def async_chat_completion(self, 
                                 prompt: str, 
                                 model: Optional[str] = None, 
                                 max_tokens: Optional[int] = None, 
                                 temperature: Optional[float] = None,
                                 validate_json: bool = False,
                                 json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Asynchronously call the specified LLM model for a chat completion with caching.

        Args:
            prompt (str): The prompt to send to the model.
            model (Optional[str]): The model to use (defaults to extraction_model).
            max_tokens (Optional[int]): Maximum tokens to generate.
            temperature (Optional[float]): Temperature for generation.
            validate_json (bool): Whether to validate the response as JSON.
            json_schema (Optional[Dict[str, Any]]): JSON schema to validate against (defaults to ANSWER_SCHEMA).
        Returns:
            Dict[str, Any]: Dictionary containing the model's response and metadata.
        Raises:
            Exception: If the OpenAI API call fails.
        """
        # Sanity check: Validate inputs
        if not prompt or not isinstance(prompt, str):
            error_msg = f"Invalid prompt: {type(prompt)}"
            logger.error(error_msg)
            return {"content": "Error: Invalid prompt", "error": error_msg, "usage": {"total_tokens": 0}}
            
        # Set default values
        model = model or model_config.extraction_model
        max_tokens = max_tokens or model_config.max_tokens
        temperature = temperature if temperature is not None else model_config.temperature
        json_schema = json_schema or ANSWER_SCHEMA
        
        # Sanity check: Validate model configuration
        if not model:
            error_msg = "No model specified and no default model configured"
            logger.error(error_msg)
            return {"content": "Error: No model specified", "error": error_msg, "usage": {"total_tokens": 0}}
            
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            error_msg = f"Invalid max_tokens value: {max_tokens}"
            logger.error(error_msg)
            max_tokens = 1024  # Use a reasonable default
            logger.info(f"Using default max_tokens value: {max_tokens}")
            
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            error_msg = f"Invalid temperature value: {temperature}"
            logger.error(error_msg)
            temperature = 0.7  # Use a reasonable default
            logger.info(f"Using default temperature value: {temperature}")
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, model, max_tokens, float(temperature))
        cached_response = self._get_from_cache(cache_key)
        if cached_response is not None:
            logger.info(f"Cache hit for prompt hash {cache_key[:8]}")
            return cached_response
        
        # Not in cache, make API call
        start_time = time.time()
        try:
            # Sanity check: Verify API key and client
            if not self.api_key or not hasattr(self, 'async_client') or self.async_client is None:
                error_msg = "API key not set or async client not initialized"
                logger.error(error_msg)
                return {"content": "Error: API configuration issue", "error": error_msg, "usage": {"total_tokens": 0}}
                
            # Sanity check: Verify system message
            if not hasattr(self, 'system_message') or not self.system_message:
                logger.warning("System message not set, using default")
                self.system_message = "You are a helpful assistant that provides accurate information based on the context provided."
            
            # Make the API call
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=model_config.timeout
            )
            usage = response.usage if hasattr(response, "usage") else {}
            
            # Sanity check: Verify response structure
            if not hasattr(response, 'choices') or not response.choices:
                error_msg = "Invalid response structure: missing choices"
                logger.error(error_msg)
                return {"content": "Error: Invalid API response", "error": error_msg, "usage": {"total_tokens": 0}}
                
            if not hasattr(response.choices[0], 'message') or not hasattr(response.choices[0].message, 'content'):
                error_msg = "Invalid response structure: missing message content"
                logger.error(error_msg)
                return {"content": "Error: Invalid API response", "error": error_msg, "usage": {"total_tokens": 0}}
            
            # Format response
            formatted_response = {
                "content": (response.choices[0].message.content or "").strip(),
                "model": response.model,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0)
                },
                "id": response.id,
                "created": response.created
            }
            
            # Sanity check: Verify content
            if not formatted_response["content"]:
                logger.warning("Empty content in API response")
                formatted_response["warning"] = "Empty content in API response"
            
            # Validate JSON if requested
            if validate_json:
                is_valid, parsed_json, error_msg = validate_json_output(
                    formatted_response["content"], 
                    schema=json_schema
                )
                
                if is_valid:
                    formatted_response["parsed_json"] = parsed_json
                else
