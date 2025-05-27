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
                with open("./logs/workapp_errors.log", "a") as error_log:
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
                        with open("./logs/workapp_errors.log", "a") as error_log:
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
                # Fallback to simple file logging if error_logging module is not available
                try:
                    from utils.config_unified import app_config
                    error_log_path = app_config.error_log_path
                except ImportError:
                    # If config is not available, use a default path
                    error_log_path = "./logs/workapp_errors.log"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                
                with open(error_log_path, "a") as error_log:
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
            error_msg = f"Error in async_chat_completion: {str(e)}"
            logger.error(error_msg)
            # Log to central error log
            try:
                from utils.error_logging import log_error
                log_error(error_msg, include_traceback=True)
            except ImportError:
                # Fallback to simple file logging if error_logging module is not available
                try:
                    from utils.config_unified import app_config
                    error_log_path = app_config.error_log_path
                except ImportError:
                    # If config is not available, use a default path
                    error_log_path = "./logs/workapp_errors.log"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                
                with open(error_log_path, "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
            return {"content": f"Error: {str(e)}", "error": str(e), "usage": {"total_tokens": 0}}
    
    async def batch_async_chat_completions(self, 
                                        prompts: List[Dict[str, Any]],
                                        batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process multiple chat completion requests in parallel with batching.
        
        Args:
            prompts: List of dictionaries with prompt details:
                     [{"prompt": str, "model": str, "max_tokens": int, "temperature": float}]            batch_size: Maximum number of concurrent requests to make (None for default)                     
        Returns:
            List of response dictionaries in the same order as the input prompts
        """
        # Use configured batch size if not specified
        batch_size = batch_size or performance_config.llm_batch_size
        results = []
        
        # Check cache first for all prompts
        cache_hits = []
        cache_misses = []
        
        for i, p in enumerate(prompts):
            model = p.get("model") or model_config.extraction_model
            max_tokens = p.get("max_tokens") or model_config.max_tokens
            temperature = p.get("temperature") if p.get("temperature") is not None else model_config.temperature
            
            cache_key = self._get_cache_key(p["prompt"], model, max_tokens, temperature)
            cached_response = self._get_from_cache(cache_key)
            
            if cached_response is not None:
                # Cache hit
                cache_hits.append((i, cached_response))
            else:
                # Cache miss
                cache_misses.append((i, p))
        
        # Initialize results list with None values
        results: List[Dict[str, Any]] = [{} for _ in prompts]
        
        # Fill in cache hits
        for i, response in cache_hits:
            results[i] = response
        
        # Process cache misses in batches
        if cache_misses:
            logger.info(f"Processing {len(cache_misses)} cache misses in batches of {batch_size}")
            
            # Group by batch
            for i in range(0, len(cache_misses), batch_size):
                batch = cache_misses[i:i+batch_size]
                
                # Create tasks for the current batch
                tasks = []
                for _, p in batch:
                    task = self.async_chat_completion(
                        prompt=p["prompt"],
                        model=p.get("model"),
                        max_tokens=p.get("max_tokens"),
                        temperature=p.get("temperature")
                    )
                    tasks.append(task)
                
                # Run current batch concurrently
                batch_results = await asyncio.gather(*tasks)
                
                # Fill in results
                for j, (idx, _) in enumerate(batch):
                    results[idx] = batch_results[j]
                
                # Add a small delay between batches to avoid rate limiting
                if i + batch_size < len(cache_misses):
                    await asyncio.sleep(0.5)
        
        return results
    
    async def process_extraction_and_formatting(self, 
                                               query: str, 
                                               context: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process both extraction and formatting in parallel when possible
        
        Args:
            query: The user's question
            context: The retrieved context
            
        Returns:
            Tuple of (extraction_response, formatting_response)
        """
        # Check for empty context
        if not context or context.strip() == "":
            error_msg = f"Empty context provided for query: {query}"
            logger.error(error_msg)
            # Log to central error log
            try:
                from utils.config_unified import app_config
                from utils.error_logging import log_error
                log_error(error_msg, error_type="WARNING", source="LLMService.process_extraction_and_formatting")
            except ImportError:
                # Fallback to simple file logging
                with open("./logs/workapp_errors.log", "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {error_msg}\n")
            return {"error": "Empty context provided"}, {"error": "Extraction failed due to empty context"}
        
        # Log context length
        logger.info(f"Processing extraction with context length: {len(context)} characters")
        
        # Generate extraction prompt
        extraction_prompt = self.generate_extraction_prompt(query, context)
        
        # Get extraction result
        start_time = time.time()
        extraction_response = None
        extraction_time = 0
        
        # Enhanced error handling for API calls
        try:
            extraction_response = await self.async_chat_completion(
                prompt=extraction_prompt,
                model=model_config.extraction_model,
                max_tokens=model_config.max_tokens,
                validate_json=True,
                json_schema=ANSWER_SCHEMA
            )
            extraction_time = time.time() - start_time
            logger.info(f"Extraction completed in {extraction_time:.2f} seconds")
        except asyncio.TimeoutError:
            error_msg = "Timeout occurred during extraction API call"
            logger.error(error_msg)
            try:
                from utils.error_logging import log_error
                log_error(error_msg, error_type="ERROR", source="LLMService.process_extraction_and_formatting")
            except ImportError:
                # Fallback logging
                with open("./logs/workapp_errors.log", "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
            extraction_response = {"error": error_msg, "content": ""}
        except openai.RateLimitError as e:
            error_msg = f"Rate limit exceeded during extraction: {str(e)}"
            logger.error(error_msg)
            try:
                from utils.error_logging import log_error
                log_error(error_msg, error_type="ERROR", source="LLMService.process_extraction_and_formatting")
            except ImportError:
                # Fallback logging
                with open("./logs/workapp_errors.log", "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
            extraction_response = {"error": error_msg, "content": ""}
        except openai.APIConnectionError as e:
            error_msg = f"API connection error during extraction: {str(e)}"
            logger.error(error_msg)
            try:
                from utils.error_logging import log_error
                log_error(error_msg, error_type="ERROR", source="LLMService.process_extraction_and_formatting")
            except ImportError:
                # Fallback logging
                with open("./logs/workapp_errors.log", "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
            extraction_response = {"error": error_msg, "content": ""}
        except Exception as e:
            error_msg = f"Unexpected error during extraction: {str(e)}"
            logger.error(error_msg, exc_info=True)
            try:
                from utils.error_logging import log_error
                log_error(error_msg, error_type="ERROR", include_traceback=True, source="LLMService.process_extraction_and_formatting")
            except ImportError:
                # Fallback logging
                with open("./logs/workapp_errors.log", "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
                    import traceback
                    error_log.write(traceback.format_exc())
            extraction_response = {"error": error_msg, "content": ""}
        
        # Check for errors or JSON validation failures
        max_retries = 3
        retry_count = 0
        context_reduction_factor = 0.5  # Start by reducing context by half
        backoff_time = 1.0  # Initial backoff time in seconds
        
        while ("error" in extraction_response or "json_validation_error" in extraction_response) and retry_count < max_retries:
            retry_count += 1
            
            # Apply exponential backoff for rate limit errors
            if "error" in extraction_response and "rate limit" in extraction_response["error"].lower():
                backoff_seconds = backoff_time * (2 ** (retry_count - 1))  # Exponential backoff
                logger.warning(f"Rate limit error, backing off for {backoff_seconds:.2f} seconds before retry {retry_count}/{max_retries}")
                await asyncio.sleep(backoff_seconds)
            
            # If there was a JSON validation error, try again with reduced context
            if "json_validation_error" in extraction_response and len(context) > 1000:
                logger.warning(f"JSON validation failed (attempt {retry_count}/{max_retries}), retrying with reduced context")
                # Log to central error log
                try:
                    from utils.config_unified import app_config
                    from utils.error_logging import log_error
                    log_error(f"JSON validation failed (attempt {retry_count}/{max_retries}), retrying with reduced context", 
                             error_type="WARNING", 
                             source="LLMService.process_extraction_and_formatting")
                except ImportError:
                    # Fallback to simple file logging
                    try:
                        from utils.config_unified import app_config
                        error_log_path = app_config.error_log_path
                    except ImportError:
                        # If config is not available, use a default path
                        error_log_path = "./logs/workapp_errors.log"
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                    
                    with open(error_log_path, "a") as error_log:
                        error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - JSON validation failed (attempt {retry_count}/{max_retries}), retrying with reduced context\n")
                
                # Calculate how much context to keep based on the retry count
                context_to_keep = int(len(context) * (1 - context_reduction_factor))
                reduced_context = context[:context_to_keep] + f"\n\n[Context truncated to {context_to_keep} characters due to length]\n"
                
                # Generate new extraction prompt with reduced context
                reduced_extraction_prompt = self.generate_extraction_prompt(query, reduced_context)
                
                # Try again with reduced context
                try:
                    start_time = time.time()
                    extraction_response = await self.async_chat_completion(
                        prompt=reduced_extraction_prompt,
                        model=model_config.extraction_model,
                        max_tokens=model_config.max_tokens,
                        validate_json=True,
                        json_schema=ANSWER_SCHEMA
                    )
                    extraction_time = time.time() - start_time
                    logger.info(f"Retry extraction {retry_count} completed in {extraction_time:.2f} seconds")
                except (asyncio.TimeoutError, openai.RateLimitError, openai.APIConnectionError) as e:
                    error_type = "Timeout" if isinstance(e, asyncio.TimeoutError) else \
                                "Rate limit" if isinstance(e, openai.RateLimitError) else "API connection error"
                    error_msg = f"{error_type} during retry {retry_count}: {str(e)}"
                    logger.error(error_msg)
                    try:
                        from utils.error_logging import log_error
                        log_error(error_msg, error_type="ERROR", source="LLMService.process_extraction_and_formatting")
                    except ImportError:
                        # Fallback logging
                        with open("./logs/workapp_errors.log", "a") as error_log:
                            error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
                    extraction_response = {"error": error_msg, "content": ""}
                    
                    # For rate limit errors, increase backoff time
                    if isinstance(e, openai.RateLimitError):
                        backoff_seconds = backoff_time * (2 ** retry_count)  # Increase backoff for next attempt
                        logger.warning(f"Rate limit error, backing off for {backoff_seconds:.2f} seconds")
                        await asyncio.sleep(backoff_seconds)
                except Exception as e:
                    error_msg = f"Unexpected error during retry {retry_count}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    try:
                        from utils.error_logging import log_error
                        log_error(error_msg, error_type="ERROR", include_traceback=True, source="LLMService.process_extraction_and_formatting")
                    except ImportError:
                        # Fallback logging
                        with open("./logs/workapp_errors.log", "a") as error_log:
                            error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
                            import traceback
                            error_log.write(traceback.format_exc())
                    extraction_response = {"error": error_msg, "content": ""}
                
                # Increase reduction factor for next attempt if needed
                context_reduction_factor += 0.15
            elif "error" in extraction_response:
                # If there's a general error, retry with the same context but with a more explicit JSON instruction
                error_msg = f"Error in extraction (attempt {retry_count}/{max_retries}): {extraction_response.get('error', 'Unknown error')}, retrying with explicit JSON instructions"
                logger.warning(error_msg)
                # Log to central error log
                try:
                    from utils.error_logging import log_error
                    log_error(error_msg, error_type="WARNING", source="LLMService.process_extraction_and_formatting")
                except ImportError:
                    # Fallback to simple file logging
                    try:
                        from utils.config_unified import app_config
                        error_log_path = app_config.error_log_path
                    except ImportError:
                        # If config is not available, use a default path
                        error_log_path = "./logs/workapp_errors.log"
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                    
                    with open(error_log_path, "a") as error_log:
                        error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {error_msg}\n")
                
                # Add explicit JSON formatting instructions
                enhanced_prompt = extraction_prompt + "\n\nIMPORTANT: Your response MUST be valid JSON with the following structure: {\"answer\": \"your detailed answer\", \"sources\": [\"source1\", \"source2\"], \"confidence\": 0.95}"
                
                # Try again with enhanced prompt
                try:
                    start_time = time.time()
                    extraction_response = await self.async_chat_completion(
                        prompt=enhanced_prompt,
                        model=model_config.extraction_model,
                        max_tokens=model_config.max_tokens,
                        validate_json=True,
                        json_schema=ANSWER_SCHEMA
                    )
                    extraction_time = time.time() - start_time
                    logger.info(f"Retry extraction {retry_count} with explicit JSON instructions completed in {extraction_time:.2f} seconds")
                except (asyncio.TimeoutError, openai.RateLimitError, openai.APIConnectionError) as e:
                    error_type = "Timeout" if isinstance(e, asyncio.TimeoutError) else \
                                "Rate limit" if isinstance(e, openai.RateLimitError) else "API connection error"
                    error_msg = f"{error_type} during retry {retry_count} with explicit JSON: {str(e)}"
                    logger.error(error_msg)
                    try:
                        from utils.error_logging import log_error
                        log_error(error_msg, error_type="ERROR", source="LLMService.process_extraction_and_formatting")
                    except ImportError:
                        # Fallback logging
                        with open("./logs/workapp_errors.log", "a") as error_log:
                            error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
                    extraction_response = {"error": error_msg, "content": ""}
                    
                    # For rate limit errors, increase backoff time
                    if isinstance(e, openai.RateLimitError):
                        backoff_seconds = backoff_time * (2 ** retry_count)  # Increase backoff for next attempt
                        logger.warning(f"Rate limit error, backing off for {backoff_seconds:.2f} seconds")
                        await asyncio.sleep(backoff_seconds)
                except Exception as e:
                    error_msg = f"Unexpected error during retry {retry_count} with explicit JSON: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    try:
                        from utils.error_logging import log_error
                        log_error(error_msg, error_type="ERROR", include_traceback=True, source="LLMService.process_extraction_and_formatting")
                    except ImportError:
                        # Fallback logging
                        with open("./logs/workapp_errors.log", "a") as error_log:
                            error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
                            import traceback
                            error_log.write(traceback.format_exc())
                    extraction_response = {"error": error_msg, "content": ""}
            else:
                # Break the loop if there's no error or JSON validation error
                break
        
        # If we've exhausted retries and still have JSON validation errors, try one last time without validation
        if ("json_validation_error" in extraction_response or "error" in extraction_response) and retry_count >= max_retries:
            error_msg = f"JSON validation or extraction still failed after {max_retries} attempts, proceeding with final attempt using minimal context"
            logger.warning(error_msg)
            # Log to central error log
            try:
                from utils.error_logging import log_error
                log_error(error_msg, error_type="WARNING", source="LLMService.process_extraction_and_formatting")
            except ImportError:
                # Fallback to simple file logging
                try:
                    from utils.config_unified import app_config
                    error_log_path = app_config.error_log_path
                except ImportError:
                    # If config is not available, use a default path
                    error_log_path = "./logs/workapp_errors.log"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                
                with open(error_log_path, "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {error_msg}\n")
            
            # Try one last time with minimal context and explicit instructions, but without validation
            minimal_context = context[:int(len(context) * 0.2)] + "\n\n[Context significantly truncated due to processing limitations]\n"
            final_prompt = self.generate_extraction_prompt(query, minimal_context) + "\n\nPlease provide a concise answer based on the limited context available."
            
            try:
                start_time = time.time()
                extraction_response = await self.async_chat_completion(
                    prompt=final_prompt,
                    model=model_config.extraction_model,
                    max_tokens=model_config.max_tokens,
                    validate_json=False  # Disable validation for the final attempt
                )
                extraction_time = time.time() - start_time
                logger.info(f"Final extraction attempt with minimal context completed in {extraction_time:.2f} seconds")
            except Exception as e:
                # For the final attempt, if it still fails, create a fallback response
                error_msg = f"Final extraction attempt failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                try:
                    from utils.error_logging import log_error
                    log_error(error_msg, error_type="ERROR", include_traceback=True, source="LLMService.process_extraction_and_formatting")
                except ImportError:
                    # Fallback logging
                    with open("./logs/workapp_errors.log", "a") as error_log:
                        error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
                        import traceback
                        error_log.write(traceback.format_exc())
                
                # Create a fallback response with an error message
                extraction_response = {
                    "content": "I'm sorry, but I encountered an issue while processing your query. Please try again or rephrase your question.",
                    "error": error_msg,
                    "model": model_config.extraction_model,
                    "usage": {"total_tokens": 0}
                }
        
        # Check for empty or very short extraction result
        if not extraction_response.get("content") or len(extraction_response.get("content", "")) < 10:
            error_msg = f"Extraction returned empty or very short result: '{extraction_response.get('content', '')}'"
            logger.warning(error_msg)
            # Log to central error log
            try:
                from utils.error_logging import log_error
                log_error(error_msg, error_type="WARNING", source="LLMService.process_extraction_and_formatting")
            except ImportError:
                # Fallback to simple file logging
                try:
                    from utils.config_unified import app_config
                    error_log_path = app_config.error_log_path
                except ImportError:
                    # If config is not available, use a default path
                    error_log_path = "./logs/workapp_errors.log"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                
                with open(error_log_path, "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {error_msg}\n")
            
            # If content is empty but no error is set, add a generic error
            if "error" not in extraction_response:
                extraction_response["error"] = "Empty or very short extraction result"
            
            # Ensure there's at least some content for formatting
            if not extraction_response.get("content"):
                extraction_response["content"] = "No answer could be extracted from the provided context."
        
        # Generate formatting prompt
        formatting_prompt = self.generate_formatting_prompt(extraction_response.get("content", ""))
        
        # Get formatting result
        start_time = time.time()
        formatting_response = None
        
        try:
            formatting_response = await self.async_chat_completion(
                prompt=formatting_prompt,
                model=model_config.formatting_model,
                max_tokens=model_config.max_tokens
            )
            formatting_time = time.time() - start_time
            logger.info(f"Formatting completed in {formatting_time:.2f} seconds")
        except (asyncio.TimeoutError, openai.RateLimitError, openai.APIConnectionError) as e:
            error_type = "Timeout" if isinstance(e, asyncio.TimeoutError) else \
                        "Rate limit" if isinstance(e, openai.RateLimitError) else "API connection error"
            error_msg = f"{error_type} during formatting: {str(e)}"
            logger.error(error_msg)
            try:
                from utils.error_logging import log_error
                log_error(error_msg, error_type="ERROR", source="LLMService.process_extraction_and_formatting")
            except ImportError:
                # Fallback logging
                with open("./logs/workapp_errors.log", "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
            formatting_response = {"error": error_msg, "content": extraction_response.get("content", "")}
        except Exception as e:
            error_msg = f"Unexpected error during formatting: {str(e)}"
            logger.error(error_msg, exc_info=True)
            try:
                from utils.error_logging import log_error
                log_error(error_msg, error_type="ERROR", include_traceback=True, source="LLMService.process_extraction_and_formatting")
            except ImportError:
                # Fallback logging
                with open("./logs/workapp_errors.log", "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
                    import traceback
                    error_log.write(traceback.format_exc())
            formatting_response = {"error": error_msg, "content": extraction_response.get("content", "")}
        
        # Check for errors in formatting and retry if needed
        if "error" in formatting_response:
            error_msg = f"Formatting failed: {formatting_response['error']}, retrying with simpler prompt"
            logger.warning(error_msg)
            # Log to central error log
            try:
                from utils.error_logging import log_error
                log_error(error_msg, error_type="WARNING", source="LLMService.process_extraction_and_formatting")
            except ImportError:
                # Fallback to simple file logging
                try:
                    from utils.config_unified import app_config
                    error_log_path = app_config.error_log_path
                except ImportError:
                    # If config is not available, use a default path
                    error_log_path = "./logs/workapp_errors.log"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                
                with open(error_log_path, "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {error_msg}\n")
            
            # Simplify the formatting prompt
            simple_formatting_prompt = f"Format the following text to be clear and readable:\n\n{extraction_response['content']}"
            
            # Retry formatting
            start_time = time.time()
            formatting_response = await self.async_chat_completion(
                prompt=simple_formatting_prompt,
                model=model_config.formatting_model,
                max_tokens=model_config.max_tokens
            )
            formatting_time = time.time() - start_time
            logger.info(f"Retry formatting with simpler prompt completed in {formatting_time:.2f} seconds")
            
            # If still failing, try with an even simpler approach
            if "error" in formatting_response:
                error_msg = "Formatting still failed with simpler prompt, using raw extraction result"
                logger.warning(error_msg)
                # Log to central error log
                try:
                    from utils.error_logging import log_error
                    log_error(error_msg, error_type="WARNING", source="LLMService.process_extraction_and_formatting")
                except ImportError:
                    # Fallback to simple file logging
                    try:
                        from utils.config_unified import app_config
                        error_log_path = app_config.error_log_path
                    except ImportError:
                        # If config is not available, use a default path
                        error_log_path = "./logs/workapp_errors.log"
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                    
                    with open(error_log_path, "a") as error_log:
                        error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {error_msg}\n")
                
                # Create a minimal formatting response using the raw extraction
                formatting_response = {
                    "content": extraction_response["content"],
                    "model": extraction_response["model"],
                    "usage": extraction_response["usage"],
                    "warning": "Formatting failed, using raw extraction result"
                }
        
        # Check for empty or very short formatting result
        if not formatting_response.get("content") or len(formatting_response.get("content", "")) < 10:
            error_msg = f"Formatting returned empty or very short result: '{formatting_response.get('content', '')}'"
            logger.warning(error_msg)
            # Log to central error log
            try:
                from utils.error_logging import log_error
                log_error(error_msg, error_type="WARNING", source="LLMService.process_extraction_and_formatting")
            except ImportError:
                # Fallback to simple file logging
                try:
                    from utils.config_unified import app_config
                    error_log_path = app_config.error_log_path
                except ImportError:
                    # If config is not available, use a default path
                    error_log_path = "./logs/workapp_errors.log"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                
                with open(error_log_path, "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {error_msg}\n")
        
        # Check formatting quality
        try:
            if formatting_response.get("content") and extraction_response.get("content"):
                quality_check = check_formatting_quality(formatting_response["content"], extraction_response["content"])
                if not quality_check:
                    warning_msg = "Formatting quality check failed - formatted answer may be missing key sections"
                    logger.warning(warning_msg)
                    # Log to central error log
                    try:
                        from utils.error_logging import log_error
                        log_error(warning_msg, error_type="WARNING", source="LLMService.process_extraction_and_formatting")
                    except ImportError:
                        # Fallback to simple file logging
                        try:
                            from utils.config_unified import app_config
                            error_log_path = app_config.error_log_path
                        except ImportError:
                            # If config is not available, use a default path
                            error_log_path = "./logs/workapp_errors.log"
                        
                        # Create directory if it doesn't exist
                        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                        
                        with open(error_log_path, "a") as error_log:
                            error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {warning_msg}\n")
                    # Add warning to the formatting response
                    formatting_response["warning"] = warning_msg
        except Exception as e:
            logger.error(f"Error in formatting quality check: {str(e)}")
            # Don't fail the whole process if quality check fails
        
        return extraction_response, formatting_response
        
    def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate an answer for a query using the provided context.
        This is a synchronous wrapper around the async extraction process.
        
        Args:
            query: The user's question
            context: The retrieved context
            
        Returns:
            Dictionary containing the answer and metadata
        """
        # Create an event loop if one doesn't exist
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            # Run the async extraction and formatting process
            extraction_response, formatting_response = loop.run_until_complete(
                self.process_extraction_and_formatting(query, context)
            )
            
            # Check for errors
            if "error" in extraction_response or "error" in formatting_response:
                error_msg = extraction_response.get("error", formatting_response.get("error", "Unknown error"))
                logger.error(f"Error generating answer: {error_msg}")
                # Log to central error log
                try:
                    from utils.error_logging import log_error
                    log_error(f"Error generating answer: {error_msg}", error_type="ERROR", source="LLMService.generate_answer")
                except ImportError:
                    # Fallback to simple file logging
                    try:
                        from utils.config_unified import app_config
                        error_log_path = app_config.error_log_path
                    except ImportError:
                        # If config is not available, use a default path
                        error_log_path = "./logs/workapp_errors.log"
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                    
                    with open(error_log_path, "a") as error_log:
                        error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - Error generating answer: {error_msg}\n")
                return {
                    "content": f"Error: {error_msg}",
                    "error": error_msg,
                    "raw_extraction": extraction_response.get("content", ""),
                    "formatted_answer": ""
                }
                
            # Return the formatted answer with metadata
            # Log the raw answer before formatting
            logger.info(f"Raw extraction answer: {extraction_response['content'][:100]}...")
            # Log the final formatted answer
            logger.info(f"Final formatted answer: {formatting_response['content'][:100]}...")
            
            # Log detailed answers for analysis
            try:
                from utils.config_unified import app_config
                raw_log_path = os.path.join(os.path.dirname(app_config.error_log_path), "workapp_raw_answers.log")
            except ImportError:
                # If config is not available, use a default path
                raw_log_path = "./logs/workapp_raw_answers.log"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(raw_log_path), exist_ok=True)
            
            with open(raw_log_path, "a") as raw_log:
                raw_log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                raw_log.write(f"Query: {query}\n")
                raw_log.write(f"Raw extraction: {extraction_response['content']}\n")
                raw_log.write(f"Formatted answer: {formatting_response['content']}\n")
                raw_log.write("---\n")
            
            return {
                "content": formatting_response["content"],
                "raw_extraction": extraction_response["content"],
                "model": formatting_response["model"],
                "usage": {
                    "total_tokens": extraction_response["usage"]["total_tokens"] + formatting_response["usage"]["total_tokens"]
                }
            }
        except Exception as e:
            error_msg = f"Error in generate_answer: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Log to central error log
            try:
                from utils.error_logging import log_error
                log_error(error_msg, include_traceback=True, source="LLMService.generate_answer")
            except ImportError:
                # Fallback to simple file logging
                try:
                    from utils.config_unified import app_config
                    error_log_path = app_config.error_log_path
                except ImportError:
                    # If config is not available, use a default path
                    error_log_path = "./logs/workapp_errors.log"
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
                
                import traceback
                with open(error_log_path, "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
                    error_log.write(traceback.format_exc())
                    error_log.write("\n")
            return {"content": f"Error: {str(e)}", "error": str(e)}

# Create a global instance for convenience
llm_service = None