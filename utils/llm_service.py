# LLM service for chat completions with caching and optimized batch processing
import asyncio
import logging
import time
import hashlib
import json
import jsonschema
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field

import openai
from openai import AsyncOpenAI

from utils.config_unified import model_config, performance_config

# Setup logging
logger = logging.getLogger(__name__)

# Import prompt utilities
try:
    from utils.prompts.extraction_prompt import generate_extraction_prompt
    from utils.prompts.formatting_prompt import generate_formatting_prompt
    from utils.prompts.system_message import get_system_message
except ImportError:
    # Define fallback functions if prompt modules are not available
    logger.warning("Prompt modules not found, using fallback prompts")
    
    def generate_extraction_prompt(query, context):
        return f"Question: {query}\n\nContext: {context}\n\nAnswer the question based on the context provided."
    
    def generate_formatting_prompt(raw_answer):
        return f"Format the following answer to be clear and well-structured:\n{raw_answer}"
    
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
    
    def _get_cache_key(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
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
    @with_error_tracking()
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
        # Set default values
        model = model or model_config.extraction_model
        max_tokens = max_tokens or model_config.max_tokens
        temperature = temperature if temperature is not None else model_config.temperature
        json_schema = json_schema or ANSWER_SCHEMA
        
        # Input validation
        if not prompt or not isinstance(prompt, str):
            error_msg = f"Invalid prompt: {type(prompt)}"
            logger.error(error_msg)
            return {"content": f"Error: {error_msg}", "error": error_msg, "usage": {"total_tokens": 0}}
            
        if not model or not isinstance(model, str):
            error_msg = f"Invalid model: {model}"
            logger.error(error_msg)
            return {"content": f"Error: {error_msg}", "error": error_msg, "usage": {"total_tokens": 0}}
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, model, max_tokens, temperature)
        cached_response = self._get_from_cache(cache_key)
        if cached_response is not None:
            logger.info(f"Cache hit for prompt hash {cache_key[:8]}")
            return cached_response
        
        # Not in cache, make API call
        start_time = time.time()
        try:
            # Validate API key before making the call
            if not self.api_key or not isinstance(self.api_key, str) or len(self.api_key.strip()) < 10:
                raise ValueError("Invalid API key configuration")
                
            # Check for token limit
            if max_tokens > model_config.max_allowed_tokens:
                logger.warning(f"Requested max_tokens ({max_tokens}) exceeds limit ({model_config.max_allowed_tokens}), capping")
                max_tokens = model_config.max_allowed_tokens
                
            # Make the API call with timeout handling
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
            
            # Format response
            formatted_response = {
                "content": response.choices[0].message.content.strip() if response.choices[0].message.content else "",
                "model": response.model,
                "usage": {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0)
                },
                "id": response.id,
                "created": response.created
            }
            
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
        except openai.RateLimitError as e:
            elapsed_time = time.time() - start_time
            error_msg = f"Rate limit exceeded: {str(e)}"
            logger.error(f"{error_msg} (after {elapsed_time:.2f}s)")
            return {
                "content": f"Error: {error_msg}", 
                "error": error_msg, 
                "error_type": "rate_limit",
                "retry_after": getattr(e, "retry_after", 30),  # Default to 30 seconds if not specified
                "usage": {"total_tokens": 0}
            }
        except openai.APITimeoutError as e:
            elapsed_time = time.time() - start_time
            error_msg = f"API request timed out after {elapsed_time:.2f}s: {str(e)}"
            logger.error(error_msg)
            return {
                "content": f"Error: {error_msg}", 
                "error": error_msg, 
                "error_type": "timeout",
                "usage": {"total_tokens": 0}
            }
        except openai.APIConnectionError as e:
            error_msg = f"API connection error: {str(e)}"
            logger.error(error_msg)
            return {
                "content": f"Error: {error_msg}", 
                "error": error_msg, 
                "error_type": "connection",
                "usage": {"total_tokens": 0}
            }
        except openai.BadRequestError as e:
            error_msg = f"Bad request: {str(e)}"
            logger.error(error_msg)
            return {
                "content": f"Error: {error_msg}", 
                "error": error_msg, 
                "error_type": "bad_request",
                "usage": {"total_tokens": 0}
            }
        except openai.AuthenticationError as e:
            error_msg = f"Authentication error: {str(e)}"
            logger.error(error_msg)
            return {
                "content": f"Error: {error_msg}", 
                "error": error_msg, 
                "error_type": "authentication",
                "usage": {"total_tokens": 0}
            }
        except Exception as e:
            error_msg = f"Error in chat_completion: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "content": f"Error: {error_msg}", 
                "error": error_msg, 
                "error_type": "general",
                "usage": {"total_tokens": 0}
            }
    
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
        # Set default values
        model = model or model_config.extraction_model
        max_tokens = max_tokens or model_config.max_tokens
        temperature = temperature if temperature is not None else model_config.temperature
        json_schema = json_schema or ANSWER_SCHEMA
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, model, max_tokens, temperature)
        cached_response = self._get_from_cache(cache_key)
        if cached_response is not None:
            logger.info(f"Cache hit for prompt hash {cache_key[:8]}")
            return cached_response
        
        # Not in cache, make API call
        start_time = time.time()
        try:
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
            
            # Format response
            formatted_response = {
                "content": response.choices[0].message.content.strip(),
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "id": response.id,
                "created": response.created
            }
            
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
            logger.error(f"Error in async_chat_completion: {str(e)}")
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
        results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
        
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
        
        return [res for res in results if res is not None]
    
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
        # Generate extraction prompt
        extraction_prompt = self.generate_extraction_prompt(query, context)
        
        # Get extraction result with JSON validation
        extraction_response = await self.async_chat_completion(
            prompt=extraction_prompt,
            model=model_config.extraction_model,
            max_tokens=model_config.max_tokens,
            validate_json=True,
            json_schema=ANSWER_SCHEMA
        )
        
        # Check for errors or JSON validation failures
        max_retries = 3
        retry_count = 0
        context_reduction_factor = 0.5  # Start by reducing context by half
        
        while ("error" in extraction_response or "json_validation_error" in extraction_response) and retry_count < max_retries:
            retry_count += 1
            
            # If there was a JSON validation error, try again with reduced context
            if "json_validation_error" in extraction_response and len(context) > 1000:
                logger.warning(f"JSON validation failed (attempt {retry_count}/{max_retries}), retrying with reduced context")
                
                # Calculate how much context to keep based on the retry count
                context_to_keep = int(len(context) * (1 - context_reduction_factor))
                reduced_context = context[:context_to_keep] + f"\n\n[Context truncated to {context_to_keep} characters due to length]\n"
                
                # Generate new extraction prompt with reduced context
                reduced_extraction_prompt = self.generate_extraction_prompt(query, reduced_context)
                
                # Try again with reduced context
                extraction_response = await self.async_chat_completion(
                    prompt=reduced_extraction_prompt,
                    model=model_config.extraction_model,
                    max_tokens=model_config.max_tokens,
                    validate_json=True,
                    json_schema=ANSWER_SCHEMA
                )
                
                # Increase reduction factor for next attempt if needed
                context_reduction_factor += 0.15
            elif "error" in extraction_response:
                # If there's a general error, retry with the same context but with a more explicit JSON instruction
                logger.warning(f"Error in extraction (attempt {retry_count}/{max_retries}), retrying with explicit JSON instructions")
                
                # Add explicit JSON formatting instructions
                enhanced_prompt = extraction_prompt + "\n\nIMPORTANT: Your response MUST be valid JSON with the following structure: {\"answer\": \"your detailed answer\", \"sources\": [\"source1\", \"source2\"], \"confidence\": 0.95}"
                
                # Try again with enhanced prompt
                extraction_response = await self.async_chat_completion(
                    prompt=enhanced_prompt,
                    model=model_config.extraction_model,
                    max_tokens=model_config.max_tokens,
                    validate_json=True,
                    json_schema=ANSWER_SCHEMA
                )
            else:
                # Break the loop if there's no error or JSON validation error
                break
        
        # If we've exhausted retries and still have JSON validation errors, try one last time without validation
        if "json_validation_error" in extraction_response and retry_count >= max_retries:
            logger.warning(f"JSON validation still failed after {max_retries} attempts with reduced context, proceeding without validation")
            
            # Try one last time with minimal context and explicit instructions, but without validation
            minimal_context = context[:int(len(context) * 0.2)] + "\n\n[Context significantly truncated due to processing limitations]\n"
            final_prompt = self.generate_extraction_prompt(query, minimal_context) + "\n\nPlease provide a concise answer based on the limited context available."
            
            extraction_response = await self.async_chat_completion(
                prompt=final_prompt,
                model=model_config.extraction_model,
                max_tokens=model_config.max_tokens,
                validate_json=False  # Disable validation for the final attempt
            )
        
        # If there's still an error, return the error
        if "error" in extraction_response:
            logger.error(f"Extraction failed after {retry_count + 1} attempts: {extraction_response.get('error', 'Unknown error')}")
            return extraction_response, {"error": "Extraction failed after multiple attempts"}
        
        # Generate formatting prompt
        formatting_prompt = self.generate_formatting_prompt(extraction_response["content"])
        
        # Get formatting result
        formatting_response = await self.async_chat_completion(
            prompt=formatting_prompt,
            model=model_config.formatting_model,
            max_tokens=model_config.max_tokens
        )
        
        # If there's an error in formatting, retry once with a simpler prompt
        if "error" in formatting_response:
            logger.warning("Error in formatting, retrying with simpler prompt")
            
            # Simplify the formatting prompt
            simple_formatting_prompt = f"Format the following text to be clear and readable:\n\n{extraction_response['content']}"
            
            # Retry formatting
            formatting_response = await self.async_chat_completion(
                prompt=simple_formatting_prompt,
                model=model_config.formatting_model,
                max_tokens=model_config.max_tokens
            )
        
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
                return {
                    "content": f"Error: {error_msg}",
                    "error": error_msg,
                    "raw_extraction": extraction_response.get("content", ""),
                    "formatted_answer": ""
                }
                
            # Return the formatted answer with metadata
            return {
                "content": formatting_response["content"],
                "raw_extraction": extraction_response["content"],
                "model": formatting_response["model"],
                "usage": {
                    "total_tokens": extraction_response["usage"]["total_tokens"] + formatting_response["usage"]["total_tokens"]
                }
            }
        except Exception as e:
            logger.error(f"Error in generate_answer: {str(e)}", exc_info=True)
            return {"content": f"Error: {str(e)}", "error": str(e)}