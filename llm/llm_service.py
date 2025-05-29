# Enhanced LLM service for chat completions with caching and optimized batch processing
import logging
import time
from typing import Dict, Any, Optional, List, Union, Tuple

import openai
from openai import AsyncOpenAI

from utils.config import model_config, performance_config

# Import extracted modules
from llm.validation import validate_json_output, ANSWER_SCHEMA
from llm.cache_manager import CacheManager
from llm.metrics import MetricsTracker
from llm.prompt_generator import PromptGenerator
from llm.batch_processor import BatchProcessor
from llm.answer_pipeline import AnswerPipeline

# Setup logging
logger = logging.getLogger(__name__)

from error_handling.enhanced_decorators import with_advanced_retry, with_timing, with_error_tracking


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

        # Initialize extracted modules
        self.cache_manager = CacheManager(
            cache_size=performance_config.query_cache_size,
            enable_cache=performance_config.enable_query_cache
        )
        self.metrics_tracker = MetricsTracker()
        self.prompt_generator = PromptGenerator()
        self.batch_processor = BatchProcessor(self, performance_config.llm_batch_size)
        self.answer_pipeline = AnswerPipeline(self, self.prompt_generator)

        logger.info(f"LLM service initialized with cache size {self.cache_manager.cache_size}")

    def generate_extraction_prompt(self, query: str, context: str) -> str:
        """
        Generate a prompt for extracting an answer from context.
        Delegates to PromptGenerator.

        Args:
            query: The user's question
            context: The retrieved context

        Returns:
            A formatted prompt string
        """
        return self.prompt_generator.generate_extraction_prompt(query, context)

    def generate_formatting_prompt(self, raw_answer: str) -> str:
        """
        Generate a prompt for formatting the raw answer.
        Delegates to PromptGenerator.

        Args:
            raw_answer: The raw answer from the extraction model

        Returns:
            A formatted prompt string
        """
        return self.prompt_generator.generate_formatting_prompt(raw_answer)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive service metrics

        Returns:
            Dictionary with service metrics including cache statistics
        """
        cache_stats = self.cache_manager.get_cache_stats()
        return self.metrics_tracker.get_comprehensive_metrics(cache_stats)

    @with_timing(threshold=1.0)
    @with_advanced_retry(
        max_attempts=model_config.retry_attempts, backoff_factor=model_config.retry_backoff
    )
    @with_error_tracking()
    def chat_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        validate_json: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
            return {
                "content": "Error: Invalid prompt",
                "error": error_msg,
                "usage": {"total_tokens": 0},
            }

        # Set default values
        model = model or model_config.extraction_model
        max_tokens = max_tokens or model_config.max_tokens
        temperature = temperature if temperature is not None else model_config.temperature
        json_schema = json_schema or ANSWER_SCHEMA

        # Sanity check: Validate model configuration
        if not model:
            error_msg = "No model specified and no default model configured"
            logger.error(error_msg)
            return {
                "content": "Error: No model specified",
                "error": error_msg,
                "usage": {"total_tokens": 0},
            }

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
        cache_key = self.cache_manager.get_cache_key(prompt, model, max_tokens, float(temperature))
        cached_response = self.cache_manager.get_from_cache(cache_key)
        if cached_response is not None:
            logger.info(f"Cache hit for prompt hash {cache_key[:8]}")
            return cached_response

        # Not in cache, make API call
        start_time = time.time()
        try:
            # Sanity check: Verify API key and client
            if not self.api_key or not hasattr(self, "client") or self.client is None:
                error_msg = "API key not set or client not initialized"
                logger.error(error_msg)
                return {
                    "content": "Error: API configuration issue",
                    "error": error_msg,
                    "usage": {"total_tokens": 0},
                }

            # Get system message
            system_message = self.prompt_generator.get_system_message()

            # Make the API call
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=model_config.timeout,
            )
            usage = response.usage if hasattr(response, "usage") else {}

            # Sanity check: Verify response structure
            if not hasattr(response, "choices") or not response.choices:
                error_msg = "Invalid response structure: missing choices"
                logger.error(error_msg)
                return {
                    "content": "Error: Invalid API response",
                    "error": error_msg,
                    "usage": {"total_tokens": 0},
                }

            if not hasattr(response.choices[0], "message") or not hasattr(
                response.choices[0].message, "content"
            ):
                error_msg = "Invalid response structure: missing message content"
                logger.error(error_msg)
                return {
                    "content": "Error: Invalid API response",
                    "error": error_msg,
                    "usage": {"total_tokens": 0},
                }

            # Format response
            formatted_response = {
                "content": (response.choices[0].message.content or "").strip(),
                "model": response.model,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                },
                "id": response.id,
                "created": response.created,
            }

            # Sanity check: Verify content
            if not formatted_response["content"]:
                logger.warning("Empty content in API response")
                formatted_response["warning"] = "Empty content in API response"

            # Validate JSON if requested
            if validate_json:
                is_valid, parsed_json, error_msg = validate_json_output(
                    formatted_response["content"], schema=json_schema
                )

                if is_valid:
                    formatted_response["parsed_json"] = parsed_json
                else:
                    logger.warning(f"JSON validation failed: {error_msg}")
                    formatted_response["json_validation_error"] = error_msg

            # Update metrics
            self.metrics_tracker.update_metrics(start_time, formatted_response)

            # Add to cache
            self.cache_manager.add_to_cache(cache_key, formatted_response)

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
                import os
                try:
                    from utils.config import resolve_path

                    fallback_log_path = resolve_path("./logs/workapp_errors.log", create_dir=True)
                except ImportError:
                    fallback_log_path = "./logs/workapp_errors.log"
                    os.makedirs(os.path.dirname(fallback_log_path), exist_ok=True)

                with open(fallback_log_path, "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
            return {"content": f"Error: {str(e)}", "error": str(e), "usage": {"total_tokens": 0}}

    @with_timing(threshold=1.0)
    @with_advanced_retry(
        max_attempts=model_config.retry_attempts, backoff_factor=model_config.retry_backoff
    )
    async def async_chat_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        validate_json: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
            return {
                "content": "Error: Invalid prompt",
                "error": error_msg,
                "usage": {"total_tokens": 0},
            }

        # Set default values
        model = model or model_config.extraction_model
        max_tokens = max_tokens or model_config.max_tokens
        temperature = temperature if temperature is not None else model_config.temperature
        json_schema = json_schema or ANSWER_SCHEMA

        # Sanity check: Validate model configuration
        if not model:
            error_msg = "No model specified and no default model configured"
            logger.error(error_msg)
            return {
                "content": "Error: No model specified",
                "error": error_msg,
                "usage": {"total_tokens": 0},
            }

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
        cache_key = self.cache_manager.get_cache_key(prompt, model, max_tokens, float(temperature))
        cached_response = self.cache_manager.get_from_cache(cache_key)
        if cached_response is not None:
            logger.info(f"Cache hit for prompt hash {cache_key[:8]}")
            return cached_response

        # Not in cache, make API call
        start_time = time.time()
        try:
            # Sanity check: Verify API key and client
            if not self.api_key or not hasattr(self, "async_client") or self.async_client is None:
                error_msg = "API key not set or async client not initialized"
                logger.error(error_msg)
                return {
                    "content": "Error: API configuration issue",
                    "error": error_msg,
                    "usage": {"total_tokens": 0},
                }

            # Get system message
            system_message = self.prompt_generator.get_system_message()

            # Make the API call
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=model_config.timeout,
            )
            usage = response.usage if hasattr(response, "usage") else {}

            # Sanity check: Verify response structure
            if not hasattr(response, "choices") or not response.choices:
                error_msg = "Invalid response structure: missing choices"
                logger.error(error_msg)
                return {
                    "content": "Error: Invalid API response",
                    "error": error_msg,
                    "usage": {"total_tokens": 0},
                }

            if not hasattr(response.choices[0], "message") or not hasattr(
                response.choices[0].message, "content"
            ):
                error_msg = "Invalid response structure: missing message content"
                logger.error(error_msg)
                return {
                    "content": "Error: Invalid API response",
                    "error": error_msg,
                    "usage": {"total_tokens": 0},
                }

            # Format response
            formatted_response = {
                "content": (response.choices[0].message.content or "").strip(),
                "model": response.model,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                },
                "id": response.id,
                "created": response.created,
            }

            # Sanity check: Verify content
            if not formatted_response["content"]:
                logger.warning("Empty content in API response")
                formatted_response["warning"] = "Empty content in API response"

            # Validate JSON if requested
            if validate_json:
                is_valid, parsed_json, error_msg = validate_json_output(
                    formatted_response["content"], schema=json_schema
                )

                if is_valid:
                    formatted_response["parsed_json"] = parsed_json
                else:
                    logger.warning(f"JSON validation failed: {error_msg}")
                    formatted_response["json_validation_error"] = error_msg

            # Update metrics
            self.metrics_tracker.update_metrics(start_time, formatted_response)

            # Add to cache
            self.cache_manager.add_to_cache(cache_key, formatted_response)

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
                import os
                try:
                    from utils.config import app_config

                    error_log_path = app_config.error_log_path
                except ImportError:
                    # If config is not available, use a default path
                    error_log_path = "./logs/workapp_errors.log"

                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(error_log_path), exist_ok=True)

                with open(error_log_path, "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
            return {"content": f"Error: {str(e)}", "error": str(e), "usage": {"total_tokens": 0}}

    # Delegate methods to extracted modules
    async def batch_async_chat_completions(
        self, prompts: List[Dict[str, Any]], batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple chat completion requests in parallel with batching.
        Delegates to BatchProcessor.
        """
        return await self.batch_processor.batch_async_chat_completions(prompts, batch_size)

    async def process_extraction_and_formatting(
        self, query: str, context: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process both extraction and formatting in parallel when possible.
        Delegates to AnswerPipeline.
        """
        return await self.answer_pipeline.process_extraction_and_formatting(query, context)

    def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate an answer for a query using the provided context.
        Delegates to AnswerPipeline.
        """
        return self.answer_pipeline.generate_answer(query, context)


# Create a global instance for convenience
llm_service = None
