"""
Optimized LLM Service with connection pooling and performance improvements
Reduces HTTP overhead and improves response times
"""

import asyncio
import logging
import time
import httpx
from typing import Dict, Any, Optional

import openai
from openai import AsyncOpenAI

from core.config import model_config, performance_config
from llm.services.cache_manager import CacheManager
from llm.metrics import MetricsTracker
from llm.services.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)


class OptimizedLLMService:
    """Enhanced LLM service with connection optimization and performance improvements"""

    def __init__(self, api_key: str):
        """
        Initialize optimized LLM service with connection pooling

        Args:
            api_key: OpenAI API key
        """
        if not api_key or not api_key.strip():
            raise ValueError("Invalid API key provided")

        self.api_key = api_key
        
        # Create optimized HTTP clients for better connection management
        self._http_client = self._create_optimized_http_client()
        self._async_http_client = self._create_optimized_async_http_client()
        
        # Initialize OpenAI clients with optimized HTTP clients
        self.client = openai.OpenAI(
            api_key=api_key,
            http_client=self._http_client
        )
        
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            http_client=self._async_http_client
        )

        # Initialize performance components
        self.cache_manager = CacheManager(
            cache_size=performance_config.query_cache_size,
            enable_cache=performance_config.enable_query_cache
        )
        self.metrics_tracker = MetricsTracker()
        self.batch_processor = BatchProcessor(self, performance_config.llm_batch_size)

        # Performance tracking
        self.connection_stats = {
            "requests_made": 0,
            "cache_hits": 0,
            "total_response_time_ms": 0,
            "fastest_response_ms": float('inf'),
            "slowest_response_ms": 0,
            "connection_reuses": 0
        }

        logger.info(f"✅ Optimized LLM service initialized with connection pooling")

    def _create_optimized_http_client(self) -> httpx.Client:
        """
        Create an optimized HTTP client with connection pooling and timeouts

        Returns:
            Configured HTTPX client
        """
        # Connection limits for optimal performance
        limits = httpx.Limits(
            max_keepalive_connections=20,  # Keep connections alive
            max_connections=50,            # Total connection pool size
            keepalive_expiry=30.0         # Keep connections alive for 30 seconds
        )

        # Optimized timeouts
        timeout = httpx.Timeout(
            connect=10.0,     # Connection timeout
            read=45.0,        # Read timeout (reduced from default 60s)
            write=10.0,       # Write timeout
            pool=5.0          # Pool timeout
        )

        # Create client with optimization
        return httpx.Client(
            limits=limits,
            timeout=timeout,
            http2=False,      # Disable HTTP/2 for compatibility
            follow_redirects=True
        )

    def _create_optimized_async_http_client(self) -> httpx.AsyncClient:
        """
        Create an optimized async HTTP client with connection pooling and timeouts

        Returns:
            Configured HTTPX AsyncClient
        """
        # Connection limits for optimal performance
        limits = httpx.Limits(
            max_keepalive_connections=20,  # Keep connections alive
            max_connections=50,            # Total connection pool size
            keepalive_expiry=30.0         # Keep connections alive for 30 seconds
        )

        # Optimized timeouts
        timeout = httpx.Timeout(
            connect=10.0,     # Connection timeout
            read=45.0,        # Read timeout (reduced from default 60s)
            write=10.0,       # Write timeout
            pool=5.0          # Pool timeout
        )

        # Create async client with optimization
        return httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            http2=False,      # Disable HTTP/2 for compatibility
            follow_redirects=True
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
        Optimized async chat completion with connection reuse and performance tracking

        Args:
            prompt: The prompt to send
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Generation temperature
            validate_json: Whether to validate JSON
            json_schema: JSON schema for validation

        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        try:
            # Input validation
            if not prompt or not isinstance(prompt, str):
                return self._create_error_response("Invalid prompt provided")

            # Set defaults
            model = model or model_config.extraction_model
            max_tokens = max_tokens or model_config.max_tokens
            temperature = temperature if temperature is not None else model_config.temperature

            # Check cache first
            cache_key = self.cache_manager.get_cache_key(prompt, model, max_tokens, float(temperature))
            cached_response = self.cache_manager.get_from_cache(cache_key)
            
            if cached_response is not None:
                self.connection_stats["cache_hits"] += 1
                response_time = (time.time() - start_time) * 1000
                logger.debug(f"Cache hit for prompt (response time: {response_time:.1f}ms)")
                return cached_response

            # Make optimized API call
            response = await self._make_optimized_api_call(
                prompt, model, max_tokens, temperature
            )

            # Handle JSON validation if requested
            if validate_json and "content" in response:
                response = await self._validate_json_response(response, json_schema)

            # Update performance metrics
            response_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(response_time_ms)

            # Cache successful responses
            if "error" not in response:
                self.cache_manager.add_to_cache(cache_key, response)

            return response

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Optimized async completion failed: {e} (time: {response_time_ms:.1f}ms)")
            return self._create_error_response(str(e))

    async def _make_optimized_api_call(
        self, prompt: str, model: str, max_tokens: int, temperature: float
    ) -> Dict[str, Any]:
        """
        Make optimized API call with connection reuse

        Args:
            prompt: The prompt text
            model: Model name
            max_tokens: Maximum tokens
            temperature: Generation temperature

        Returns:
            API response dictionary
        """
        try:
            # Get system message from prompt generator
            system_message = "You are a helpful assistant that provides accurate and concise answers."
            
            # Make the API call with optimized client
            api_start = time.time()
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=30.0  # Reduced timeout for faster failures
            )
            api_time = (time.time() - api_start) * 1000

            # Track connection reuse (if we can determine it)
            self.connection_stats["connection_reuses"] += 1

            # Format response
            usage = response.usage if hasattr(response, 'usage') else {}
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
                "api_time_ms": round(api_time, 2)
            }

            logger.debug(f"API call completed in {api_time:.1f}ms")
            return formatted_response

        except Exception as e:
            logger.error(f"API call failed: {e}")
            return self._create_error_response(f"API call failed: {str(e)}")

    async def _validate_json_response(
        self, response: Dict[str, Any], json_schema: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate JSON response if requested

        Args:
            response: The API response
            json_schema: JSON schema for validation

        Returns:
            Response with validation results
        """
        try:
            from llm.pipeline.validation import validate_json_output, ANSWER_SCHEMA
            
            schema = json_schema or ANSWER_SCHEMA
            is_valid, parsed_json, error_msg = validate_json_output(
                response["content"], schema=schema
            )

            if is_valid:
                response["parsed_json"] = parsed_json
            else:
                response["json_validation_error"] = error_msg
                logger.warning(f"JSON validation failed: {error_msg}")

            return response

        except Exception as e:
            logger.error(f"JSON validation error: {e}")
            response["json_validation_error"] = str(e)
            return response

    def _update_performance_stats(self, response_time_ms: float) -> None:
        """
        Update performance statistics

        Args:
            response_time_ms: Response time in milliseconds
        """
        self.connection_stats["requests_made"] += 1
        self.connection_stats["total_response_time_ms"] += response_time_ms
        
        if response_time_ms < self.connection_stats["fastest_response_ms"]:
            self.connection_stats["fastest_response_ms"] = response_time_ms
            
        if response_time_ms > self.connection_stats["slowest_response_ms"]:
            self.connection_stats["slowest_response_ms"] = response_time_ms

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create standardized error response

        Args:
            error_message: Error message

        Returns:
            Error response dictionary
        """
        return {
            "content": f"Error: {error_message}",
            "error": error_message,
            "usage": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
            "model": "error",
            "id": "error_response",
            "created": int(time.time())
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get detailed performance statistics

        Returns:
            Performance statistics dictionary
        """
        avg_response_time = 0
        if self.connection_stats["requests_made"] > 0:
            avg_response_time = (
                self.connection_stats["total_response_time_ms"] / 
                self.connection_stats["requests_made"]
            )

        cache_hit_rate = 0
        total_requests = self.connection_stats["requests_made"] + self.connection_stats["cache_hits"]
        if total_requests > 0:
            cache_hit_rate = self.connection_stats["cache_hits"] / total_requests

        return {
            "requests_made": self.connection_stats["requests_made"],
            "cache_hits": self.connection_stats["cache_hits"],
            "cache_hit_rate": round(cache_hit_rate, 3),
            "avg_response_time_ms": round(avg_response_time, 2),
            "fastest_response_ms": round(self.connection_stats["fastest_response_ms"], 2) if self.connection_stats["fastest_response_ms"] != float('inf') else 0,
            "slowest_response_ms": round(self.connection_stats["slowest_response_ms"], 2),
            "connection_reuses": self.connection_stats["connection_reuses"],
            "total_response_time_ms": round(self.connection_stats["total_response_time_ms"], 2)
        }

    async def batch_async_chat_completions(
        self, prompts: list, batch_size: Optional[int] = None
    ) -> list:
        """
        Batch processing with optimized connections

        Args:
            prompts: List of prompt dictionaries
            batch_size: Batch size for processing

        Returns:
            List of responses
        """
        return await self.batch_processor.batch_async_chat_completions(prompts, batch_size)

    async def close_connections(self) -> None:
        """Close HTTP connections for cleanup"""
        try:
            if hasattr(self, '_http_client'):
                self._http_client.close()
            if hasattr(self, '_async_http_client'):
                # Only close if not externally managed
                try:
                    await asyncio.wait_for(self._async_http_client.aclose(), timeout=1.0)
                except (asyncio.TimeoutError, RuntimeError) as e:
                    logger.debug(f"Async client cleanup timeout/error (non-critical): {e}")
            logger.info("HTTP connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

    def close_connections_sync(self) -> None:
        """Synchronous connection cleanup"""
        try:
            if hasattr(self, '_http_client'):
                self._http_client.close()
            # Don't try to close async client synchronously - causes the event loop error
            logger.info("Sync HTTP connections closed")
        except Exception as e:
            logger.error(f"Error closing sync connections: {e}")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            # Only close sync resources in destructor to avoid event loop issues
            if hasattr(self, '_http_client'):
                self._http_client.close()
        except:
            pass  # Ignore errors during cleanup

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_connections()

    # Delegate methods for compatibility
    def chat_completion(self, *args, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for compatibility - safe for use in event loops"""
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in a loop, we can't use asyncio.run(), so create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_async_completion, *args, **kwargs)
                return future.result(timeout=60)  # 60 second timeout
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(self.async_chat_completion(*args, **kwargs))
    
    def _run_async_completion(self, *args, **kwargs) -> Dict[str, Any]:
        """Helper method to run async completion in a new event loop"""
        return asyncio.run(self.async_chat_completion(*args, **kwargs))

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        performance_stats = self.get_performance_stats()
        cache_stats = self.cache_manager.get_cache_stats()
        
        return {
            "performance": performance_stats,
            "cache": cache_stats,
            "batch_processor": self.batch_processor.get_batch_stats() if hasattr(self.batch_processor, 'get_batch_stats') else {}
        }
