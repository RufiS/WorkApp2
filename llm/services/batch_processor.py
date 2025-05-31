"""
LLM batch processing coordination

Extracted from llm/llm_service.py
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing of LLM requests"""

    def __init__(self, llm_service, batch_size: int = 5):
        """
        Initialize batch processor

        Args:
            llm_service: Reference to the LLM service instance
            batch_size: Maximum number of concurrent requests
        """
        self.llm_service = llm_service
        self.batch_size = batch_size

    async def batch_async_chat_completions(
        self, prompts: List[Dict[str, Any]], batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple chat completion requests in parallel with batching.

        Args:
            prompts: List of dictionaries with prompt details:
                     [{"prompt": str, "model": str, "max_tokens": int, "temperature": float}]
            batch_size: Maximum number of concurrent requests to make (None for default)
        Returns:
            List of response dictionaries in the same order as the input prompts
        """
        # Use configured batch size if not specified
        effective_batch_size = batch_size or self.batch_size
        results = []

        # Check cache first for all prompts
        cache_hits = []
        cache_misses = []

        for i, p in enumerate(prompts):
            # Import here to avoid circular imports
            from utils.config import model_config
            
            model = p.get("model") or model_config.extraction_model
            max_tokens = p.get("max_tokens") or model_config.max_tokens
            temperature = (
                p.get("temperature")
                if p.get("temperature") is not None
                else model_config.temperature
            )

            cache_key = self.llm_service.cache_manager.get_cache_key(p["prompt"], model, max_tokens, temperature)
            cached_response = self.llm_service.cache_manager.get_from_cache(cache_key)

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
            logger.info(f"Processing {len(cache_misses)} cache misses in batches of {effective_batch_size}")

            # Group by batch
            for i in range(0, len(cache_misses), effective_batch_size):
                batch = cache_misses[i : i + effective_batch_size]

                # Create tasks for the current batch
                tasks = []
                for _, p in batch:
                    task = self.llm_service.async_chat_completion(
                        prompt=p["prompt"],
                        model=p.get("model"),
                        max_tokens=p.get("max_tokens"),
                        temperature=p.get("temperature"),
                    )
                    tasks.append(task)

                # Run current batch concurrently
                batch_results = await asyncio.gather(*tasks)

                # Fill in results
                for j, (idx, _) in enumerate(batch):
                    results[idx] = batch_results[j]

                # Add a small delay between batches to avoid rate limiting
                if i + effective_batch_size < len(cache_misses):
                    await asyncio.sleep(0.5)

        return results

    async def process_parallel_requests(
        self, 
        requests: List[Dict[str, Any]], 
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple requests in parallel with concurrency control

        Args:
            requests: List of request dictionaries
            max_concurrent: Maximum concurrent requests (None for default)

        Returns:
            List of response dictionaries
        """
        max_concurrent = max_concurrent or self.batch_size
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.llm_service.async_chat_completion(**request_data)
        
        # Create tasks for all requests
        tasks = [process_single_request(req) for req in requests]
        
        # Execute all tasks concurrently (with semaphore limiting)
        return await asyncio.gather(*tasks)

    def get_batch_stats(self) -> Dict[str, Any]:
        """
        Get batch processing statistics

        Returns:
            Dictionary with batch processing stats
        """
        return {
            "configured_batch_size": self.batch_size,
            "cache_manager_stats": self.llm_service.cache_manager.get_cache_stats() if hasattr(self.llm_service, 'cache_manager') else {}
        }

    def update_batch_size(self, new_batch_size: int) -> None:
        """
        Update the batch size for processing

        Args:
            new_batch_size: New batch size to use
        """
        if new_batch_size > 0:
            self.batch_size = new_batch_size
            logger.info(f"Updated batch size to {new_batch_size}")
        else:
            logger.warning(f"Invalid batch size {new_batch_size}, keeping current size {self.batch_size}")
