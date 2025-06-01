"""Async utility functions and helpers"""

import asyncio
import logging
import time
from typing import List, Callable, Any, Dict, Optional, Coroutine
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AsyncHelper:
    """Utility class for async operations"""

    @staticmethod
    async def run_in_thread(func: Callable, *args, **kwargs) -> Any:
        """
        Run a synchronous function in a thread pool

        Args:
            func: Function to run
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    @staticmethod
    async def gather_with_concurrency(tasks: List[Coroutine], max_concurrency: int = 10) -> List[Any]:
        """
        Run tasks with limited concurrency

        Args:
            tasks: List of coroutines to run
            max_concurrency: Maximum number of concurrent tasks

        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_task(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_task(task) for task in tasks]
        return await asyncio.gather(*bounded_tasks)

    @staticmethod
    async def timeout_after(seconds: float, coro: Coroutine, default=None) -> Any:
        """
        Run a coroutine with a timeout

        Args:
            seconds: Timeout in seconds
            coro: Coroutine to run
            default: Default value to return on timeout

        Returns:
            Coroutine result or default on timeout
        """
        try:
            return await asyncio.wait_for(coro, timeout=seconds)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {seconds} seconds")
            return default

    @staticmethod
    async def retry_async(func: Callable, max_attempts: int = 3, delay: float = 1.0,
                         backoff: float = 2.0, *args, **kwargs) -> Any:
        """
        Retry an async function with exponential backoff

        Args:
            func: Async function to retry
            max_attempts: Maximum number of attempts
            delay: Initial delay between attempts
            backoff: Backoff multiplier
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: Last exception if all attempts fail
        """
        last_exception = None
        current_delay = delay

        for attempt in range(max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {current_delay}s")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
                else:
                    logger.error(f"All {max_attempts} attempts failed")

        raise last_exception

    @staticmethod
    async def batch_process(items: List[Any], batch_size: int,
                           processor: Callable, delay_between_batches: float = 0.0) -> List[Any]:
        """
        Process items in batches

        Args:
            items: Items to process
            batch_size: Size of each batch
            processor: Async function to process each batch
            delay_between_batches: Delay between batches in seconds

        Returns:
            List of all results
        """
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await processor(batch)
            results.extend(batch_results)

            # Add delay between batches if specified
            if delay_between_batches > 0 and i + batch_size < len(items):
                await asyncio.sleep(delay_between_batches)

        return results


def async_timer(func):
    """Decorator to time async functions"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger.debug(f"{func.__name__} took {duration:.2f} seconds")
    return wrapper


def ensure_async(func_or_coro):
    """Ensure a function or coroutine is async"""
    if asyncio.iscoroutinefunction(func_or_coro):
        return func_or_coro
    elif asyncio.iscoroutine(func_or_coro):
        async def wrapper():
            return func_or_coro
        return wrapper()
    else:
        async def wrapper(*args, **kwargs):
            return func_or_coro(*args, **kwargs)
        return wrapper
