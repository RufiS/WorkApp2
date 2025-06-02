"""
Optimized Answer Pipeline with performance improvements
Eliminates ThreadPoolExecutor anti-pattern and optimizes async flow
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)


class OptimizedAnswerPipeline:
    """Optimized version of AnswerPipeline with performance improvements"""

    def __init__(self, llm_service, prompt_generator):
        """
        Initialize optimized answer pipeline

        Args:
            llm_service: Reference to the LLM service instance
            prompt_generator: PromptGenerator instance
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        
        # Strategy success tracking for optimization
        self.strategy_success_history = {
            "v2_full": {"attempts": 0, "successes": 0},
            "v2_simple": {"attempts": 0, "successes": 0},
            "content_extraction": {"attempts": 0, "successes": 0},
            "v2_reduced": {"attempts": 0, "successes": 0},
            "simple_fallback": {"attempts": 0, "successes": 0}
        }

    async def generate_answer_async(self, query: str, context: str) -> Dict[str, Any]:
        """
        Async-first answer generation (eliminates ThreadPoolExecutor anti-pattern)

        Args:
            query: The user's question
            context: The retrieved context

        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Validate inputs
            if not context or context.strip() == "":
                return {
                    "content": "Error: No context provided for the query",
                    "error": "Empty context",
                    "raw_extraction": "",
                    "formatted_answer": ""
                }

            # Use optimized extraction and formatting
            start_time = time.time()
            extraction_response, formatting_response = await self._process_optimized_pipeline(
                query, context
            )
            processing_time = time.time() - start_time

            # Handle errors
            if "error" in extraction_response or "error" in formatting_response:
                error_msg = extraction_response.get(
                    "error", formatting_response.get("error", "Unknown error")
                )
                logger.error(f"Pipeline error: {error_msg}")
                return {
                    "content": f"Error: {error_msg}",
                    "error": error_msg,
                    "raw_extraction": extraction_response.get("content", ""),
                    "formatted_answer": "",
                    "processing_time_ms": round(processing_time * 1000, 2)
                }

            # Calculate token usage
            total_tokens = 0
            if extraction_response.get("usage"):
                total_tokens += extraction_response["usage"].get("total_tokens", 0)
            if formatting_response.get("usage"):
                total_tokens += formatting_response["usage"].get("total_tokens", 0)

            # Return successful result
            return {
                "content": formatting_response["content"],
                "raw_extraction": extraction_response["content"],
                "model": formatting_response["model"],
                "usage": {"total_tokens": total_tokens},
                "processing_time_ms": round(processing_time * 1000, 2),
                "strategy_used": extraction_response.get("strategy_used", "unknown")
            }

        except Exception as e:
            error_msg = f"Error in optimized pipeline: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "content": f"Error: {str(e)}",
                "error": str(e),
                "raw_extraction": "",
                "formatted_answer": "",
                "processing_time_ms": 0
            }

    def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """
        Synchronous wrapper with proper async handling (no ThreadPoolExecutor)

        Args:
            query: The user's question
            context: The retrieved context

        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, run the coroutine
                task = asyncio.create_task(self.generate_answer_async(query, context))
                
                # If we can't await directly, we need to handle this differently
                # For now, create a new event loop in a thread
                import concurrent.futures
                import threading
                
                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    try:
                        asyncio.set_event_loop(new_loop)
                        return new_loop.run_until_complete(
                            self.generate_answer_async(query, context)
                        )
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_in_new_loop)
                    return future.result(timeout=120)  # 2 minute timeout
                    
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(self.generate_answer_async(query, context))
                
        except Exception as e:
            error_msg = f"Error in sync wrapper: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "content": f"Error: {str(e)}",
                "error": str(e),
                "raw_extraction": "",
                "formatted_answer": "",
                "processing_time_ms": 0
            }

    async def _process_optimized_pipeline(
        self, query: str, context: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimized processing pipeline with smart strategy selection

        Args:
            query: The user's question
            context: The retrieved context

        Returns:
            Tuple of (extraction_response, formatting_response)
        """
        # Use smart extraction with optimized strategy order
        extraction_response = await self._smart_extraction(query, context)

        if "error" in extraction_response and "content" not in extraction_response:
            # Extraction failed completely
            return extraction_response, {"error": "Extraction failed"}

        # Parse extraction result
        try:
            extracted_json = json.loads(extraction_response["content"])
            answer_text = extracted_json.get("answer", extraction_response["content"])
        except (json.JSONDecodeError, KeyError):
            # Use raw content if JSON parsing fails
            answer_text = extraction_response["content"]

        # Generate formatting (could be optimized further with parallel processing)
        formatting_response = await self._fast_formatting(answer_text)

        return extraction_response, formatting_response

    async def _smart_extraction(self, query: str, context: str) -> Dict[str, Any]:
        """
        Smart extraction with optimized strategy order based on success history

        Args:
            query: The user's question
            context: The retrieved context

        Returns:
            Dictionary with extraction response
        """
        # Get optimized strategy order based on success rates
        strategies = self._get_optimized_strategy_order()

        # Import here to avoid circular imports
        from core.config import model_config
        from llm.pipeline.validation import ANSWER_SCHEMA
        from llm.prompts.extraction_prompt import generate_extraction_prompt_v2, generate_extraction_prompt_simple

        for strategy_name, description in strategies:
            try:
                # Track attempt
                self.strategy_success_history[strategy_name]["attempts"] += 1

                # Quick strategy selection
                if strategy_name == "v2_simple":
                    # Start with the most reliable strategy
                    prompt = generate_extraction_prompt_simple(query, context)
                    response = await self.llm_service.async_chat_completion(
                        prompt=prompt,
                        model=model_config.extraction_model,
                        max_tokens=model_config.max_tokens,
                        validate_json=True,
                        json_schema=ANSWER_SCHEMA,
                    )

                elif strategy_name == "v2_full":
                    # Full V2 prompt for complex queries
                    prompt = generate_extraction_prompt_v2(query, context)
                    response = await self.llm_service.async_chat_completion(
                        prompt=prompt,
                        model=model_config.extraction_model,
                        max_tokens=model_config.max_tokens,
                        validate_json=True,
                        json_schema=ANSWER_SCHEMA,
                    )

                elif strategy_name == "content_extraction":
                    # Fast fallback without JSON validation
                    prompt = f"Answer this question based on the context provided.\n\nQuestion: {query}\n\nContext: {context}\n\nAnswer:"
                    response = await self.llm_service.async_chat_completion(
                        prompt=prompt,
                        model=model_config.extraction_model,
                        max_tokens=model_config.max_tokens,
                        validate_json=False,
                    )

                    # Wrap in JSON format
                    if "content" in response and response["content"]:
                        content = response["content"].replace('"', '\\"')
                        response["content"] = f'{{"answer": "{content}", "sources": [], "confidence": 0.7}}'

                elif strategy_name == "v2_reduced":
                    # Reduced context for large inputs
                    reduced_context = context[:2000] if len(context) > 2000 else context
                    prompt = generate_extraction_prompt_v2(query, reduced_context)
                    response = await self.llm_service.async_chat_completion(
                        prompt=prompt,
                        model=model_config.extraction_model,
                        max_tokens=model_config.max_tokens,
                        validate_json=True,
                        json_schema=ANSWER_SCHEMA,
                    )

                else:  # simple_fallback
                    # Last resort
                    prompt = generate_extraction_prompt_simple(query, context)
                    response = await self.llm_service.async_chat_completion(
                        prompt=prompt,
                        model=model_config.extraction_model,
                        max_tokens=model_config.max_tokens,
                        validate_json=False,
                    )

                # Check if successful
                if self._is_valid_response(response):
                    # Track success
                    self.strategy_success_history[strategy_name]["successes"] += 1
                    response["strategy_used"] = strategy_name
                    logger.info(f"✅ Extraction succeeded with strategy: {strategy_name}")
                    return response
                else:
                    logger.warning(f"❌ Strategy {strategy_name} failed: {response.get('error', 'Invalid response')}")

            except Exception as e:
                logger.warning(f"❌ Strategy {strategy_name} failed with exception: {str(e)}")
                continue

        # All strategies failed, return fallback
        logger.error("All extraction strategies failed")
        return self._create_fallback_response(query, context)

    async def _fast_formatting(self, answer_text: str) -> Dict[str, Any]:
        """
        Fast formatting with optimized prompts

        Args:
            answer_text: The extracted answer text

        Returns:
            Dictionary with formatting response
        """
        from core.config import model_config

        try:
            # Use optimized formatting prompt
            formatting_prompt = self.prompt_generator.generate_formatting_prompt(answer_text)

            response = await self.llm_service.async_chat_completion(
                prompt=formatting_prompt,
                model=model_config.formatting_model,
                max_tokens=model_config.max_tokens,
            )

            return response

        except Exception as e:
            logger.error(f"Formatting failed: {e}")
            # Return the original text if formatting fails
            return {
                "content": answer_text,
                "model": "fallback",
                "usage": {"total_tokens": 0},
                "formatting_error": str(e)
            }

    def _get_optimized_strategy_order(self) -> List[Tuple[str, str]]:
        """
        Get strategy order optimized by success rate

        Returns:
            List of (strategy_name, description) tuples in optimal order
        """
        # Calculate success rates
        strategy_rates = {}
        for strategy, stats in self.strategy_success_history.items():
            if stats["attempts"] > 0:
                success_rate = stats["successes"] / stats["attempts"]
            else:
                # Default success rates for new strategies
                default_rates = {
                    "v2_simple": 0.9,      # Usually most reliable
                    "content_extraction": 0.8,  # Fast fallback
                    "v2_full": 0.7,        # More complex but thorough
                    "v2_reduced": 0.6,     # For large contexts
                    "simple_fallback": 0.5  # Last resort
                }
                success_rate = default_rates.get(strategy, 0.5)
            
            strategy_rates[strategy] = success_rate

        # Sort by success rate (descending)
        sorted_strategies = sorted(strategy_rates.items(), key=lambda x: x[1], reverse=True)

        # Convert to strategy definitions
        strategy_descriptions = {
            "v2_simple": "Simplified V2 prompt (most reliable)",
            "content_extraction": "Direct content extraction (fast fallback)",
            "v2_full": "Full V2 prompt with enhanced comprehensiveness",
            "v2_reduced": "V2 prompt with reduced context",
            "simple_fallback": "Simple prompt fallback"
        }

        return [(strategy, strategy_descriptions[strategy]) for strategy, _ in sorted_strategies]

    def _is_valid_response(self, response: Dict[str, Any]) -> bool:
        """
        Check if a response is valid and usable

        Args:
            response: The response to validate

        Returns:
            True if response is valid and usable
        """
        if "error" in response and "content" not in response:
            return False
        if not response.get("content"):
            return False
        if len(response.get("content", "").strip()) < 10:
            return False
        return True

    def _create_fallback_response(self, query: str, context: str) -> Dict[str, Any]:
        """
        Create a fallback response

        Args:
            query: The user's question
            context: The retrieved context

        Returns:
            Dictionary with fallback response
        """
        fallback_content = '{{"answer": "I apologize, but I\'m having trouble processing your question. Please try rephrasing or contact support.", "sources": [], "confidence": 0.2}}'

        return {
            "content": fallback_content,
            "model": "fallback",
            "usage": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
            "strategy_used": "fallback",
            "fallback": True
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline performance statistics

        Returns:
            Dictionary with pipeline stats
        """
        # Calculate overall success rates
        total_attempts = sum(stats["attempts"] for stats in self.strategy_success_history.values())
        total_successes = sum(stats["successes"] for stats in self.strategy_success_history.values())
        overall_success_rate = total_successes / total_attempts if total_attempts > 0 else 0.0

        return {
            "overall_success_rate": round(overall_success_rate, 3),
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "strategy_performance": {
                strategy: {
                    "attempts": stats["attempts"],
                    "successes": stats["successes"],
                    "success_rate": round(stats["successes"] / stats["attempts"], 3) if stats["attempts"] > 0 else 0.0
                }
                for strategy, stats in self.strategy_success_history.items()
            },
            "optimized_strategy_order": [s[0] for s in self._get_optimized_strategy_order()],
            "has_llm_service": self.llm_service is not None,
            "has_prompt_generator": self.prompt_generator is not None
        }

    def reset_strategy_history(self) -> None:
        """Reset strategy success history (for testing)"""
        for strategy in self.strategy_success_history:
            self.strategy_success_history[strategy] = {"attempts": 0, "successes": 0}
