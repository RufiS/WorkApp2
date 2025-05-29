"""
LLM answer generation pipeline with extraction and formatting

Extracted from llm/llm_service.py
"""
import asyncio
import logging
import time
import os
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Import prompt utilities with fallback
try:
    from utils.prompts.formatting_prompt import check_formatting_quality
except ImportError:
    def check_formatting_quality(formatted_answer, raw_answer) -> bool:
        return True  # Fallback always returns True


class AnswerPipeline:
    """Handles the complete answer generation pipeline"""

    def __init__(self, llm_service, prompt_generator):
        """
        Initialize answer pipeline

        Args:
            llm_service: Reference to the LLM service instance
            prompt_generator: PromptGenerator instance
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator

    async def process_extraction_and_formatting(
        self, query: str, context: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
                from utils.config import app_config
                from utils.error_logging import log_error

                log_error(
                    error_msg,
                    error_type="WARNING",
                    source="AnswerPipeline.process_extraction_and_formatting",
                )
            except ImportError:
                # Fallback to simple file logging
                try:
                    from utils.config import resolve_path

                    fallback_log_path = resolve_path("./logs/workapp_errors.log", create_dir=True)
                except ImportError:
                    fallback_log_path = "./logs/workapp_errors.log"
                    os.makedirs(os.path.dirname(fallback_log_path), exist_ok=True)

                with open(fallback_log_path, "a") as error_log:
                    error_log.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {error_msg}\n"
                    )
            return {"error": "Empty context provided"}, {
                "error": "Extraction failed due to empty context"
            }

        # Log context length
        logger.info(f"Processing extraction with context length: {len(context)} characters")

        # Generate extraction prompt
        extraction_prompt = self.prompt_generator.generate_extraction_prompt(query, context)

        # Import here to avoid circular imports
        from utils.config import model_config
        from llm.validation import ANSWER_SCHEMA

        # Enhanced multi-strategy extraction with 5 different approaches
        extraction_response = await self._attempt_multi_strategy_extraction(query, context)
        
        # If extraction completely failed, return error
        if "error" in extraction_response and "content" not in extraction_response:
            logger.error(f"All extraction strategies failed: {extraction_response.get('error', 'Unknown error')}")
            return extraction_response, {"error": "All extraction strategies failed"}

        # Parse JSON from extraction response to get the actual answer
        try:
            import json
            extracted_json = json.loads(extraction_response["content"])
            answer_text = extracted_json.get("answer", extraction_response["content"])
            logger.info(f"Successfully parsed JSON from extraction, answer length: {len(answer_text)} chars")
        except (json.JSONDecodeError, KeyError) as e:
            # If JSON parsing fails, use the raw content
            logger.warning(f"Failed to parse JSON from extraction response: {str(e)}, using raw content")
            answer_text = extraction_response["content"]
        
        # Generate formatting prompt with the extracted answer text
        formatting_prompt = self.prompt_generator.generate_formatting_prompt(answer_text)

        # Get formatting result
        formatting_response = await self.llm_service.async_chat_completion(
            prompt=formatting_prompt,
            model=model_config.formatting_model,
            max_tokens=model_config.max_tokens,
        )

        # If there's an error in formatting, retry once with a simpler prompt
        if "error" in formatting_response:
            logger.warning("Error in formatting, retrying with simpler prompt")

            # Simplify the formatting prompt
            simple_formatting_prompt = self.prompt_generator.generate_simple_formatting_prompt(
                extraction_response["content"]
            )

            # Retry formatting
            formatting_response = await self.llm_service.async_chat_completion(
                prompt=simple_formatting_prompt,
                model=model_config.formatting_model,
                max_tokens=model_config.max_tokens,
            )

        # Check formatting quality
        try:
            if formatting_response.get("content") and extraction_response.get("content"):
                quality_check = check_formatting_quality(
                    formatting_response["content"], extraction_response["content"]
                )
                if not quality_check:
                    warning_msg = "Formatting quality check failed - formatted answer may be missing key sections"
                    logger.warning(warning_msg)
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
                error_msg = extraction_response.get(
                    "error", formatting_response.get("error", "Unknown error")
                )
                logger.error(f"Error generating answer: {error_msg}")
                # Log to central error log
                try:
                    from utils.error_logging import log_error

                    log_error(
                        f"Error generating answer: {error_msg}",
                        error_type="ERROR",
                        source="AnswerPipeline.generate_answer",
                    )
                except ImportError:
                    # Fallback to simple file logging
                    try:
                        from utils.config import resolve_path

                        fallback_log_path = resolve_path(
                            "./logs/workapp_errors.log", create_dir=True
                        )
                    except ImportError:
                        fallback_log_path = "./logs/workapp_errors.log"
                        os.makedirs(os.path.dirname(fallback_log_path), exist_ok=True)

                    with open(fallback_log_path, "a") as error_log:
                        error_log.write(
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - Error generating answer: {error_msg}\n"
                        )
                return {
                    "content": f"Error: {error_msg}",
                    "error": error_msg,
                    "raw_extraction": extraction_response.get("content", ""),
                    "formatted_answer": "",
                }

            # Return the formatted answer with metadata
            return {
                "content": formatting_response["content"],
                "raw_extraction": extraction_response["content"],
                "model": formatting_response["model"],
                "usage": {
                    "total_tokens": extraction_response["usage"]["total_tokens"]
                    + formatting_response["usage"]["total_tokens"]
                },
            }
        except Exception as e:
            error_msg = f"Error in generate_answer: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Log to central error log
            try:
                from utils.error_logging import log_error

                log_error(error_msg, include_traceback=True, source="AnswerPipeline.generate_answer")
            except ImportError:
                # Fallback to simple file logging
                try:
                    from utils.config import resolve_path

                    fallback_log_path = resolve_path("./logs/workapp_errors.log", create_dir=True)
                except ImportError:
                    fallback_log_path = "./logs/workapp_errors.log"
                    os.makedirs(os.path.dirname(fallback_log_path), exist_ok=True)

                import traceback

                with open(fallback_log_path, "a") as error_log:
                    error_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ERROR - {error_msg}\n")
                    error_log.write(traceback.format_exc())
                    error_log.write("\n")
            return {"content": f"Error: {str(e)}", "error": str(e)}

    async def _attempt_multi_strategy_extraction(self, query: str, context: str) -> Dict[str, Any]:
        """
        Attempt extraction using multiple strategies for maximum reliability
        
        Args:
            query: The user's question
            context: The retrieved context
            
        Returns:
            Dictionary with extraction response (always succeeds with some strategy)
        """
        from utils.config import model_config
        from llm.validation import ANSWER_SCHEMA
        from utils.prompts.extraction_prompt import generate_extraction_prompt_v2, generate_extraction_prompt_legacy, generate_extraction_prompt_simple
        
        # Strategy definitions
        strategies = [
            ("v2_full", "Full V2 prompt with enhanced comprehensiveness"),
            ("v2_reduced", "V2 prompt with reduced context"),
            ("v2_simple", "Simplified V2 prompt"),
            ("legacy_fallback", "Original legacy prompt"),
            ("content_extraction", "Extract usable content from any response")
        ]
        
        for strategy_name, description in strategies:
            try:
                logger.info(f"Attempting extraction strategy: {description}")
                
                if strategy_name == "v2_full":
                    # Strategy 1: Full V2 prompt with JSON validation
                    prompt = generate_extraction_prompt_v2(query, context)
                    response = await self.llm_service.async_chat_completion(
                        prompt=prompt,
                        model=model_config.extraction_model,
                        max_tokens=model_config.max_tokens,
                        validate_json=True,
                        json_schema=ANSWER_SCHEMA,
                    )
                    
                elif strategy_name == "v2_reduced":
                    # Strategy 2: V2 prompt with reduced context (if context is large)
                    if len(context) > 2000:
                        reduced_context = context[:int(len(context) * 0.7)] + "\n\n[Context truncated for processing]"
                        prompt = generate_extraction_prompt_v2(query, reduced_context)
                    else:
                        prompt = generate_extraction_prompt_v2(query, context)
                    
                    response = await self.llm_service.async_chat_completion(
                        prompt=prompt,
                        model=model_config.extraction_model,
                        max_tokens=model_config.max_tokens,
                        validate_json=True,
                        json_schema=ANSWER_SCHEMA,
                    )
                    
                elif strategy_name == "v2_simple":
                    # Strategy 3: Simplified V2 prompt
                    prompt = generate_extraction_prompt_simple(query, context)
                    response = await self.llm_service.async_chat_completion(
                        prompt=prompt,
                        model=model_config.extraction_model,
                        max_tokens=model_config.max_tokens,
                        validate_json=True,
                        json_schema=ANSWER_SCHEMA,
                    )
                    
                elif strategy_name == "legacy_fallback":
                    # Strategy 4: Legacy prompt as fallback
                    prompt = generate_extraction_prompt_legacy(query, context)
                    response = await self.llm_service.async_chat_completion(
                        prompt=prompt,
                        model=model_config.extraction_model,
                        max_tokens=model_config.max_tokens,
                        validate_json=True,
                        json_schema=ANSWER_SCHEMA,
                    )
                    
                elif strategy_name == "content_extraction":
                    # Strategy 5: Extract usable content from any response (no JSON validation)
                    prompt = f"Based on this context, answer this question comprehensively: {query}\n\nContext: {context}"
                    response = await self.llm_service.async_chat_completion(
                        prompt=prompt,
                        model=model_config.extraction_model,
                        max_tokens=model_config.max_tokens,
                        validate_json=False,  # No validation - accept any response
                    )
                    
                    # Convert raw response to expected JSON format
                    if "content" in response and response["content"]:
                        response["content"] = f'{{"answer": "{response["content"].replace('"', '\"')}", "sources": [], "confidence": 0.7}}'
                
                # Check if this strategy succeeded
                if self._is_valid_response(response):
                    logger.info(f"Extraction succeeded with strategy: {description}")
                    return response
                else:
                    logger.warning(f"Strategy {strategy_name} failed: {response.get('error', 'Invalid response')}")
                    
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed with exception: {str(e)}")
                continue
        
        # Final fallback: create a minimal valid response
        logger.error("All extraction strategies failed, creating fallback response")
        return self._create_fallback_response(query, context)
    
    def _is_valid_response(self, response: Dict[str, Any]) -> bool:
        """
        Check if a response is valid and usable
        
        Args:
            response: The response to validate
            
        Returns:
            True if response is valid and usable
        """
        # Must have content and no critical errors
        if "error" in response and "content" not in response:
            return False
        if not response.get("content"):
            return False
        if len(response.get("content", "").strip()) < 10:
            return False
        return True
    
    def _create_fallback_response(self, query: str, context: str) -> Dict[str, Any]:
        """
        Create a fallback response that always works
        
        Args:
            query: The user's question
            context: The retrieved context
            
        Returns:
            Dictionary with fallback response
        """
        fallback_content = '{{"answer": "Unable to process the response. Please try rephrasing your question or contact a manager.", "sources": [], "confidence": 0.3}}'
        
        return {
            "content": fallback_content,
            "model": "fallback",
            "usage": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
            "id": "fallback_response",
            "created": int(time.time()),
            "fallback": True
        }

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline processing statistics

        Returns:
            Dictionary with pipeline stats
        """
        return {
            "has_prompt_generator": self.prompt_generator is not None,
            "has_llm_service": self.llm_service is not None,
        }
