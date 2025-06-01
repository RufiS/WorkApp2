"""
LLM answer generation pipeline with extraction and formatting

Enhanced with comprehensive JSON debug logging
"""
import asyncio
import logging
import time
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Import prompt utilities with fallback
try:
    from llm.prompts.formatting_prompt import check_formatting_quality
except ImportError:
    def check_formatting_quality(formatted_answer, raw_answer) -> bool:
        return True  # Fallback always returns True


class AnswerPipelineDebugger:
    """Handles JSON debug logging for the answer pipeline"""

    def __init__(self, debug_file_path: str = "logs/answer_pipeline_debug.json"):
        """
        Initialize the debugger

        Args:
            debug_file_path: Path to the debug log file
        """
        self.debug_file_path = debug_file_path
        self.session_id = None
        self.debug_data = {}

        # Ensure debug directory exists
        os.makedirs(os.path.dirname(debug_file_path), exist_ok=True)

    def start_session(self, query: str, context: str) -> str:
        """
        Start a new debug session

        Args:
            query: The user's query
            context: The retrieved context

        Returns:
            Session ID
        """
        self.session_id = str(uuid.uuid4())[:8]

        # Get current retrieval settings
        try:
            from core.config import retrieval_config, performance_config
            similarity_threshold = retrieval_config.similarity_threshold
            vector_weight = getattr(retrieval_config, 'vector_weight', 0.5)
            enhanced_mode = getattr(retrieval_config, 'enhanced_mode', False)
            enable_reranking = getattr(performance_config, 'enable_reranking', False)

            # Determine retrieval method
            if enable_reranking:
                retrieval_method = "reranking"
            elif enhanced_mode:
                retrieval_method = f"hybrid_search (vector_weight: {vector_weight})"
            else:
                retrieval_method = "basic_vector_search"
        except Exception as e:
            similarity_threshold = "unknown"
            retrieval_method = f"unknown (error: {str(e)})"

        self.debug_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "query": query,
            "pipeline_trace": {
                "input": {
                    "query": query,
                    "context_length": len(context),
                    "context_preview": self._truncate_text(context, 200),
                    "similarity_threshold": similarity_threshold,
                    "retrieval_method": retrieval_method
                }
            },
            "errors": [],
            "warnings": []
        }

        logger.info(f"DEBUG SESSION STARTED: {self.session_id} for query: '{self._truncate_text(query, 50)}'")
        return self.session_id

    def log_extraction_attempt(self, strategy_name: str, prompt: str, response: Dict[str, Any], success: bool):
        """
        Log an extraction attempt

        Args:
            strategy_name: Name of the strategy attempted
            prompt: The prompt sent to the model
            response: The response from the model
            success: Whether the attempt succeeded
        """
        if "extraction_attempts" not in self.debug_data["pipeline_trace"]:
            self.debug_data["pipeline_trace"]["extraction_attempts"] = []

        attempt_data = {
            "strategy": strategy_name,
            "prompt_preview": self._truncate_text(prompt, 300),
            "prompt_length": len(prompt),
            "success": success,
            "response": {
                "content": self._truncate_text(response.get("content", ""), 500),
                "content_length": len(response.get("content", "")),
                "model": response.get("model", "unknown"),
                "usage": response.get("usage", {}),
                "has_error": "error" in response
            }
        }

        if "error" in response:
            attempt_data["error"] = response["error"]

        self.debug_data["pipeline_trace"]["extraction_attempts"].append(attempt_data)

    def log_successful_extraction(self, extraction_response: Dict[str, Any], parsed_answer: str, parsing_success: bool, parsing_error: str = None):
        """
        Log successful extraction and JSON parsing

        Args:
            extraction_response: The successful extraction response
            parsed_answer: The parsed answer text
            parsing_success: Whether JSON parsing succeeded
            parsing_error: Error message if parsing failed
        """
        self.debug_data["pipeline_trace"]["extraction"] = {
            "final_response": {
                "content": self._truncate_text(extraction_response.get("content", ""), 500),
                "content_length": len(extraction_response.get("content", "")),
                "model": extraction_response.get("model", "unknown"),
                "usage": extraction_response.get("usage", {})
            },
            "json_parsing": {
                "success": parsing_success,
                "extracted_answer_text": self._truncate_text(parsed_answer, 300),
                "extracted_answer_length": len(parsed_answer),
                "parsing_error": parsing_error
            }
        }

    def log_formatting_attempt(self, input_text: str, prompt: str, response: Dict[str, Any]):
        """
        Log formatting attempt

        Args:
            input_text: The text sent for formatting
            prompt: The formatting prompt
            response: The formatting response
        """
        self.debug_data["pipeline_trace"]["formatting"] = {
            "input_text": self._truncate_text(input_text, 300),
            "input_text_length": len(input_text),
            "prompt_sent": self._truncate_text(prompt, 300),
            "prompt_length": len(prompt),
            "response": {
                "content": self._truncate_text(response.get("content", ""), 500),
                "content_length": len(response.get("content", "")),
                "model": response.get("model", "unknown"),
                "usage": response.get("usage", {}),
                "has_error": "error" in response
            }
        }

        if "error" in response:
            self.debug_data["pipeline_trace"]["formatting"]["error"] = response["error"]

    def log_final_result(self, success: bool, final_content: str, errors: list = None):
        """
        Log the final pipeline result

        Args:
            success: Whether the pipeline succeeded
            final_content: The final content returned
            errors: List of errors encountered
        """
        self.debug_data["pipeline_trace"]["final_result"] = {
            "success": success,
            "content": self._truncate_text(final_content, 500),
            "content_length": len(final_content)
        }

        if errors:
            self.debug_data["errors"].extend(errors)

        # Detect common issues
        if "The draft is already formatted correctly" in final_content:
            self.debug_data["warnings"].append("Formatting model thinks input is already a formatted draft - possible prompt confusion")

        if "No changes are needed" in final_content:
            self.debug_data["warnings"].append("Formatting model returned 'no changes needed' - likely receiving wrong input type")

    def add_error(self, error_msg: str):
        """Add an error to the debug log"""
        self.debug_data["errors"].append(error_msg)

    def add_warning(self, warning_msg: str):
        """Add a warning to the debug log"""
        self.debug_data["warnings"].append(warning_msg)

    def save_debug_log(self):
        """Save the debug log to file"""
        try:
            # Read existing debug entries
            existing_entries = []
            if os.path.exists(self.debug_file_path):
                try:
                    with open(self.debug_file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                existing_entries.append(json.loads(line))
                except Exception as e:
                    logger.warning(f"Could not read existing debug file: {str(e)}")

            # Keep only the last 49 entries (we'll add 1 more)
            if len(existing_entries) >= 50:
                existing_entries = existing_entries[-49:]

            # Add our new entry
            existing_entries.append(self.debug_data)

            # Write all entries back
            with open(self.debug_file_path, 'w') as f:
                for entry in existing_entries:
                    f.write(json.dumps(entry, indent=None, separators=(',', ':')) + '\n')

            logger.info(f"DEBUG SESSION SAVED: {self.session_id} to {self.debug_file_path}")

        except Exception as e:
            logger.error(f"Failed to save debug log: {str(e)}")
            # Fallback: save to a backup location
            try:
                backup_path = f"logs/debug_backup_{self.session_id}.json"
                with open(backup_path, 'w') as f:
                    json.dump(self.debug_data, f, indent=2)
                logger.info(f"Debug data saved to backup: {backup_path}")
            except Exception as backup_error:
                logger.error(f"Failed to save debug backup: {str(backup_error)}")

    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        Truncate text for debug output

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text with ellipsis if needed
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


class AnswerPipeline:
    """Handles the complete answer generation pipeline with debug logging"""

    def __init__(self, llm_service, prompt_generator):
        """
        Initialize answer pipeline

        Args:
            llm_service: Reference to the LLM service instance
            prompt_generator: PromptGenerator instance
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        self.debugger = AnswerPipelineDebugger()

    async def process_extraction_and_formatting(
        self, query: str, context: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process both extraction and formatting with comprehensive debug logging

        Args:
            query: The user's question
            context: The retrieved context

        Returns:
            Tuple of (extraction_response, formatting_response)
        """
        # Start debug session
        session_id = self.debugger.start_session(query, context)

        try:
            # Check for empty context
            if not context or context.strip() == "":
                error_msg = f"Empty context provided for query: {query}"
                logger.error(error_msg)
                self.debugger.add_error(error_msg)
                self.debugger.log_final_result(False, "Error: Empty context provided", [error_msg])
                self.debugger.save_debug_log()

                # Log to central error log
                try:
                    from core.config import app_config
                    from utils.error_logging import log_error

                    log_error(
                        error_msg,
                        error_type="WARNING",
                        source="AnswerPipeline.process_extraction_and_formatting",
                    )
                except ImportError:
                    # Fallback to simple file logging
                    try:
                        from core.config import resolve_path

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

            # Import here to avoid circular imports
            from core.config import model_config
            from llm.pipeline.validation import ANSWER_SCHEMA

            # Enhanced multi-strategy extraction with 5 different approaches
            extraction_response = await self._attempt_multi_strategy_extraction(query, context)

            # If extraction completely failed, return error
            if "error" in extraction_response and "content" not in extraction_response:
                error_msg = f"All extraction strategies failed: {extraction_response.get('error', 'Unknown error')}"
                logger.error(error_msg)
                self.debugger.add_error(error_msg)
                self.debugger.log_final_result(False, f"Error: {error_msg}", [error_msg])
                self.debugger.save_debug_log()
                return extraction_response, {"error": "All extraction strategies failed"}

            # Parse JSON from extraction response to get the actual answer
            parsing_success = False
            parsing_error = None
            try:
                import json
                extracted_json = json.loads(extraction_response["content"])
                answer_text = extracted_json.get("answer", extraction_response["content"])
                parsing_success = True
                logger.info(f"Successfully parsed JSON from extraction, answer length: {len(answer_text)} chars")
            except (json.JSONDecodeError, KeyError) as e:
                # If JSON parsing fails, use the raw content
                parsing_error = str(e)
                logger.warning(f"Failed to parse JSON from extraction response: {str(e)}, using raw content")
                answer_text = extraction_response["content"]
                self.debugger.add_warning(f"JSON parsing failed: {str(e)}")

            # Log extraction results
            self.debugger.log_successful_extraction(extraction_response, answer_text, parsing_success, parsing_error)

            # Generate formatting prompt with the extracted answer text
            formatting_prompt = self.prompt_generator.generate_formatting_prompt(answer_text)

            # Get formatting result
            formatting_response = await self.llm_service.async_chat_completion(
                prompt=formatting_prompt,
                model=model_config.formatting_model,
                max_tokens=model_config.max_tokens,
            )

            # Log formatting attempt
            self.debugger.log_formatting_attempt(answer_text, formatting_prompt, formatting_response)

            # If there's an error in formatting, retry once with a simpler prompt
            if "error" in formatting_response:
                logger.warning("Error in formatting, retrying with simpler prompt")
                self.debugger.add_warning("Formatting failed, retrying with simpler prompt")

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

                # Log retry attempt
                self.debugger.log_formatting_attempt(
                    extraction_response["content"],
                    simple_formatting_prompt,
                    formatting_response
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
                        self.debugger.add_warning(warning_msg)
            except Exception as e:
                logger.error(f"Error in formatting quality check: {str(e)}")
                self.debugger.add_error(f"Quality check error: {str(e)}")
                # Don't fail the whole process if quality check fails

            # Log final result
            final_content = formatting_response.get("content", "")
            success = bool(final_content and "error" not in formatting_response)
            self.debugger.log_final_result(success, final_content)
            self.debugger.save_debug_log()

            return extraction_response, formatting_response

        except Exception as e:
            error_msg = f"Unexpected error in pipeline: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.debugger.add_error(error_msg)
            self.debugger.log_final_result(False, f"Error: {error_msg}", [error_msg])
            self.debugger.save_debug_log()
            raise

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
        # Use modern async pattern
        try:
            # Try to get existing event loop
            loop = asyncio.get_running_loop()
            # If we're already in an async context, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.process_extraction_and_formatting(query, context)
                )
                extraction_response, formatting_response = future.result()
        except RuntimeError:
            # No event loop running, use asyncio.run() directly
            extraction_response, formatting_response = asyncio.run(
                self.process_extraction_and_formatting(query, context)
            )

        try:
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
                        from core.config import resolve_path

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
                    from core.config import resolve_path

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
        Attempt extraction using multiple strategies with debug logging

        Args:
            query: The user's question
            context: The retrieved context

        Returns:
            Dictionary with extraction response (always succeeds with some strategy)
        """
        from core.config import model_config
        from llm.pipeline.validation import ANSWER_SCHEMA
        from llm.prompts.extraction_prompt import generate_extraction_prompt_v2, generate_extraction_prompt_simple

        # Strategy definitions
        strategies = [
            ("v2_full", "Full V2 prompt with enhanced comprehensiveness"),
            ("v2_reduced", "V2 prompt with reduced context"),
            ("v2_simple", "Simplified V2 prompt"),
            ("simple_fallback", "Simple prompt fallback"),
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

                elif strategy_name == "simple_fallback":
                    # Strategy 4: Simple prompt as fallback
                    prompt = generate_extraction_prompt_simple(query, context)
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

                # Log this attempt
                success = self._is_valid_response(response)
                self.debugger.log_extraction_attempt(strategy_name, prompt, response, success)

                # Check if this strategy succeeded
                if success:
                    logger.info(f"Extraction succeeded with strategy: {description}")
                    return response
                else:
                    logger.warning(f"Strategy {strategy_name} failed: {response.get('error', 'Invalid response')}")

            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed with exception: {str(e)}")
                # Log the failed attempt
                error_response = {"error": str(e)}
                self.debugger.log_extraction_attempt(strategy_name, "Error occurred before prompt generation", error_response, False)
                continue

        # Final fallback: create a minimal valid response
        logger.error("All extraction strategies failed, creating fallback response")
        self.debugger.add_error("All extraction strategies failed")
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
            "debug_file_path": self.debugger.debug_file_path,
        }
