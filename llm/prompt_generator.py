"""
LLM prompt generation and optimization

Extracted from llm/llm_service.py
"""
import logging
import time
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Import prompt utilities
try:
    from llm.prompts.extraction_prompt import generate_extraction_prompt
    from llm.prompts.formatting_prompt import generate_formatting_prompt, check_formatting_quality
    from llm.prompts.system_message import get_system_message
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


class PromptGenerator:
    """Generates and optimizes prompts for LLM requests"""

    def __init__(self):
        """Initialize prompt generator"""
        self.system_message = get_system_message()

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
                from utils.config import app_config
                from utils.error_logging import log_error

                log_error(
                    truncation_msg,
                    error_type="WARNING",
                    source="PromptGenerator.generate_extraction_prompt",
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
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} - WARNING - {truncation_msg}\n"
                    )

            # Truncate context without splitting sentences
            context = self._truncate_context_smartly(context, max_context_tokens)

        return generate_extraction_prompt(query, context, version="v2")

    def generate_formatting_prompt(self, raw_answer: str) -> str:
        """
        Generate a prompt for formatting the raw answer.

        Args:
            raw_answer: The raw answer from the extraction model

        Returns:
            A formatted prompt string
        """
        return generate_formatting_prompt(raw_answer)

    def _truncate_context_smartly(self, context: str, max_tokens: int) -> str:
        """
        Truncate context intelligently without breaking sentences

        Args:
            context: The context to truncate
            max_tokens: Maximum number of tokens to allow

        Returns:
            Truncated context string
        """
        # First, split by sentences
        sentences = re.split(r"(?<=[.!?]) +", context)

        # Calculate approximate tokens per sentence
        tokens_per_sentence = [len(s.split()) * 1.3 for s in sentences]

        # Add sentences until we reach the limit
        truncated_context = []
        current_tokens = 0

        for i, (sentence, tokens) in enumerate(zip(sentences, tokens_per_sentence)):
            if current_tokens + tokens <= max_tokens:
                truncated_context.append(sentence)
                current_tokens += tokens
            else:
                break

        # Join sentences back together
        result = " ".join(truncated_context)

        # Add truncation notice
        result += "\n\n[Note: Context has been truncated due to length limitations.]"

        return result

    def generate_enhanced_extraction_prompt(self, query: str, context: str, retry_count: int = 0) -> str:
        """
        Generate an enhanced extraction prompt with explicit JSON instructions

        Args:
            query: The user's question
            context: The retrieved context
            retry_count: Number of previous attempts (for increasingly explicit instructions)

        Returns:
            Enhanced prompt with explicit formatting instructions
        """
        base_prompt = self.generate_extraction_prompt(query, context)

        if retry_count == 0:
            return base_prompt
        elif retry_count == 1:
            return (
                base_prompt
                + '\n\nIMPORTANT: You MUST respond with valid JSON only. Example:\n{{\n  "answer": "your answer here",\n  "sources": ["source1", "source2"],\n  "confidence": 0.9\n}}'
            )
        else:
            return (
                base_prompt
                + '\n\nCRITICAL: You MUST respond with ONLY valid JSON. No additional text before or after. Example:\n{{\n  "answer": "your comprehensive answer here",\n  "sources": ["document section 1", "document section 2"],\n  "confidence": 0.95\n}}'
            )

    def generate_simple_formatting_prompt(self, raw_answer: str) -> str:
        """
        Generate a simple formatting prompt for retry scenarios

        Args:
            raw_answer: The raw answer to format

        Returns:
            Simple formatting prompt
        """
        return f"Format the following text to be clear and readable:\n\n{raw_answer}"

    def get_system_message(self) -> str:
        """
        Get the system message for LLM requests

        Returns:
            System message string
        """
        return self.system_message

    def validate_context_size(self, context: str, max_tokens: int = 4096) -> bool:
        """
        Check if context is within acceptable size limits

        Args:
            context: The context to check
            max_tokens: Maximum allowed tokens

        Returns:
            True if context is within limits, False otherwise
        """
        approx_token_count = len(context.split()) * 1.3
        return approx_token_count <= max_tokens

    def get_context_stats(self, context: str) -> dict:
        """
        Get statistics about the context

        Args:
            context: The context to analyze

        Returns:
            Dictionary with context statistics
        """
        words = context.split()
        sentences = re.split(r"[.!?]+", context)

        return {
            "character_count": len(context),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "estimated_tokens": len(words) * 1.3,
            "avg_words_per_sentence": len(words) / max(len([s for s in sentences if s.strip()]), 1)
        }
