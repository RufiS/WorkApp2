"""
Response streaming service for faster perceived performance
Allows partial responses to be shown as they arrive
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass
class StreamingChunk:
    """Represents a streaming response chunk"""
    content: str
    is_complete: bool = False
    chunk_index: int = 0
    total_content: str = ""
    response_time_ms: float = 0
    error: Optional[str] = None


class StreamingService:
    """Service for handling streaming LLM responses"""

    def __init__(self, api_key: str, http_client=None):
        """
        Initialize streaming service

        Args:
            api_key: OpenAI API key
            http_client: Optional HTTP client for connection reuse
        """
        self.api_key = api_key
        self._external_http_client = http_client is not None
        
        # Initialize async client with optional HTTP client
        if http_client:
            self.async_client = AsyncOpenAI(api_key=api_key, http_client=http_client)
        else:
            self.async_client = AsyncOpenAI(api_key=api_key)
        
        # Streaming statistics
        self.streaming_stats = {
            "streams_started": 0,
            "streams_completed": 0,
            "total_chunks_sent": 0,
            "avg_time_to_first_chunk_ms": 0,
            "avg_total_streaming_time_ms": 0
        }

    async def stream_chat_completion(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        system_message: Optional[str] = None
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Stream chat completion responses chunk by chunk

        Args:
            prompt: The prompt to send
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            system_message: Optional system message

        Yields:
            StreamingChunk objects with partial responses
        """
        start_time = time.time()
        chunk_index = 0
        total_content = ""
        first_chunk_time = None
        
        try:
            self.streaming_stats["streams_started"] += 1
            
            # Prepare messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            # Create streaming completion
            stream = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                timeout=60.0
            )

            # Process streaming chunks
            async for chunk in stream:
                chunk_time = time.time()
                
                # Track first chunk timing
                if first_chunk_time is None:
                    first_chunk_time = chunk_time
                    time_to_first_chunk = (chunk_time - start_time) * 1000
                else:
                    time_to_first_chunk = 0

                # Extract content from chunk
                content = ""
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content

                # Update totals
                total_content += content
                chunk_index += 1
                self.streaming_stats["total_chunks_sent"] += 1

                # Create streaming chunk
                streaming_chunk = StreamingChunk(
                    content=content,
                    is_complete=False,
                    chunk_index=chunk_index,
                    total_content=total_content,
                    response_time_ms=time_to_first_chunk
                )

                yield streaming_chunk

                # Small delay to prevent overwhelming the receiver
                if chunk_index % 5 == 0:  # Every 5th chunk
                    await asyncio.sleep(0.01)  # 10ms delay

            # Send final completion chunk
            total_time = (time.time() - start_time) * 1000
            final_chunk = StreamingChunk(
                content="",
                is_complete=True,
                chunk_index=chunk_index,
                total_content=total_content,
                response_time_ms=total_time
            )

            self.streaming_stats["streams_completed"] += 1
            self._update_streaming_stats(
                time_to_first_chunk=(first_chunk_time - start_time) * 1000 if first_chunk_time else 0,
                total_time=total_time
            )

            yield final_chunk

        except Exception as e:
            error_msg = f"Streaming error: {str(e)}"
            logger.error(error_msg)
            
            # Send error chunk
            error_chunk = StreamingChunk(
                content="",
                is_complete=True,
                chunk_index=chunk_index,
                total_content=total_content,
                response_time_ms=(time.time() - start_time) * 1000,
                error=error_msg
            )
            
            yield error_chunk

    async def stream_with_callback(
        self,
        prompt: str,
        callback: Callable[[StreamingChunk], None],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stream response with callback function for real-time processing

        Args:
            prompt: The prompt to send
            callback: Function to call with each chunk
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Generation temperature
            system_message: Optional system message

        Returns:
            Final response summary
        """
        total_content = ""
        chunk_count = 0
        start_time = time.time()
        error = None

        try:
            async for chunk in self.stream_chat_completion(
                prompt, model, max_tokens, temperature, system_message
            ):
                # Call the provided callback
                callback(chunk)
                
                # Update totals
                total_content = chunk.total_content
                chunk_count = chunk.chunk_index
                
                if chunk.error:
                    error = chunk.error
                    break

        except Exception as e:
            error = str(e)
            logger.error(f"Streaming with callback failed: {e}")

        # Return final summary
        total_time = (time.time() - start_time) * 1000
        return {
            "content": total_content,
            "chunks_received": chunk_count,
            "total_time_ms": round(total_time, 2),
            "error": error,
            "success": error is None
        }

    async def stream_to_buffer(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.0,
        system_message: Optional[str] = None,
        buffer_size: int = 100
    ) -> AsyncGenerator[str, None]:
        """
        Stream response to a buffer, yielding when buffer is full or complete

        Args:
            prompt: The prompt to send
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Generation temperature
            system_message: Optional system message
            buffer_size: Size of buffer before yielding

        Yields:
            Buffered content strings
        """
        buffer = ""
        
        async for chunk in self.stream_chat_completion(
            prompt, model, max_tokens, temperature, system_message
        ):
            if chunk.error:
                if buffer:  # Yield any remaining buffer content
                    yield buffer
                raise Exception(chunk.error)
            
            buffer += chunk.content
            
            # Yield buffer when it reaches size or stream is complete
            if len(buffer) >= buffer_size or chunk.is_complete:
                if buffer:  # Only yield if buffer has content
                    yield buffer
                    buffer = ""  # Reset buffer

    def _update_streaming_stats(
        self, time_to_first_chunk: float, total_time: float
    ) -> None:
        """
        Update streaming performance statistics

        Args:
            time_to_first_chunk: Time to first chunk in ms
            total_time: Total streaming time in ms
        """
        # Update time to first chunk average
        completed = self.streaming_stats["streams_completed"]
        if completed == 1:
            self.streaming_stats["avg_time_to_first_chunk_ms"] = time_to_first_chunk
        else:
            current_avg = self.streaming_stats["avg_time_to_first_chunk_ms"]
            self.streaming_stats["avg_time_to_first_chunk_ms"] = (
                (current_avg * (completed - 1) + time_to_first_chunk) / completed
            )

        # Update total streaming time average
        if completed == 1:
            self.streaming_stats["avg_total_streaming_time_ms"] = total_time
        else:
            current_avg = self.streaming_stats["avg_total_streaming_time_ms"]
            self.streaming_stats["avg_total_streaming_time_ms"] = (
                (current_avg * (completed - 1) + total_time) / completed
            )

    def get_streaming_stats(self) -> Dict[str, Any]:
        """
        Get streaming performance statistics

        Returns:
            Dictionary with streaming stats
        """
        completion_rate = 0
        if self.streaming_stats["streams_started"] > 0:
            completion_rate = (
                self.streaming_stats["streams_completed"] / 
                self.streaming_stats["streams_started"]
            )

        return {
            "streams_started": self.streaming_stats["streams_started"],
            "streams_completed": self.streaming_stats["streams_completed"],
            "completion_rate": round(completion_rate, 3),
            "total_chunks_sent": self.streaming_stats["total_chunks_sent"],
            "avg_chunks_per_stream": round(
                self.streaming_stats["total_chunks_sent"] / 
                max(self.streaming_stats["streams_completed"], 1), 2
            ),
            "avg_time_to_first_chunk_ms": round(
                self.streaming_stats["avg_time_to_first_chunk_ms"], 2
            ),
            "avg_total_streaming_time_ms": round(
                self.streaming_stats["avg_total_streaming_time_ms"], 2
            )
        }

    def reset_stats(self) -> None:
        """Reset streaming statistics"""
        self.streaming_stats = {
            "streams_started": 0,
            "streams_completed": 0,
            "total_chunks_sent": 0,
            "avg_time_to_first_chunk_ms": 0,
            "avg_total_streaming_time_ms": 0
        }

    async def close(self) -> None:
        """Properly close async resources"""
        try:
            if hasattr(self, 'async_client') and not self._external_http_client:
                await self.async_client.close()
            logger.debug("Streaming service closed successfully")
        except Exception as e:
            logger.warning(f"Error closing streaming service: {e}")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


class StreamingAnswerPipeline:
    """Answer pipeline with streaming support for real-time responses"""

    def __init__(self, llm_service, prompt_generator, streaming_service=None):
        """
        Initialize streaming answer pipeline

        Args:
            llm_service: LLM service instance
            prompt_generator: Prompt generator instance
            streaming_service: Optional streaming service
        """
        self.llm_service = llm_service
        self.prompt_generator = prompt_generator
        
        # Initialize streaming service
        if streaming_service:
            self.streaming_service = streaming_service
        else:
            # Create streaming service with same API key as LLM service
            api_key = getattr(llm_service, 'api_key', None)
            if api_key:
                # Reuse HTTP client if available
                http_client = getattr(llm_service, '_http_client', None)
                self.streaming_service = StreamingService(api_key, http_client)
            else:
                self.streaming_service = None
                logger.warning("No API key available for streaming service")

    async def stream_answer_generation(
        self,
        query: str,
        context: str,
        callback: Optional[Callable[[StreamingChunk], None]] = None
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Stream answer generation with real-time updates

        Args:
            query: User query
            context: Retrieved context
            callback: Optional callback for each chunk

        Yields:
            StreamingChunk objects with partial answers
        """
        if not self.streaming_service:
            raise ValueError("Streaming service not available")

        try:
            # Generate extraction prompt
            from llm.prompts.extraction_prompt import generate_extraction_prompt_simple
            prompt = generate_extraction_prompt_simple(query, context)

            # Import model config
            from core.config import model_config
            
            # Get system message
            system_message = self.prompt_generator.get_system_message() if hasattr(
                self.prompt_generator, 'get_system_message'
            ) else None

            # Stream the response
            async for chunk in self.streaming_service.stream_chat_completion(
                prompt=prompt,
                model=model_config.extraction_model,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                system_message=system_message
            ):
                if callback:
                    callback(chunk)
                yield chunk

        except Exception as e:
            logger.error(f"Streaming answer generation failed: {e}")
            # Yield error chunk
            error_chunk = StreamingChunk(
                content="",
                is_complete=True,
                error=str(e)
            )
            if callback:
                callback(error_chunk)
            yield error_chunk

    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        if self.streaming_service:
            return self.streaming_service.get_streaming_stats()
        return {"error": "Streaming service not available"}
