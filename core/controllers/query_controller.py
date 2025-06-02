"""Query Controller for WorkApp2.

Handles query processing, async operations, and result display.
Extracted from the monolithic workapp3.py for better maintainability.

Modernized async patterns using asyncio.run() and proper event loop handling.
"""

import streamlit as st
import logging
import asyncio
import time
from typing import Optional, Dict, Any

# Type hints for internal modules are allowed to use # type: ignore with reason
from core.document_processor import DocumentProcessor  # type: ignore[import] # TODO: Add proper types
from llm.services.llm_service import LLMService  # type: ignore[import] # TODO: Add proper types
from retrieval.retrieval_system import UnifiedRetrievalSystem  # type: ignore[import] # TODO: Add proper types
from utils.ui import (  # type: ignore[import] # TODO: Add proper UI types
    QueryProgressTracker,
    display_enhanced_answer,
    display_error_message,
)
from utils.error_handling.enhanced_decorators import with_advanced_retry, with_error_tracking  # type: ignore[import] # TODO: Add proper types


logger = logging.getLogger(__name__)


class QueryController:
    """Controller responsible for query processing and result display."""

    def __init__(self, app_orchestrator: Optional[Any] = None, production_mode: bool = False) -> None:
        """Initialize the query controller.

        Args:
            app_orchestrator: The main application orchestrator for service coordination
            production_mode: Whether the application is running in production mode
        """
        self.app_orchestrator = app_orchestrator
        self.production_mode = production_mode
        self.logger = logger

    @with_advanced_retry(max_attempts=2, backoff_factor=1.5)
    @with_error_tracking()
    async def process_query_async(
        self,
        query: str,
        doc_processor: DocumentProcessor,
        llm_service: LLMService,
        retrieval_system: UnifiedRetrievalSystem,
        debug_mode: bool,
        progress_tracker: Optional[QueryProgressTracker] = None,
    ) -> Dict[str, Any]:
        """Process a query asynchronously with progress tracking.

        Args:
            query: The user's question
            doc_processor: Document processor instance
            llm_service: LLM service instance
            retrieval_system: Retrieval system instance
            debug_mode: Whether to show debug information
            progress_tracker: Query progress tracker for UI updates

        Returns:
            Dictionary with processing results

        Raises:
            RuntimeError: If any processing step fails
        """
        start_time = time.time()
        results = {}
        clean_ctx = None

        # Initialize progress tracker if provided
        if progress_tracker is not None:
            progress_tracker.initialize()

        # 1) Retrieve context with parallel processing
        if progress_tracker is not None:
            progress_tracker.update_stage("retrieval")

        try:
            clean_ctx, retrieval_time, num_chunks, retrieval_scores = retrieval_system.retrieve(query)

            # Update progress tracker with the highest retrieval score if available
            if progress_tracker is not None and retrieval_scores and len(retrieval_scores) > 0:
                max_score = max(retrieval_scores) if retrieval_scores else 0.0
                avg_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0
                progress_tracker.update_stage(
                    "retrieval", score=avg_score, max_score=1.0
                )  # FAISS scores are typically between 0 and 1

            results["retrieval"] = {
                "context": clean_ctx,
                "time": retrieval_time,
                "chunks": num_chunks,
                "scores": retrieval_scores,
            }

            if not clean_ctx:
                error_msg = "No relevant context found for this query. This could be because:\n"
                error_msg += "1. The similarity threshold may be too high\n"
                error_msg += "2. The index may not contain relevant information for this query\n"
                error_msg += "3. The query may need to be rephrased\n"
                error_msg += "Try using the 'Debug Index' button to diagnose the issue."
                self.logger.warning("No relevant context found for this query")

                # Update progress tracker if provided
                if progress_tracker is not None:
                    progress_tracker.complete(success=False)

                return {"error": error_msg, "retrieval": results.get("retrieval")}

        except Exception as e:
            error_msg = f"Error retrieving context: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            # Update progress tracker if provided
            if progress_tracker is not None:
                progress_tracker.complete(success=False)

            return {"error": error_msg}

        # Debug information
        if debug_mode and clean_ctx:
            with st.expander("Retrieved Context"):
                st.text_area("Context", clean_ctx, height=200)

        # 2) Extract answer and format in parallel
        if progress_tracker is not None:
            progress_tracker.update_stage("extraction")

        try:
            # Process both extraction and formatting in parallel
            if not hasattr(llm_service, "process_extraction_and_formatting") or not callable(
                getattr(llm_service, "process_extraction_and_formatting", None)
            ):
                raise AttributeError(
                    "LLM service does not have a properly implemented 'process_extraction_and_formatting' method"
                )

            extraction_response, formatting_response = (
                await llm_service.process_extraction_and_formatting(query, clean_ctx)
            )

            # Check for errors in extraction
            if "error" in extraction_response:
                error_msg = f"Error from extraction model: {extraction_response['error']}"
                self.logger.error(error_msg)

                # Update progress tracker if provided
                if progress_tracker is not None:
                    progress_tracker.complete(success=False)

                return {"error": error_msg, "retrieval": results.get("retrieval")}

            # Store extraction results
            answer_raw = extraction_response["content"]
            results["extraction"] = {"response": extraction_response, "raw_answer": answer_raw}

            # Update progress for formatting
            if progress_tracker is not None:
                progress_tracker.update_stage("formatting")

            # Check for errors in formatting
            if "error" in formatting_response:
                error_msg = f"Error from formatting model: {formatting_response['error']}"
                self.logger.error(error_msg)

                # Update progress tracker if provided
                if progress_tracker is not None:
                    progress_tracker.complete(success=False)

                return {
                    "error": error_msg,
                    "retrieval": results.get("retrieval"),
                    "extraction": results.get("extraction"),
                }

            # Store formatting results
            answer_formatted = formatting_response["content"]
            results["formatting"] = {
                "response": formatting_response,
                "formatted_answer": answer_formatted,
            }

            # Mark progress as complete
            if progress_tracker is not None:
                progress_tracker.update_stage("complete")
                progress_tracker.complete()

            # Calculate total processing time
            total_time = time.time() - start_time
            results["total_time"] = total_time

            return results

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.logger.error(error_msg, exc_info=True)

            # Update progress tracker if provided
            if progress_tracker is not None:
                progress_tracker.complete(success=False)

            return {
                "error": error_msg,
                "retrieval": results.get("retrieval"),
                "extraction": results.get("extraction", {}),
            }

    def process_query(
        self,
        query: str,
        debug_mode: bool = False
    ) -> None:
        """Process a query and display the results in the Streamlit UI.

        Args:
            query: The user's question
            debug_mode: Whether to show debug information
        """
        if not self.app_orchestrator:
            st.error("Application orchestrator not available")
            return

        try:
            # Get services
            doc_processor, llm_service, retrieval_system = self.app_orchestrator.get_services()
        except Exception as e:
            st.error(f"Failed to get services: {str(e)}")
            return

        # Create a progress tracker with enhanced metrics display
        progress_tracker = QueryProgressTracker()

        # Create placeholders for results
        answer_placeholder = st.empty()
        metrics_placeholder = st.empty()

        # Process the query asynchronously
        try:
            # Modern async pattern using asyncio.run()
            try:
                # Try to get existing event loop
                loop = asyncio.get_running_loop()
                # If we're already in an async context, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.process_query_async(
                            query, doc_processor, llm_service, retrieval_system, debug_mode, progress_tracker
                        )
                    )
                    results = future.result()
            except RuntimeError:
                # No event loop running, use asyncio.run() directly
                results = asyncio.run(
                    self.process_query_async(
                        query, doc_processor, llm_service, retrieval_system, debug_mode, progress_tracker
                    )
                )

            # Check for errors
            if "error" in results:
                display_error_message(
                    results["error"],
                    suggestions=[
                        "Try rephrasing your question",
                        "Upload more relevant documents",
                        "Check the 'Debug Index' to see what content is available",
                    ],
                )
                return

            # Display the formatted answer
            if "formatting" in results and "formatted_answer" in results["formatting"]:
                formatted_answer = results["formatting"]["formatted_answer"]
                display_enhanced_answer(formatted_answer)

                # Display retrieval metrics if available
                if (
                    "retrieval" in results
                    and "scores" in results["retrieval"]
                    and results["retrieval"]["scores"]
                ):
                    scores = results["retrieval"]["scores"]
                    if scores:
                        avg_score = sum(scores) / len(scores) if scores else 0
                        max_score = max(scores) if scores else 0
                        # Display a small progress bar showing the max retrieval score
                        with metrics_placeholder.container():
                            st.caption("Retrieval Quality")
                            st.progress(min(1.0, max_score))
                            st.caption(f"Max score: {max_score:.4f}, Avg score: {avg_score:.4f}")

                # Display raw context if enabled
                if (
                    st.session_state.get("show_raw_context", False)
                    and "retrieval" in results
                    and "context" in results["retrieval"]
                ):
                    with st.expander("Raw Context"):
                        st.text_area("Retrieved Context", results["retrieval"]["context"], height=300)
                        # Display retrieval scores if available
                        if "scores" in results["retrieval"] and results["retrieval"]["scores"]:
                            scores = results["retrieval"]["scores"]
                            avg_score = sum(scores) / len(scores) if scores else 0
                            max_score = max(scores) if scores else 0
                            st.write(
                                f"Retrieval metrics: Avg score: {avg_score:.4f}, Max score: {max_score:.4f}"
                            )

                # Log successful query
                self.logger.info(f"Successfully processed query: {query[:50]}...")

                # Display debug information if enabled
                self._display_debug_info(debug_mode, results, query, formatted_answer)

            else:
                display_error_message(
                    "Failed to generate a formatted answer",
                    suggestions=[
                        "Try rephrasing your question",
                        "Check the application logs for more details",
                    ],
                )

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            display_error_message(
                error_msg,
                suggestions=["Try again later", "Check the application logs for more details"],
            )

    def _display_debug_info(
        self,
        debug_mode: bool,
        results: Dict[str, Any],
        query: str,
        formatted_answer: str
    ) -> None:
        """Display debug information if debug mode is enabled.

        Args:
            debug_mode: Whether debug mode is enabled
            results: Query processing results
            query: Original query
            formatted_answer: Final formatted answer
        """
        if debug_mode and "retrieval" in results and "extraction" in results:
            with st.expander("Debug Information"):
                st.subheader("Query")
                st.text(query)

                st.subheader("Context")
                if "context" in results["retrieval"]:
                    st.text_area(
                        "Retrieved Context", results["retrieval"]["context"], height=200
                    )

                st.subheader("Raw Answer")
                if "raw_answer" in results["extraction"]:
                    st.text_area(
                        "Extracted Answer", results["extraction"]["raw_answer"], height=150
                    )

                st.subheader("Formatted Answer")
                st.text_area("Final Answer", formatted_answer, height=150)

                st.subheader("Performance")
                if "total_time" in results:
                    st.write(f"Total processing time: {results['total_time']:.2f} seconds")
                if "retrieval" in results and "time" in results["retrieval"]:
                    st.write(f"Retrieval time: {results['retrieval']['time']:.2f} seconds")
                if "retrieval" in results and "chunks" in results["retrieval"]:
                    st.write(f"Chunks retrieved: {results['retrieval']['chunks']}")

    def validate_query_inputs(self, query: str, has_index: bool) -> Optional[str]:
        """Validate query inputs and return error message if invalid.

        Args:
            query: The user's query
            has_index: Whether the system has an index available

        Returns:
            Error message if validation fails, None if valid
        """
        if not query or query.strip() == "":
            return "Please enter a question"

        if not has_index:
            return "Please upload documents first to build an index"

        if len(query.strip()) < 3:
            return "Please enter a more detailed question (at least 3 characters)"

        return None

    def get_query_suggestions(self) -> list[str]:
        """Get suggested queries for the user.

        Returns:
            List of suggested query strings
        """
        return [
            "What is the main topic of these documents?",
            "Can you summarize the key points?",
            "What are the most important findings?",
            "What recommendations are made?",
            "What methodology was used?",
        ]
