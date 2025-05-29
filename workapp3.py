# ----------------------------------------------------------------
#  Enhanced Streamlit QA app with improved architecture and performance
#  Version 0.4.0
# ----------------------------------------------------------------
import streamlit as st
import asyncio
import logging
import sys
import time
import re
import json
import os
import tempfile
import numpy as np
import faiss
import argparse
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Parse command line arguments
def parse_args():
    """
    Parse command line arguments for the application.

    Returns:
        argparse.Namespace: The parsed command line arguments

    Arguments:
        --dry-run: Run the application without saving index changes to disk.
                  This is useful for testing document processing without modifying
                  the existing index.
    """
    parser = argparse.ArgumentParser(description="WorkApp Document QA System")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview index changes without writing to disk"
    )
    return parser.parse_args()


# Get command line arguments
args = parse_args()

# Configure logger
logger = logging.getLogger(__name__)

# Avoid occasional import bugs
sys.modules["torch.classes"] = None

# Import utility modules
from utils.config import (
    app_config,
    model_config,
    retrieval_config,
    ui_config,
    performance_config,
    config_manager,
)
from core.document_processor import DocumentProcessor
from llm.llm_service import LLMService
from retrieval.retrieval_system import UnifiedRetrievalSystem
from utils.index_management.index_coordinator import index_coordinator
from utils.text_processing.context_processing import clean_context, extract_hyperlinks
from error_handling.enhanced_decorators import with_advanced_retry, with_error_tracking

# Import UI components from existing modules (removing all duplicate fallback code)
from utils.ui import (
    ProgressManager,
    QueryProgressTracker,
    display_enhanced_answer,
    display_search_results,
    display_error_message,
    display_system_status,
    create_collapsible_section,
    display_answer,
    display_debug_info,
    display_debug_info_for_index,
    display_confidence_meter,
    smart_wrap,
)
from utils.ui.config_sidebar import render_config_sidebar

# ─── Setup logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)

# ─── Initialize services ─────────────────────────────────────────
# Use ui_config from the config module
# Note: ui_config is imported from utils.config above


@st.cache_resource
@with_advanced_retry(max_attempts=3, backoff_factor=2.0)
def initialize_services() -> Tuple[DocumentProcessor, LLMService, UnifiedRetrievalSystem]:
    """
    Initialize and cache the application services

    Returns:
        Tuple containing document processor, LLM service, and retrieval system

    Raises:
        RuntimeError: If service initialization fails
    """
    try:
        # Use the unified document processor
        doc_processor = DocumentProcessor()

        # Use the consolidated LLM service
        llm_service = LLMService(app_config.api_keys.get("openai", ""))

        # Use the unified retrieval system
        retrieval_system = UnifiedRetrievalSystem(doc_processor)
        logger.info("Services initialized successfully")

        # Ensure index is loaded if it exists
        if doc_processor.has_index() and (
            doc_processor.index is None or doc_processor.texts is None
        ):
            try:
                doc_processor.load_index(retrieval_config.index_path)
                logger.info("Loaded existing index")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {str(e)}")
                # Continue without index - user can upload files

        return doc_processor, llm_service, retrieval_system
    except Exception as e:
        error_msg = f"Failed to initialize services: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg)


# ─── Query processing ────────────────────────────────────────────
@with_advanced_retry(max_attempts=2, backoff_factor=1.5)
@with_error_tracking()
async def process_query_async(
    query: str,
    doc_processor: DocumentProcessor,
    llm_service: LLMService,
    retrieval_system: UnifiedRetrievalSystem,
    debug_mode: bool,
    progress_tracker: Optional[QueryProgressTracker] = None,
) -> Dict[str, Any]:
    """
    Process a query asynchronously with progress tracking

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
            error_msg += (
                "1. The similarity threshold may be too high (currently "
                + str(retrieval_config.similarity_threshold)
                + ")\n"
            )
            error_msg += "2. The index may not contain relevant information for this query\n"
            error_msg += "3. The query may need to be rephrased\n"
            error_msg += "Try using the 'Debug Index' button to diagnose the issue."
            logger.warning("No relevant context found for this query")

            # Update progress tracker if provided
            if progress_tracker is not None:
                progress_tracker.complete(success=False)

            return {"error": error_msg, "retrieval": results.get("retrieval")}

    except Exception as e:
        error_msg = f"Error retrieving context: {str(e)}"
        logger.error(error_msg, exc_info=True)

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
            logger.error(error_msg)

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
            logger.error(error_msg)

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
        logger.error(error_msg, exc_info=True)

        # Update progress tracker if provided
        if progress_tracker is not None:
            progress_tracker.complete(success=False)

        return {
            "error": error_msg,
            "retrieval": results.get("retrieval"),
            "extraction": results.get("extraction", {}),
        }
def process_query(query, doc_processor, llm_service, retrieval_system, debug_mode=False):
    """
    Process a query and display the results in the Streamlit UI

    Args:
        query: The user's question
        doc_processor: Document processor instance
        llm_service: LLM service instance
        retrieval_system: Retrieval system instance
        debug_mode: Whether to show debug information
    """
    # Create a progress tracker with enhanced metrics display
    progress_tracker = QueryProgressTracker()

    # Create a placeholder for the answer
    answer_placeholder = st.empty()

    # Add a metrics section to display retrieval quality
    metrics_placeholder = st.empty()

    # Process the query asynchronously
    try:
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(
            process_query_async(
                query, doc_processor, llm_service, retrieval_system, debug_mode, progress_tracker
            )
        )
        loop.close()

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
            logger.info(f"Successfully processed query: {query[:50]}...")

            # Display debug information if enabled
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
        logger.error(error_msg, exc_info=True)
        display_error_message(
            error_msg,
            suggestions=["Try again later", "Check the application logs for more details"],
        )


# ─── Main application ────────────────────────────────────────────
def main():
    # Set page config
    st.set_page_config(
        page_title=app_config.page_title,
        page_icon=app_config.icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS if available
    if hasattr(ui_config, "custom_css") and ui_config.custom_css:
        st.markdown(ui_config.custom_css, unsafe_allow_html=True)

    # Display application header
    st.title(app_config.page_title)
    if app_config.subtitle:
        st.subheader(app_config.subtitle)

    # Display dry-run mode indicator if enabled
    if args.dry_run:
        st.warning("⚠️ DRY RUN MODE: Index changes will not be saved to disk")

    # Add version info in footer
    with st.container():
        st.caption(f"Application version: {getattr(app_config, 'version', '0.4.0')}")

    # Initialize services with proper error handling
    try:
        doc_processor, llm_service, retrieval_system = initialize_services()
    except Exception as e:
        display_error_message(
            f"Failed to initialize services: {str(e)}",
            suggestions=[
                "Check your OpenAI API key",
                "Verify that required dependencies are installed",
                "Check the application logs for more details",
            ],
        )
        st.stop()

    # File uploader with clear instructions (if enabled)
    uploads = None
    if ui_config.show_document_upload:
        st.subheader("Document Upload")
        st.write("Upload PDF, TXT, or DOCX files to build a searchable index.")

        # Add a button to clear the index and file uploader
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            uploads = st.file_uploader(
                "Choose files",
                type=["pdf", "txt", "docx", "doc"],
                accept_multiple_files=True,
                help="Select PDF, TXT, or DOCX files to process",
            )
        with col2:
            if ui_config.show_clear_index_button and st.button(
                "Clear Index", help="Remove all documents from the index"
            ):
                try:
                    doc_processor.clear_index()
                    # Also clear the processed files tracking
                    if "processed_files" in st.session_state:
                        st.session_state.processed_files = set()

                    # Check if we're in dry-run mode
                    if args.dry_run:
                        st.success("Index cleared in memory only (dry-run mode)")
                        logger.info("Dry run mode: Index cleared in memory only")
                    else:
                        st.success("Index cleared successfully!")
                        logger.info("Index cleared successfully")
                    # Don't rerun here, let the user upload new files
                except Exception as e:
                    display_error_message(f"Error clearing index: {str(e)}")
        with col3:
            if st.button(
                "Reset Tracking", help="Reset the file tracking without clearing the index"
            ):
                if "processed_files" in st.session_state:
                    st.session_state.processed_files = set()
                st.success("File tracking reset. You can re-upload the same files.")

    # Process uploads using existing architecture
    if uploads:
        # Initialize session state for tracking
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()

        # Create progress manager
        progress_manager = ProgressManager(total_steps=len(uploads) + 1)
        progress_manager.initialize("Processing files...")

        # Process files using existing DocumentProcessor
        all_chunks = []
        completed_files = []
        current_file_names = {file.name for file in uploads}

        # Check if files were already processed
        already_processed = current_file_names.issubset(st.session_state.processed_files)
        if already_processed:
            st.info("These files have already been processed in this session.")
            logger.info("Files already processed, skipping upload processing.")
        else:
            for i, file in enumerate(uploads):
                try:
                    progress_manager.update(i, f"Processing {file.name}...")
                    
                    # Use existing DocumentProcessor.process_file() method
                    chunks = doc_processor.process_file(file)
                    
                    if chunks:
                        all_chunks.extend(chunks)
                        completed_files.append(file)
                        logger.info(f"Successfully processed {file.name}: {len(chunks)} chunks")
                    else:
                        st.warning(f"No content extracted from {file.name}. File may be empty or unsupported.")
                        
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    logger.error(f"Error processing {file.name}: {str(e)}", exc_info=True)

            # Update index if we have chunks
            if all_chunks:
                try:
                    progress_manager.update(len(uploads), "Updating index...")
                    
                    # Extract texts from chunks for index coordinator
                    texts = [chunk.get('text', chunk.get('content', '')) for chunk in all_chunks if chunk.get('text') or chunk.get('content')]
                    
                    if texts:
                        # Use existing index coordinator for index updates
                        embeddings = doc_processor.batch_embed_chunks(texts)
                        success, message = index_coordinator.update_index(
                            texts, embeddings, append=True, dry_run=args.dry_run
                        )
                        
                        if success:
                            # CRITICAL FIX: Reload DocumentProcessor index after coordinator update
                            if not args.dry_run:
                                try:
                                    doc_processor.load_index(retrieval_config.index_path)
                                    logger.info("DocumentProcessor reloaded updated index")
                                except Exception as e:
                                    logger.warning(f"Failed to reload DocumentProcessor index: {str(e)}")
                            
                            # Update processed files tracking
                            st.session_state.processed_files.update(current_file_names)
                            
                            if args.dry_run:
                                st.success(f"Processed {len(completed_files)} files successfully! (Dry-run mode - changes not saved)")
                                logger.info(f"Index updated in memory only (dry-run mode): {message}")
                            else:
                                st.success(f"Processed {len(completed_files)} files successfully!")
                                logger.info(f"Index updated successfully: {message}")
                            
                            # Show index statistics
                            if ui_config.show_index_statistics and hasattr(doc_processor, "get_metrics"):
                                metrics = doc_processor.get_metrics()
                                st.info(f"Index contains {metrics.get('total_chunks', 0)} chunks from {len(st.session_state.processed_files)} files")
                                
                                with st.expander("Files in Index"):
                                    for file_name in sorted(st.session_state.processed_files):
                                        st.write(f"- {file_name}")
                                
                                with st.expander("Index Statistics"):
                                    st.json(metrics)
                        else:
                            st.error(f"Failed to update index: {message}")
                            logger.error(f"Failed to update index: {message}")
                    else:
                        st.error("No valid text content found in uploaded files.")
                        
                except Exception as e:
                    st.error(f"Error updating index: {str(e)}")
                    logger.error(f"Error updating index: {str(e)}", exc_info=True)
            else:
                st.warning("No content extracted from any uploaded files.")
                
        progress_manager.complete()
    
    elif doc_processor.has_index():
        # Load existing index if available
        try:
            if doc_processor.index is None or doc_processor.texts is None:
                doc_processor.load_index(retrieval_config.index_path)
                logger.info("Loaded existing index")

                if ui_config.show_index_statistics:
                    if hasattr(doc_processor, "get_metrics"):
                        metrics = doc_processor.get_metrics()
                        st.info(f"Index contains {metrics.get('total_chunks', 0)} chunks")

                        # Initialize processed_files if not already in session state
                        if "processed_files" not in st.session_state:
                            st.session_state.processed_files = set()
                            # Try to get processed files from metadata if available
                            try:
                                from utils.config import resolve_path

                                resolved_index_dir = resolve_path(retrieval_config.index_path)
                                metadata_path = os.path.join(resolved_index_dir, "metadata.json")
                                if os.path.exists(metadata_path):
                                    with open(metadata_path, "r") as f:
                                        metadata = json.load(f)
                                        if "processed_files" in metadata:
                                            st.session_state.processed_files = set(
                                                metadata["processed_files"]
                                            )
                            except Exception as e:
                                logger.warning(
                                    f"Could not load processed files from metadata: {str(e)}"
                                )

                        # Show list of processed files if available
                        if st.session_state.processed_files:
                            with st.expander("Files in Index"):
                                for file_name in sorted(st.session_state.processed_files):
                                    st.write(f"- {file_name}")

                        # Show more detailed statistics in an expander
                        with st.expander("Index Statistics"):
                            st.json(metrics)
                    else:
                        st.info(f"Index contains {metrics.get('total_chunks', 0)} chunks")
        except Exception as e:
            st.warning(f"Failed to load existing index: {str(e)}")
            logger.warning(f"Failed to load existing index: {str(e)}")

    # Query input and processing
    st.subheader("Ask a Question")
    
    # Show current search method status
    col1, col2 = st.columns([3, 1])
    with col1:
        # Determine and display current search method
        if performance_config.enable_reranking:
            st.info("🔄 **Active Search Method:** Reranking (Enhanced quality)")
        elif retrieval_config.enhanced_mode:
            weight_info = f"(Vector: {retrieval_config.vector_weight:.1f}, Keyword: {1-retrieval_config.vector_weight:.1f})"
            st.info(f"🔍 **Active Search Method:** Hybrid Search {weight_info}")
        else:
            st.info("⚡ **Active Search Method:** Basic Vector Search")
    with col2:
        if st.button("ℹ️ Search Help", help="Learn about search methods"):
            with st.expander("Search Methods Explained", expanded=True):
                st.markdown("""
                **Basic Vector Search**: Fast semantic similarity using embeddings
                
                **Hybrid Search**: Combines vector similarity + keyword matching for broader coverage
                
                **Reranking**: Uses a cross-encoder model to re-score results for highest quality (slower)
                
                💡 Configure these in the sidebar under "Advanced Configuration"
                """)

    # Render configuration sidebar
    debug_mode = render_config_sidebar(args, doc_processor)

    # Create a form for the query
    with st.form(key="query_form"):
        query = st.text_area(
            "Enter your question",
            height=100,
            placeholder="What would you like to know about the documents?",
            key="query_input",
        )

        # Store the previous query in session state to avoid reprocessing on rerun
        if "previous_query" not in st.session_state:
            st.session_state["previous_query"] = ""

        # Add buttons in columns for better layout
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button_pressed = st.form_submit_button("Ask")
        with col2:
            # Add a keyboard shortcut hint
            st.markdown("<small>Press Ctrl+Enter to submit</small>", unsafe_allow_html=True)

    # Check if Enter was pressed in the text area
    enter_pressed = query != st.session_state["previous_query"] and query and not ask_button_pressed

    # Process query if index exists and query is not empty
    if query and doc_processor.has_index() and (ask_button_pressed or enter_pressed):
        # Check if index is loaded
        if doc_processor.index is None or doc_processor.texts is None:
            try:
                doc_processor.load_index(retrieval_config.index_path)
                logger.info("Loaded index for query processing")
            except Exception as e:
                logger.error(f"Error loading index for query: {str(e)}")
                st.error(f"Error loading index: {str(e)}")
                st.stop()

        # Process the query
        process_query(query, doc_processor, llm_service, retrieval_system, debug_mode=debug_mode)
        # Update the previous query after processing
        st.session_state["previous_query"] = query
    elif ask_button_pressed and not query:
        st.warning("Please enter a question")
    elif ask_button_pressed and not doc_processor.has_index():
        st.warning("Please upload documents first to build an index")


if __name__ == "__main__":
    main()
