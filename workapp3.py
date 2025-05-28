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
    parser.add_argument("--dry-run", action="store_true", help="Preview index changes without writing to disk")
    return parser.parse_args()

# Get command line arguments
args = parse_args()

# Configure logger
logger = logging.getLogger(__name__)

# Avoid occasional import bugs
sys.modules['torch.classes'] = None

# Import utility modules
from utils.config_unified import (
    app_config,
    model_config,
    retrieval_config,
    ui_config,
    performance_config,
    config_manager
)
from core.document_processor_unified import DocumentProcessor
from llm.llm_service import LLMService
from retrieval.unified_retrieval_system import UnifiedRetrievalSystem
from utils.index_management.index_manager_unified import index_manager
from utils.text_processing.context_processing import clean_context, extract_hyperlinks
from error_handling.enhanced_decorators import with_advanced_retry, with_error_tracking

# Import enhanced UI components
try:
    from utils.ui import (
        ProgressManager,
        QueryProgressTracker,
        display_enhanced_answer,
        display_search_results,
        display_error_message,
        display_system_status,
        create_collapsible_section
    )
except ImportError:
    # Define fallback UI components if enhanced components are not available
    logger.warning("Enhanced UI components not found, using fallback components")
    
    class ProgressManager:
        def __init__(self, total_steps=1):
            self.total_steps = total_steps
            self.progress_bar = None
            
        def initialize(self, message="Processing..."):
            self.progress_bar = st.progress(0)
            st.text(message)
            
        def update(self, step, message=""):
            if self.progress_bar:
                self.progress_bar.progress(min(1.0, step / self.total_steps))
                if message:
                    st.text(message)
                    
        def complete(self):
            if self.progress_bar:
                self.progress_bar.progress(1.0)
    
class QueryProgressTracker:
        def __init__(self):
            self.progress_bar = None
            self.stages = ["retrieval", "extraction", "formatting", "complete"]
            self.current_stage = None
            self.retrieval_score = 0.0
            self.max_retrieval_score = 0.0
            
        def initialize(self):
            self.progress_bar = st.progress(0)
            self.current_stage = None
            self.retrieval_score = 0.0
            self.max_retrieval_score = 0.0
            
        def update_stage(self, stage, score=None, max_score=None):
            if stage in self.stages:
                self.current_stage = stage
                stage_idx = self.stages.index(stage)
                
                # If we have a retrieval score, use it to adjust the progress bar
                if stage == "retrieval" and score is not None:
                    self.retrieval_score = score
                    if max_score is not None:
                        self.max_retrieval_score = max_score
                    
                    # Scale the progress based on the retrieval score
                    # Higher score = better match = more progress
                    if self.max_retrieval_score > 0:
                        # Normalize the score between 0 and 1
                        normalized_score = min(1.0, self.retrieval_score / self.max_retrieval_score)
                    else:
                        normalized_score = min(1.0, self.retrieval_score)
                    
                    # Use the normalized score to adjust progress within the retrieval stage
                    # This gives visual feedback on the quality of the retrieval
                    progress_value = (stage_idx + normalized_score) / len(self.stages)
                    if self.progress_bar:
                        self.progress_bar.progress(min(0.75, max(0.0, progress_value)))
                else:
                    # Normal stage-based progress
                    if self.progress_bar:
                        self.progress_bar.progress((stage_idx) / (len(self.stages) - 1))
                    
        def complete(self, success=True):
            if self.progress_bar:
                self.progress_bar.progress(1.0)
    
def display_enhanced_answer(answer):
    st.markdown(answer, unsafe_allow_html=True)
    
def display_search_results(results):
    for i, result in enumerate(results):
        st.markdown(f"**Result {i+1}**")
        st.markdown(result.get("text", ""))
        st.markdown("---")
        
def display_error_message(message, suggestions=None):
    st.error(message)
    if suggestions:
        st.markdown("**Suggestions:**")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
            
def display_system_status(status):
    st.info(status)
    
def create_collapsible_section(title, content):
    with st.expander(title):
        st.markdown(content)

# Import original UI components for backward compatibility
try:
    from utils.ui import (
        display_answer,
        display_debug_info,
        display_debug_info_for_index,
        display_confidence_meter,
        smart_wrap
    )
except ImportError:
    # Define fallback UI components if original components are not available
    logger.warning("Original UI components not found, using fallback components")
    
    def display_answer(answer, confidence=None):
        st.markdown(answer)
        
    def display_debug_info(query, context, answer, metadata=None):
        with st.expander("Debug Information"):
            st.subheader("Query")
            st.text(query)
            st.subheader("Context")
            st.text(context)
            st.subheader("Answer")
            st.text(answer)
            if metadata:
                st.subheader("Metadata")
                st.json(metadata)
                
    def display_debug_info_for_index(doc_processor, query):
        st.subheader("Index Debug Information")
        
        # Display index statistics
        try:
            if hasattr(doc_processor, 'get_index_stats'):
                stats = doc_processor.get_index_stats()
                st.write(f"Index contains {stats.get('num_chunks', 0)} chunks")
            elif hasattr(doc_processor, 'index_manager') and hasattr(doc_processor.index_manager, 'get_stats'):
                stats = doc_processor.index_manager.get_stats()
                st.write(f"Index contains {stats.get('num_chunks', 0)} chunks")
            elif hasattr(doc_processor, 'get_metrics'):
                metrics = doc_processor.get_metrics()
                st.write(f"Index contains {len(doc_processor.texts) if hasattr(doc_processor, 'texts') else 0} chunks")
                with st.expander("Index Metrics"):
                    st.json(metrics)
            else:
                st.write("Index statistics not available")
        except Exception as e:
            st.error(f"Error getting index stats: {str(e)}")
        
        # Show top matches for the query
        if query:
            st.write(f"Top matches for query: '{query}'")
            try:
                # Get top matches using the appropriate method
                results = None
                if hasattr(doc_processor, 'get_top_matches'):
                    matches = doc_processor.get_top_matches(query, k=5)
                    # Display each match with score
                    for i, (score, text) in enumerate(matches):
                        with st.expander(f"Match {i+1} (Score: {score:.4f})"):
                            st.text_area(f"Content", text, height=150)
                elif hasattr(doc_processor, 'retrieve'):
                    matches = doc_processor.retrieve(query, k=5)
                    # Display each match with score
                    for i, (score, text) in enumerate(matches):
                        with st.expander(f"Match {i+1} (Score: {score:.4f})"):
                            st.text_area(f"Content", text, height=150)
                elif hasattr(doc_processor, 'search'):
                    results = doc_processor.search(query, top_k=5)
                    # Display each result
                    for i, result in enumerate(results):
                        with st.expander(f"Result {i+1} (Score: {result.get('score', 0):.4f})"):
                            st.text_area(f"Content", result.get("text", ""), height=150)
                else:
                    st.warning("Top matches retrieval not available")
            except Exception as e:
                st.error(f"Error retrieving matches: {str(e)}")
                
    def display_confidence_meter(confidence):
        st.progress(min(1.0, max(0.0, confidence)))
        
    def smart_wrap(text, width=80):
        import textwrap
        return "\n".join(textwrap.wrap(text, width=width))

# ─── Setup logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)

# ─── Initialize services ─────────────────────────────────────────
# Use ui_config from the unified config module
# Note: ui_config is already imported from utils.config_unified above

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
        if doc_processor.has_index() and (doc_processor.index is None or doc_processor.texts is None):
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
    progress_tracker: Optional[QueryProgressTracker] = None
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
            progress_tracker.update_stage("retrieval", score=avg_score, max_score=1.0)  # FAISS scores are typically between 0 and 1
        
        results["retrieval"] = {
            "context": clean_ctx,
            "time": retrieval_time,
            "chunks": num_chunks,
            "scores": retrieval_scores
        }
        
        if not clean_ctx:
            error_msg = "No relevant context found for this query. This could be because:\n"
            error_msg += "1. The similarity threshold may be too high (currently " + str(retrieval_config.similarity_threshold) + ")\n"
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
        if not hasattr(llm_service, 'process_extraction_and_formatting') or not callable(getattr(llm_service, 'process_extraction_and_formatting', None)):
            raise AttributeError("LLM service does not have a properly implemented 'process_extraction_and_formatting' method")
            
        extraction_response, formatting_response = await llm_service.process_extraction_and_formatting(query, clean_ctx)
        
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
        results["extraction"] = {
            "response": extraction_response,
            "raw_answer": answer_raw
        }
        
        # Debug information for extraction
        if debug_mode:
            with st.expander("Raw Answer"):
                st.text_area("Raw", answer_raw, height=150)
            with st.expander("Extraction Metadata"):
                metadata = {}
                # Safely extract metadata fields if they exist
                if "model" in extraction_response:
                    metadata["model"] = extraction_response["model"]
                if "usage" in extraction_response:
                    metadata["usage"] = extraction_response["usage"]
                if "id" in extraction_response:
                    metadata["id"] = extraction_response["id"]
                st.json(metadata)
        
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
                "extraction": results.get("extraction")
            }
        
        # Store formatting results
        answer_formatted = formatting_response["content"]
        results["formatting"] = {
            "response": formatting_response,
            "formatted_answer": answer_formatted
        }
        
        # Debug information for formatting
        if debug_mode:
            with st.expander("Formatted Answer"):
                st.text_area("Formatted", answer_formatted, height=150)
            with st.expander("Formatting Metadata"):
                metadata = {}
                # Safely extract metadata fields if they exist
                if "model" in formatting_response:
                    metadata["model"] = formatting_response["model"]
                if "usage" in formatting_response:
                    metadata["usage"] = formatting_response["usage"]
                if "id" in formatting_response:
                    metadata["id"] = formatting_response["id"]
                st.json(metadata)
        
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
            "extraction": results.get("extraction", {})
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
            process_query_async(query, doc_processor, llm_service, retrieval_system, debug_mode, progress_tracker)
        )
        loop.close()
        
        # Check for errors
        if "error" in results:
            display_error_message(
                results["error"],
                suggestions=[
                    "Try rephrasing your question",
                    "Upload more relevant documents",
                    "Check the 'Debug Index' to see what content is available"
                ]
            )
            return
        
        # Display the formatted answer
        if "formatting" in results and "formatted_answer" in results["formatting"]:
            formatted_answer = results["formatting"]["formatted_answer"]
            display_enhanced_answer(formatted_answer)
            
            # Display retrieval metrics if available
            if "retrieval" in results and "scores" in results["retrieval"] and results["retrieval"]["scores"]:
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
            if st.session_state.get("show_raw_context", False) and "retrieval" in results and "context" in results["retrieval"]:
                with st.expander("Raw Context"):
                    st.text_area("Retrieved Context", results["retrieval"]["context"], height=300)
                    # Display retrieval scores if available
                    if "scores" in results["retrieval"] and results["retrieval"]["scores"]:
                        scores = results["retrieval"]["scores"]
                        avg_score = sum(scores) / len(scores) if scores else 0
                        max_score = max(scores) if scores else 0
                        st.write(f"Retrieval metrics: Avg score: {avg_score:.4f}, Max score: {max_score:.4f}")
            
            # Log successful query
            logger.info(f"Successfully processed query: {query[:50]}...")
            
            # Display debug information if enabled
            if debug_mode and "retrieval" in results and "extraction" in results:
                with st.expander("Debug Information"):
                    st.subheader("Query")
                    st.text(query)
                    
                    st.subheader("Context")
                    if "context" in results["retrieval"]:
                        st.text_area("Retrieved Context", results["retrieval"]["context"], height=200)
                    
                    st.subheader("Raw Answer")
                    if "raw_answer" in results["extraction"]:
                        st.text_area("Extracted Answer", results["extraction"]["raw_answer"], height=150)
                    
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
                    "Check the application logs for more details"
                ]
            )
            
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg, exc_info=True)
        display_error_message(
            error_msg,
            suggestions=[
                "Try again later",
                "Check the application logs for more details"
            ]
        )

# ─── Main application ────────────────────────────────────────────
def main():
    # Set page config
    st.set_page_config(
        page_title=app_config.page_title,
        page_icon=app_config.icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS if available
    if hasattr(ui_config, 'custom_css') and ui_config.custom_css:
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
                "Check the application logs for more details"
            ]
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
                help="Select PDF, TXT, or DOCX files to process"
            )
        with col2:
            if ui_config.show_clear_index_button and st.button("Clear Index", help="Remove all documents from the index"):
                try:
                    doc_processor.clear_index()
                    # Also clear the processed files tracking
                    if 'processed_files' in st.session_state:
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
            if st.button("Reset Tracking", help="Reset the file tracking without clearing the index"):
                if 'processed_files' in st.session_state:
                    st.session_state.processed_files = set()
                st.success("File tracking reset. You can re-upload the same files.")
    
    # Process uploads or load existing index
    if uploads:
        # Store a flag to track if we've already cleared the index in session state
        if 'index_cleared' not in st.session_state:
            st.session_state.index_cleared = False
        
        # Create a progress manager for file processing
        progress_manager = ProgressManager(total_steps=len(uploads) + 2)  # +2 for embedding and indexing
        progress_manager.initialize("Processing files...")
        
        # Process each file
        all_texts = []
        completed_files = []
        
        for i, file in enumerate(uploads):
            try:
                # Update progress
                progress_manager.update(i, f"Processing {file.name}...")
                
                # Get file extension
                file_ext = os.path.splitext(file.name)[1].lower()
                
                # Process file based on type
                if file_ext == ".pdf":
                    # Process PDF file
                    texts = doc_processor.process_file(file)
                    if texts:
                        all_texts.extend(texts)
                        completed_files.append(file)
                    else:
                        st.warning(f"No text extracted from {file.name}. The PDF might be scanned or have security restrictions.")
                elif file_ext == ".txt":
                    # Process text file
                    text = file.read().decode("utf-8")
                    # Create a temporary file with the text content
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                        temp_file.write(text.encode("utf-8"))
                        temp_file_path = temp_file.name
                    
                    try:
                        # Use the existing process_file method with the temporary file
                        texts = doc_processor.process_file(temp_file_path)
                        if texts:
                            all_texts.extend(texts)
                            completed_files.append(file)
                        else:
                            st.warning(f"No text extracted from {file.name}. The file might be empty or corrupted.")
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                elif file_ext in [".docx", ".doc"]:
                    # Process DOCX file
                    # Create a temporary file with the document content
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                        temp_file.write(file.getvalue())
                        temp_file_path = temp_file.name
                    
                    try:
                        # Use the existing process_file method with the temporary file
                        texts = doc_processor.process_file(temp_file_path)
                        if texts:
                            all_texts.extend(texts)
                            completed_files.append(file)
                        else:
                            st.warning(f"No text extracted from {file.name}. The file might be empty or corrupted.")
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                else:
                    st.warning(f"Unsupported file type: {file_ext}. Only PDF, TXT, and DOCX files are supported.")
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                logger.error(f"Error processing {file.name}", exc_info=True)
        
        # Check if we have any texts to process
        if all_texts:
            try:
                # Update progress
                progress_manager.update(len(uploads), "Creating embeddings...")
                
                # Create embeddings
                texts, embeddings = doc_processor.create_embeddings(all_texts)
                
                if texts and len(texts) > 0 and embeddings is not None and embeddings.shape[0] > 0:
                    # Update doc_processor's in-memory state before saving
                    doc_processor.texts = texts
                    doc_processor.chunks = texts  # keep alias in sync
                    # Update progress
                    progress_manager.update(len(uploads) + 1, "Updating index...")
                    
                    # Store the file paths of the current uploads to track what's been processed
                    if 'processed_files' not in st.session_state:
                        st.session_state.processed_files = set()
                    
                    # Get file names from the uploads
                    current_file_names = {file.name for file in uploads}
                    
                    # Check if these files have already been processed in this session
                    already_processed = current_file_names.issubset(st.session_state.processed_files)
                    
                    if already_processed:
                        logger.info("These files have already been processed in this session. Skipping index update.")
                    else:
                        # Update with new documents, appending to existing index
                        success, message = index_manager.update_index(texts, embeddings, append=True, dry_run=args.dry_run)
                        if not success:
                            logger.error(f"Failed to update index: {message}")
                            # Use centralized error logging
                            from utils.error_logging import log_error
                            log_error(f"Failed to update index: {message}", include_traceback=False)
                        else:
                            if args.dry_run:
                                logger.info(f"Index updated in memory only (dry-run mode): {message}")
                            else:
                                logger.info(f"Index updated successfully: {message}")
                            # Add current files to processed set
                            st.session_state.processed_files.update(current_file_names)
                    
                    # Check if we're in dry-run mode
                    if args.dry_run:
                        st.info("⚠️ DRY RUN MODE: Index changes will not be saved to disk")
                        logger.info("Dry run mode: Skipping index save to disk")
                    else:
                        # Create directory if it doesn't exist  
                        from utils.config_unified import resolve_path
                        resolved_index_dir = resolve_path(retrieval_config.index_path, create_dir=True)
                        
                        # Save index to disk
                        index_path = os.path.join(resolved_index_dir, "index.faiss")
                        # Check if index is on GPU before converting
                        if doc_processor.gpu_available and hasattr(doc_processor.index, 'getDevice') and doc_processor.index.getDevice() >= 0:
                            cpu_index = faiss.index_gpu_to_cpu(doc_processor.index)
                        else:
                            cpu_index = doc_processor.index
                        faiss.write_index(cpu_index, index_path)
                        
                        # Save texts
                        texts_path = os.path.join(resolved_index_dir, "texts.npy")
                        np.save(texts_path, np.array(doc_processor.texts, dtype=object), allow_pickle=True)
                        
                        # Save parameters
                        params_path = os.path.join(resolved_index_dir, "metadata.json")
                        params = {
                            "embedding_model": doc_processor.embedding_model_name,
                            "chunk_size": doc_processor.chunk_size,
                            "chunk_overlap": doc_processor.chunk_overlap,
                            "embedding_dim": doc_processor.embedding_dim,
                            "processed_files": list(st.session_state.get('processed_files', set()))
                        }
                        with open(params_path, "w") as f:
                            json.dump(params, f)
                    
                    # Complete progress
                    progress_manager.complete()
                    
                    if ui_config.show_index_statistics:
                        if args.dry_run:
                            st.success(f"Processed {len(uploads)} files successfully! (Changes not saved to disk in dry-run mode)")
                        else:
                            st.success(f"Processed {len(uploads)} files successfully!")
                        # Get metrics safely
                        if hasattr(doc_processor, 'get_metrics'):
                            metrics = doc_processor.get_metrics()
                            st.info(f"Index contains {metrics.get('total_chunks', 0)} chunks from {len(st.session_state.get('processed_files', set()))} files")
                            
                            # Show list of processed files
                            if 'processed_files' in st.session_state and st.session_state.processed_files:
                                with st.expander("Files in Index"):
                                    for file_name in sorted(st.session_state.processed_files):
                                        st.write(f"- {file_name}")
                            
                            # Show more detailed statistics in an expander
                            with st.expander("Index Statistics"):
                                st.json(metrics)
                        else:
                            st.info(f"Index contains {metrics.get('total_chunks', 0)} chunks from {len(st.session_state.get('processed_files', set()))} files")
                else:
                    st.error("Failed to create embeddings. Please check the logs for more information.")
                    logger.error("Failed to create embeddings: empty texts or embeddings")
            except Exception as e:
                st.error(f"Error creating index: {str(e)}")
                logger.error("Error creating index", exc_info=True)
        else:
            st.warning("No text extracted from any of the uploaded files.")
    elif doc_processor.has_index():
        # Load existing index if available
        try:
            if doc_processor.index is None or doc_processor.texts is None:
                doc_processor.load_index(retrieval_config.index_path)
                logger.info("Loaded existing index")
                
                if ui_config.show_index_statistics:
                    if hasattr(doc_processor, 'get_metrics'):
                        metrics = doc_processor.get_metrics()
                        st.info(f"Index contains {metrics.get('total_chunks', 0)} chunks")
                        
                        # Initialize processed_files if not already in session state
                        if 'processed_files' not in st.session_state:
                            st.session_state.processed_files = set()
                            # Try to get processed files from metadata if available
                            try:
                                from utils.config_unified import resolve_path
                                resolved_index_dir = resolve_path(retrieval_config.index_path)
                                metadata_path = os.path.join(resolved_index_dir, "metadata.json")
                                if os.path.exists(metadata_path):
                                    with open(metadata_path, "r") as f:
                                        metadata = json.load(f)
                                        if "processed_files" in metadata:
                                            st.session_state.processed_files = set(metadata["processed_files"])
                            except Exception as e:
                                logger.warning(f"Could not load processed files from metadata: {str(e)}")
                        
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
    
    # Add sidebar with configuration and debug options
    with st.sidebar:
        st.header("Settings")
        
        # Display dry-run mode indicator in sidebar if enabled
        if args.dry_run:
            st.warning("⚠️ DRY RUN MODE")
        
        # Configuration section
        st.subheader("Configuration")
        
        # Basic configuration options
        retrieval_k = st.slider(
            "Number of chunks to retrieve", 
            min_value=1, 
            max_value=100, 
            value=retrieval_config.top_k,
            help="Number of document chunks to retrieve for each query"
        )
        if retrieval_k != retrieval_config.top_k:
            retrieval_config.top_k = retrieval_k
            # Update config file
            config_updates = {"retrieval": {"top_k": retrieval_k}}
            config_manager.update_config(config_updates)
            
        similarity_threshold = st.slider(
            "Similarity threshold", 
            min_value=-1.0, 
            max_value=1.0, 
            value=retrieval_config.similarity_threshold,
            step=0.05,
            help="Minimum similarity score for retrieved chunks (higher = more strict)"
        )
        if similarity_threshold != retrieval_config.similarity_threshold:
            retrieval_config.similarity_threshold = similarity_threshold
            # Update config file
            config_updates = {"retrieval": {"similarity_threshold": similarity_threshold}}
            config_manager.update_config(config_updates)
        
        # Advanced configuration section in an expander
        with st.expander("Advanced Configuration"):
            # Model selection
            extraction_model = st.selectbox(
                "Extraction model",
                options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                index=0 if model_config.extraction_model == "gpt-3.5-turbo" else 
                      1 if model_config.extraction_model == "gpt-4" else 2,
                help="Model used for extracting answers from context"
            )
            if extraction_model != model_config.extraction_model:
                model_config.extraction_model = extraction_model
                # Update config file
                config_updates = {"model": {"extraction_model": extraction_model}}
                config_manager.update_config(config_updates)
            
            formatting_model = st.selectbox(
                "Formatting model",
                options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                index=0 if model_config.formatting_model == "gpt-3.5-turbo" else 
                      1 if model_config.formatting_model == "gpt-4" else 2,
                help="Model used for formatting the final answer"
            )
            if formatting_model != model_config.formatting_model:
                model_config.formatting_model = formatting_model
                # Update config file
                config_updates = {"model": {"formatting_model": formatting_model}}
                config_manager.update_config(config_updates)
            
            # Chunking parameters
            chunk_size = st.number_input(
                "Chunk size",
                min_value=100,
                max_value=2000,
                value=retrieval_config.chunk_size,
                step=100,
                help="Size of document chunks in characters"
            )
            if chunk_size != retrieval_config.chunk_size:
                retrieval_config.chunk_size = chunk_size
                # Update config file
                config_updates = {"retrieval": {"chunk_size": chunk_size}}
                config_manager.update_config(config_updates)
            
            chunk_overlap = st.number_input(
                "Chunk overlap",
                min_value=0,
                max_value=500,
                value=retrieval_config.chunk_overlap,
                step=50,
                help="Overlap between document chunks in characters"
            )
            if chunk_overlap != retrieval_config.chunk_overlap:
                retrieval_config.chunk_overlap = chunk_overlap
                # Update config file
                config_updates = {"retrieval": {"chunk_overlap": chunk_overlap}}
                config_manager.update_config(config_updates)
            
            # Performance settings
            st.subheader("Performance Settings")
            
            use_gpu = st.checkbox(
                "Use GPU for embeddings",
                value=performance_config.use_gpu_for_faiss,
                help="Use GPU for FAISS index operations (if available)"
            )
            if use_gpu != performance_config.use_gpu_for_faiss:
                performance_config.use_gpu_for_faiss = use_gpu
                # Update performance config file
                config_updates = {"performance": {"use_gpu_for_faiss": use_gpu}}
                config_manager.update_config(config_updates)
            
            enable_reranking = st.checkbox(
                "Enable reranking",
                value=performance_config.enable_reranking,
                help="Use a reranker model to improve retrieval quality (slower)"
            )
            if enable_reranking != performance_config.enable_reranking:
                performance_config.enable_reranking = enable_reranking
                # Update performance config file
                config_updates = {"performance": {"enable_reranking": enable_reranking}}
                config_manager.update_config(config_updates)
            
            # Enhanced retrieval settings
            st.subheader("Enhanced Retrieval")
            
            enhanced_mode = st.checkbox(
                "Enable hybrid search",
                value=retrieval_config.enhanced_mode,
                help="Combine vector search with keyword search for better results"
            )
            if enhanced_mode != retrieval_config.enhanced_mode:
                retrieval_config.enhanced_mode = enhanced_mode
                # Update config file
                config_updates = {"retrieval": {"enhanced_mode": enhanced_mode}}
                config_manager.update_config(config_updates)
            
            # Only show vector weight slider if enhanced mode is enabled
            if enhanced_mode:
                vector_weight = st.slider(
                    "Vector search weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=retrieval_config.vector_weight,
                    step=0.1,
                    help="Weight for vector search in hybrid mode (0.0 = keyword only, 1.0 = vector only)"
                )
                if vector_weight != retrieval_config.vector_weight:
                    retrieval_config.vector_weight = vector_weight
                    # Update config file
                    config_updates = {"retrieval": {"vector_weight": vector_weight}}
                    config_manager.update_config(config_updates)
            
            # Add a toggle for raw context display
            st.subheader("Display Options")
            
            show_raw_context = st.checkbox(
                "Show raw context",
                value=st.session_state.get("show_raw_context", False),
                help="Display the raw context used to generate the answer"
            )
            if show_raw_context:
                st.info("Raw context will be displayed in a collapsible section below each answer, including retrieval scores.")
            
            if "show_raw_context" not in st.session_state or show_raw_context != st.session_state.get("show_raw_context", False):
                st.session_state["show_raw_context"] = show_raw_context
        
        # Debug mode toggle
        st.subheader("Debug Options")
        debug_mode = st.checkbox("Debug Mode", value=False, help="Show detailed debug information")
        
        # Add a debug index button if in debug mode
        if debug_mode:
            st.subheader("Debug Tools")
            debug_query = st.text_input("Debug Query", placeholder="Enter a query to debug the index")
            if st.button("Debug Index"):
                if debug_query:
                    display_debug_info_for_index(doc_processor, debug_query)
                else:
                    st.warning("Please enter a query to debug the index")
    
    # Create a form for the query
    with st.form(key="query_form"):
        query = st.text_area(
            "Enter your question",
            height=100,
            placeholder="What would you like to know about the documents?",
            key="query_input"
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
    if (query and doc_processor.has_index() and (ask_button_pressed or enter_pressed)):
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
