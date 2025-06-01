"""Document Controller for WorkApp2.

Handles document upload, processing, and index management.
Extracted from the monolithic workapp3.py for better maintainability.
"""

import streamlit as st
import logging
import json
import os
from typing import Optional, List, Dict, Any, Tuple

# Type hints for internal modules are allowed to use # type: ignore with reason
from core.document_processor import DocumentProcessor  # type: ignore[import] # TODO: Add proper types
from core.index_management.index_coordinator import index_coordinator  # type: ignore[import] # TODO: Add proper types
from core.config import retrieval_config  # type: ignore[import] # TODO: Add proper config types
from utils.ui import ProgressManager  # type: ignore[import] # TODO: Add proper UI types


logger = logging.getLogger(__name__)


class DocumentController:
    """Controller responsible for document upload and processing operations."""

    def __init__(self, app_orchestrator: Optional[Any] = None) -> None:
        """Initialize the document controller.

        Args:
            app_orchestrator: The main application orchestrator for service coordination
        """
        self.app_orchestrator = app_orchestrator
        self.logger = logger

    def render_upload_section(self, ui_config: Any, dry_run_mode: bool = False) -> Optional[List[Any]]:
        """Render the document upload section of the UI.

        Args:
            ui_config: UI configuration object
            dry_run_mode: Whether the application is in dry-run mode

        Returns:
            List of uploaded files or None if no uploads
        """
        uploads = None

        if ui_config.show_document_upload:
            st.subheader("Document Upload")
            st.write("Upload PDF, TXT, or DOCX files to build a searchable index.")

            # Add buttons for upload controls
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
                    self._handle_clear_index(dry_run_mode)

            with col3:
                if st.button(
                    "Reset Tracking", help="Reset the file tracking without clearing the index"
                ):
                    self._handle_reset_tracking()

        return uploads

    def _handle_clear_index(self, dry_run_mode: bool) -> None:
        """Handle clearing the document index.

        Args:
            dry_run_mode: Whether the application is in dry-run mode
        """
        try:
            if self.app_orchestrator:
                doc_processor = self.app_orchestrator.get_document_processor()
                doc_processor.clear_index()

            # Also clear the processed files tracking
            if "processed_files" in st.session_state:
                st.session_state.processed_files = set()

            # Check if we're in dry-run mode
            if dry_run_mode:
                st.success("Index cleared in memory only (dry-run mode)")
                self.logger.info("Dry run mode: Index cleared in memory only")
            else:
                st.success("Index cleared successfully!")
                self.logger.info("Index cleared successfully")
        except Exception as e:
            st.error(f"Error clearing index: {str(e)}")
            self.logger.error(f"Error clearing index: {str(e)}")

    def _handle_reset_tracking(self) -> None:
        """Handle resetting the file tracking."""
        if "processed_files" in st.session_state:
            st.session_state.processed_files = set()
        st.success("File tracking reset. You can re-upload the same files.")
        self.logger.info("File tracking reset")

    def process_uploaded_files(
        self,
        uploads: List[Any],
        dry_run_mode: bool = False
    ) -> Tuple[bool, str]:
        """Process uploaded files and update the document index.

        Args:
            uploads: List of uploaded file objects from Streamlit
            dry_run_mode: Whether the application is in dry-run mode

        Returns:
            Tuple of (success, message)
        """
        if not uploads:
            return False, "No files uploaded"

        try:
            # Get document processor
            if not self.app_orchestrator:
                return False, "Application orchestrator not available"

            doc_processor = self.app_orchestrator.get_document_processor()

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
                self.logger.info("Files already processed, skipping upload processing.")
                progress_manager.complete()
                return True, "Files already processed"

            # Process each file
            for i, file in enumerate(uploads):
                try:
                    progress_manager.update(i, f"Processing {file.name}...")

                    # Use existing DocumentProcessor.process_file() method
                    chunks = doc_processor.process_file(file)

                    if chunks:
                        all_chunks.extend(chunks)
                        completed_files.append(file)
                        self.logger.info(f"Successfully processed {file.name}: {len(chunks)} chunks")
                    else:
                        st.warning(f"No content extracted from {file.name}. File may be empty or unsupported.")

                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    self.logger.error(f"Error processing {file.name}: {str(e)}", exc_info=True)

            # Update index if we have chunks
            if all_chunks:
                try:
                    progress_manager.update(len(uploads), "Updating index...")

                    # Extract texts from chunks for index coordinator
                    texts = [
                        chunk.get('text', chunk.get('content', ''))
                        for chunk in all_chunks
                        if chunk.get('text') or chunk.get('content')
                    ]

                    if texts:
                        # Use existing index coordinator for index updates
                        embeddings = doc_processor.batch_embed_chunks(texts)
                        success, message = index_coordinator.update_index(
                            texts, embeddings, append=True, dry_run=dry_run_mode
                        )

                        if success:
                            # CRITICAL FIX: Reload DocumentProcessor index after coordinator update
                            if not dry_run_mode:
                                try:
                                    doc_processor.load_index(retrieval_config.index_path)
                                    self.logger.info("DocumentProcessor reloaded updated index")
                                except Exception as e:
                                    self.logger.warning(f"Failed to reload DocumentProcessor index: {str(e)}")

                            # Update processed files tracking
                            st.session_state.processed_files.update(current_file_names)

                            if dry_run_mode:
                                success_msg = f"Processed {len(completed_files)} files successfully! (Dry-run mode - changes not saved)"
                                st.success(success_msg)
                                self.logger.info(f"Index updated in memory only (dry-run mode): {message}")
                            else:
                                success_msg = f"Processed {len(completed_files)} files successfully!"
                                st.success(success_msg)
                                self.logger.info(f"Index updated successfully: {message}")

                            progress_manager.complete()
                            return True, success_msg
                        else:
                            error_msg = f"Failed to update index: {message}"
                            st.error(error_msg)
                            self.logger.error(error_msg)
                            progress_manager.complete()
                            return False, error_msg
                    else:
                        error_msg = "No valid text content found in uploaded files."
                        st.error(error_msg)
                        progress_manager.complete()
                        return False, error_msg

                except Exception as e:
                    error_msg = f"Error updating index: {str(e)}"
                    st.error(error_msg)
                    self.logger.error(error_msg, exc_info=True)
                    progress_manager.complete()
                    return False, error_msg
            else:
                warning_msg = "No content extracted from any uploaded files."
                st.warning(warning_msg)
                progress_manager.complete()
                return False, warning_msg

        except Exception as e:
            error_msg = f"Error processing uploaded files: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False, error_msg

    def display_index_statistics(self, ui_config: Any) -> None:
        """Display index statistics and file information.

        Args:
            ui_config: UI configuration object
        """
        try:
            if not self.app_orchestrator:
                return

            doc_processor = self.app_orchestrator.get_document_processor()

            # Load existing index if available and not already loaded
            if doc_processor.has_index():
                try:
                    if doc_processor.index is None or doc_processor.texts is None:
                        doc_processor.load_index(retrieval_config.index_path)
                        self.logger.info("Loaded existing index for statistics display")

                    if ui_config.show_index_statistics:
                        if hasattr(doc_processor, "get_metrics"):
                            metrics = doc_processor.get_metrics()
                            st.info(f"Index contains {metrics.get('total_chunks', 0)} chunks")

                            # Initialize processed_files if not already in session state
                            if "processed_files" not in st.session_state:
                                st.session_state.processed_files = set()
                                # Try to get processed files from metadata if available
                                try:
                                    from core.config import resolve_path  # type: ignore[import] # TODO: Add proper config types

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
                                    self.logger.warning(
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
                            if hasattr(doc_processor, "get_metrics"):
                                metrics = doc_processor.get_metrics()
                                st.info(f"Index contains {metrics.get('total_chunks', 0)} chunks")
                except Exception as e:
                    st.warning(f"Failed to load existing index: {str(e)}")
                    self.logger.warning(f"Failed to load existing index: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error displaying index statistics: {str(e)}")

    def get_processed_files_count(self) -> int:
        """Get the number of processed files.

        Returns:
            Number of processed files
        """
        if "processed_files" in st.session_state:
            return len(st.session_state.processed_files)
        return 0

    def is_file_processed(self, filename: str) -> bool:
        """Check if a file has been processed.

        Args:
            filename: Name of the file to check

        Returns:
            True if the file has been processed, False otherwise
        """
        if "processed_files" in st.session_state:
            return filename in st.session_state.processed_files
        return False
