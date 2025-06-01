"""UI Controller for WorkApp2.

Handles the main UI rendering, layout, and user interaction coordination.
Extracted from the monolithic workapp3.py for better maintainability.
"""

import streamlit as st
import logging
from typing import Optional, Dict, Any

# Type hints for internal modules are allowed to use # type: ignore with reason
from core.config import app_config, ui_config  # type: ignore[import] # TODO: Add proper config types


logger = logging.getLogger(__name__)


class UIController:
    """Controller responsible for UI rendering and layout management."""

    def __init__(self, app_orchestrator: Optional[Any] = None) -> None:
        """Initialize the UI controller.

        Args:
            app_orchestrator: The main application orchestrator for service coordination
        """
        self.app_orchestrator = app_orchestrator
        self.logger = logger

    def configure_page(self) -> None:
        """Configure the Streamlit page settings and layout."""
        st.set_page_config(
            page_title=app_config.page_title,
            page_icon=app_config.icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Apply custom CSS if available
        if hasattr(ui_config, "custom_css") and ui_config.custom_css:
            st.markdown(ui_config.custom_css, unsafe_allow_html=True)

        self.logger.info("Page configuration applied")

    def render_header(self) -> None:
        """Render the application header and title section."""
        st.title(app_config.page_title)
        if app_config.subtitle:
            st.subheader(app_config.subtitle)

        # Add version info in footer
        with st.container():
            st.caption(f"Application version: {getattr(app_config, 'version', '0.4.0')}")

        self.logger.debug("Application header rendered")

    def render_dry_run_indicator(self, dry_run_mode: bool) -> None:
        """Render dry-run mode indicator if enabled.

        Args:
            dry_run_mode: Whether the application is in dry-run mode
        """
        if dry_run_mode:
            st.warning("⚠️ DRY RUN MODE: Index changes will not be saved to disk")
            self.logger.info("Dry-run mode indicator displayed")

    def render_search_method_status(self, performance_config: Any, retrieval_config: Any) -> None:
        """Render the current search method status display.

        Args:
            performance_config: Performance configuration object
            retrieval_config: Retrieval configuration object
        """
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

        self.logger.debug("Search method status rendered")

    def render_query_form(self) -> tuple[str, bool]:
        """Render the query input form and return the query and submit status.

        Returns:
            Tuple of (query_text, was_submitted)
        """
        st.subheader("Ask a Question")

        # Create a form for the query
        with st.form(key="query_form"):
            query = st.text_input(
                "Enter your question",
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
                st.markdown("<small>Press Enter to submit</small>", unsafe_allow_html=True)

        self.logger.debug(f"Query form rendered, submitted: {ask_button_pressed}")
        return query, ask_button_pressed

    def display_warning(self, message: str) -> None:
        """Display a warning message to the user.

        Args:
            message: The warning message to display
        """
        st.warning(message)
        self.logger.warning(f"Warning displayed: {message}")

    def display_error(self, message: str, suggestions: Optional[list[str]] = None) -> None:
        """Display an error message with optional suggestions.

        Args:
            message: The error message to display
            suggestions: Optional list of suggestions for the user
        """
        # Import UI utility for error display
        try:
            from utils.ui import display_error_message  # type: ignore[import] # TODO: Add proper UI types

            if suggestions:
                display_error_message(message, suggestions=suggestions)
            else:
                display_error_message(message)
        except ImportError:
            # Fallback to basic Streamlit error display
            st.error(message)
            if suggestions:
                st.write("**Suggestions:**")
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")

        self.logger.error(f"Error displayed: {message}")

    def display_success(self, message: str) -> None:
        """Display a success message to the user.

        Args:
            message: The success message to display
        """
        st.success(message)
        self.logger.info(f"Success message displayed: {message}")

    def display_info(self, message: str) -> None:
        """Display an info message to the user.

        Args:
            message: The info message to display
        """
        st.info(message)
        self.logger.debug(f"Info message displayed: {message}")

    def stop_execution(self) -> None:
        """Stop the Streamlit execution (typically after critical errors)."""
        st.stop()
        self.logger.critical("Streamlit execution stopped")
