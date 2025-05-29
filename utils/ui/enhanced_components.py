"""Enhanced UI components for the Streamlit app."""

import streamlit as st
import logging
import time
from typing import List, Dict, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)


class ProgressManager:
    """Manage progress bars in the Streamlit UI"""

    def __init__(self, total_steps=1):
        """Initialize the progress manager"""
        self.total_steps = total_steps
        self.progress_bar = None

    def initialize(self, message="Processing..."):
        """Initialize the progress bar"""
        self.progress_bar = st.progress(0)
        st.text(message)

    def update(self, step, message=""):
        """Update the progress bar"""
        if self.progress_bar:
            self.progress_bar.progress(min(1.0, step / self.total_steps))
            if message:
                st.text(message)

    def complete(self):
        """Complete the progress bar"""
        if self.progress_bar:
            self.progress_bar.progress(1.0)


class QueryProgressTracker:
    """Track progress of query processing"""

    def __init__(self):
        """Initialize the query progress tracker"""
        self.progress_bar = None
        self.stages = ["retrieval", "extraction", "formatting", "complete"]
        self.current_stage = None
        self.retrieval_score = 0.0
        self.max_retrieval_score = 0.0

    def initialize(self):
        """Initialize the progress bar"""
        self.progress_bar = st.progress(0)
        self.current_stage = None
        self.retrieval_score = 0.0
        self.max_retrieval_score = 0.0

    def update_stage(self, stage, score=None, max_score=None):
        """Update the current stage"""
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
        """Complete the progress bar"""
        if self.progress_bar:
            self.progress_bar.progress(1.0)


def display_enhanced_answer(answer):
    """
    Display an enhanced answer with markdown formatting and quality checks

    Args:
        answer: The formatted answer to display
    """
    # Display the answer as markdown
    st.markdown(answer, unsafe_allow_html=True)

    # Log the final formatted answer
    logger.info(f"Final formatted answer displayed to user (first 100 chars): {answer[:100]}...")

    # Basic quality checks for the answer
    if len(answer.strip()) < 10:
        warning_msg = "Warning: Answer appears to be too short or empty"
        logger.warning(warning_msg)
        st.warning(warning_msg)
    elif "Answer not found" in answer:
        logger.info("Answer indicates information was not found in the documents")
        st.info("💡 The requested information was not found in the uploaded documents. Consider uploading additional relevant documents.")


def display_search_results(results):
    """Display search results"""
    for i, result in enumerate(results):
        st.markdown(f"**Result {i+1}**")
        st.markdown(result.get("text", ""))
        st.markdown("---")


def display_error_message(message, suggestions=None):
    """Display an error message with suggestions"""
    st.error(message)
    if suggestions:
        st.markdown("**Suggestions:**")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")


def display_system_status(status):
    """Display system status"""
    st.info(status)


def create_collapsible_section(title, content):
    """Create a collapsible section"""
    with st.expander(title):
        st.markdown(content)
