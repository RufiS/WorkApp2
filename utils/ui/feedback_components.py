"""Feedback UI components for WorkApp2.

Components for collecting and displaying user feedback on query responses.
"""

import streamlit as st
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import logging

from core.models.feedback_models import FeedbackRequest

logger = logging.getLogger(__name__)


def render_feedback_widget(
    question: str,
    context: str,
    answer: str,
    query_results: Dict[str, Any],
    on_feedback_submit: Callable[[FeedbackRequest], str],
    widget_key: Optional[str] = None,
    production_mode: bool = False
) -> None:
    """Render the feedback widget for user responses.
    
    Args:
        question: Original user question
        context: Retrieved context used for answer
        answer: Generated answer
        query_results: Complete query processing results
        on_feedback_submit: Callback function to handle feedback submission
        widget_key: Unique key for the widget
        production_mode: Whether app is in production mode
    """
    # Create unique key if not provided
    if widget_key is None:
        widget_key = f"feedback_{hash(question + answer)}"
    
    # Don't show if feedback was already submitted for this response
    feedback_submitted_key = f"{widget_key}_submitted"
    if st.session_state.get(feedback_submitted_key, False):
        _display_feedback_thanks()
        return
    
    # Container for feedback widget
    with st.container():
        st.markdown("---")
        
        # Main feedback section
        feedback_col, spacer_col = st.columns([4, 1])
        
        with feedback_col:
            st.markdown("**Was this answer helpful?**")
            
            # Create columns for thumbs buttons
            thumb_col1, thumb_col2, comment_col = st.columns([1, 1, 3])
            
            with thumb_col1:
                if st.button("👍 Yes", key=f"{widget_key}_positive", help="This answer was helpful"):
                    _handle_positive_feedback(
                        question, context, answer, query_results, 
                        on_feedback_submit, feedback_submitted_key
                    )
                    # Use lighter rerun to preserve answer
                    st.rerun()
            
            with thumb_col2:
                if st.button("👎 No", key=f"{widget_key}_negative", help="This answer was not helpful"):
                    # Store negative feedback trigger in session state
                    st.session_state[f"{widget_key}_show_comment"] = True
                    # Use lighter rerun to preserve answer
                    st.rerun()
            
            # Show comment box if negative feedback was clicked
            if st.session_state.get(f"{widget_key}_show_comment", False):
                _render_negative_feedback_form(
                    question, context, answer, query_results,
                    on_feedback_submit, widget_key, feedback_submitted_key
                )


def _handle_positive_feedback(
    question: str,
    context: str,
    answer: str,
    query_results: Dict[str, Any],
    on_feedback_submit: Callable[[FeedbackRequest], str],
    feedback_submitted_key: str
) -> None:
    """Handle positive feedback submission.
    
    Args:
        question: Original user question
        context: Retrieved context used for answer
        answer: Generated answer
        query_results: Complete query processing results
        on_feedback_submit: Callback function to handle feedback submission
        feedback_submitted_key: Session state key to track submission
    """
    try:
        feedback_request = FeedbackRequest(
            feedback_type="positive",
            feedback_text=None
        )
        
        logger.info(f"Submitting positive feedback for question: {question[:50]}...")
        feedback_id = on_feedback_submit(feedback_request)
        
        # Mark as submitted
        st.session_state[feedback_submitted_key] = True
        st.session_state[f"{feedback_submitted_key}_type"] = "positive"
        st.session_state[f"{feedback_submitted_key}_id"] = feedback_id
        
        logger.info(f"Positive feedback submitted successfully: {feedback_id}")
        st.success("Thank you for your positive feedback!")
        
    except Exception as e:
        logger.error(f"Failed to submit positive feedback: {str(e)}", exc_info=True)
        st.error(f"Failed to submit feedback: {str(e)}")


def _render_negative_feedback_form(
    question: str,
    context: str,
    answer: str,
    query_results: Dict[str, Any],
    on_feedback_submit: Callable[[FeedbackRequest], str],
    widget_key: str,
    feedback_submitted_key: str
) -> None:
    """Render the negative feedback form with comment box.
    
    Args:
        question: Original user question
        context: Retrieved context used for answer
        answer: Generated answer
        query_results: Complete query processing results
        on_feedback_submit: Callback function to handle feedback submission
        widget_key: Widget key for unique identification
        feedback_submitted_key: Session state key to track submission
    """
    st.markdown("**Please tell us what went wrong (optional):**")
    
    # Comment text area
    comment_text = st.text_area(
        "Your feedback helps us improve",
        placeholder="e.g., The answer was incomplete, incorrect, or didn't address my question...",
        key=f"{widget_key}_comment_text",
        height=80
    )
    
    # Submit buttons
    submit_col1, submit_col2, cancel_col = st.columns([2, 2, 1])
    
    with submit_col1:
        if st.button("Submit Feedback", key=f"{widget_key}_submit_negative"):
            _handle_negative_feedback(
                question, context, answer, query_results,
                on_feedback_submit, feedback_submitted_key,
                comment_text.strip() if comment_text else None
            )
            st.rerun()
    
    with submit_col2:
        if st.button("Submit without comment", key=f"{widget_key}_submit_negative_no_comment"):
            _handle_negative_feedback(
                question, context, answer, query_results,
                on_feedback_submit, feedback_submitted_key,
                None
            )
            st.rerun()
    
    with cancel_col:
        if st.button("Cancel", key=f"{widget_key}_cancel"):
            # Clear the comment form
            st.session_state[f"{widget_key}_show_comment"] = False
            st.rerun()


def _handle_negative_feedback(
    question: str,
    context: str,
    answer: str,
    query_results: Dict[str, Any],
    on_feedback_submit: Callable[[FeedbackRequest], str],
    feedback_submitted_key: str,
    comment_text: Optional[str]
) -> None:
    """Handle negative feedback submission.
    
    Args:
        question: Original user question
        context: Retrieved context used for answer
        answer: Generated answer
        query_results: Complete query processing results
        on_feedback_submit: Callback function to handle feedback submission
        feedback_submitted_key: Session state key to track submission
        comment_text: Optional user comment
    """
    try:
        feedback_request = FeedbackRequest(
            feedback_type="negative",
            feedback_text=comment_text
        )
        
        logger.info(f"Submitting negative feedback for question: {question[:50]}... (comment: {'Yes' if comment_text else 'No'})")
        feedback_id = on_feedback_submit(feedback_request)
        
        # Mark as submitted
        st.session_state[feedback_submitted_key] = True
        st.session_state[f"{feedback_submitted_key}_type"] = "negative"
        st.session_state[f"{feedback_submitted_key}_id"] = feedback_id
        
        logger.info(f"Negative feedback submitted successfully: {feedback_id}")
        st.success("Thank you for your feedback! This helps us improve.")
        
    except Exception as e:
        logger.error(f"Failed to submit negative feedback: {str(e)}", exc_info=True)
        st.error(f"Failed to submit feedback: {str(e)}")


def _display_feedback_thanks() -> None:
    """Display thank you message after feedback submission."""
    with st.container():
        st.markdown("---")
        st.success("✅ Thank you for your feedback!")


def render_feedback_analytics_widget(
    feedback_analytics: Dict[str, Any],
    production_mode: bool = False
) -> None:
    """Render feedback analytics widget for development mode.
    
    Args:
        feedback_analytics: Analytics data from feedback service
        production_mode: Whether app is in production mode
    """
    # Only show in development mode
    if production_mode:
        return
    
    if not feedback_analytics or feedback_analytics.get("total_feedback", 0) == 0:
        return
    
    with st.expander("📊 Recent Feedback Analytics", expanded=False):
        # Overall stats
        total = feedback_analytics.get("total_feedback", 0)
        positive = feedback_analytics.get("positive_count", 0)
        negative = feedback_analytics.get("negative_count", 0)
        positive_rate = feedback_analytics.get("positive_rate", 0)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Feedback", total)
        
        with col2:
            st.metric("Positive", positive, delta=f"{positive_rate:.1%}")
        
        with col3:
            st.metric("Negative", negative)
        
        with col4:
            avg_score = feedback_analytics.get("avg_retrieval_score", 0)
            st.metric("Avg Score", f"{avg_score:.3f}")
        
        # Performance comparison
        if "positive_avg_score" in feedback_analytics and "negative_avg_score" in feedback_analytics:
            st.markdown("**Retrieval Score Comparison:**")
            pos_score = feedback_analytics["positive_avg_score"]
            neg_score = feedback_analytics["negative_avg_score"]
            
            score_col1, score_col2 = st.columns(2)
            with score_col1:
                st.metric("Positive Feedback Avg Score", f"{pos_score:.3f}")
            with score_col2:
                st.metric("Negative Feedback Avg Score", f"{neg_score:.3f}")
        
        # Engine breakdown
        if "engine_breakdown" in feedback_analytics:
            st.markdown("**By Search Engine:**")
            engine_data = feedback_analytics["engine_breakdown"]
            
            for engine, stats in engine_data.items():
                if stats["total"] > 0:
                    engine_rate = stats.get("positive_rate", 0)
                    st.write(f"• {engine}: {stats['positive']}/{stats['total']} positive ({engine_rate:.1%})")


def render_feedback_summary_card(
    recent_feedback_summary: Dict[str, Any],
    production_mode: bool = False
) -> None:
    """Render a compact feedback summary card.
    
    Args:
        recent_feedback_summary: Recent feedback summary data
        production_mode: Whether app is in production mode
    """
    # Only show in development mode
    if production_mode:
        return
    
    if not recent_feedback_summary or recent_feedback_summary.get("total", 0) == 0:
        return
    
    total = recent_feedback_summary.get("total", 0)
    positive_rate = recent_feedback_summary.get("positive_rate", 0)
    
    # Show as a small info box
    if total > 0:
        st.info(f"📝 Recent feedback: {total} responses, {positive_rate:.1%} positive")


def create_feedback_callback(
    feedback_service,
    question: str,
    context: str,
    answer: str,
    query_results: Dict[str, Any],
    config_snapshot: Dict[str, Any],
    session_context: Dict[str, Any],
    model_config: Dict[str, Any]
) -> Callable[[FeedbackRequest], str]:
    """Create a feedback callback function for the widget.
    
    Args:
        feedback_service: FeedbackService instance
        question: Original user question
        context: Retrieved context used for answer
        answer: Generated answer
        query_results: Complete query processing results
        config_snapshot: Current configuration state
        session_context: Session information
        model_config: Model configuration
        
    Returns:
        Callback function that takes FeedbackRequest and returns feedback_id
    """
    def callback(feedback_request: FeedbackRequest) -> str:
        """Handle feedback submission."""
        return feedback_service.store_feedback(
            question=question,
            context=context,
            answer=answer,
            feedback_request=feedback_request,
            query_results=query_results,
            config_snapshot=config_snapshot,
            session_context=session_context,
            model_config=model_config
        )
    
    return callback


def clear_feedback_state(widget_key: str) -> None:
    """Clear feedback-related session state for a widget.
    
    Args:
        widget_key: Widget key to clear state for
    """
    keys_to_clear = [
        f"{widget_key}_submitted",
        f"{widget_key}_type",
        f"{widget_key}_id",
        f"{widget_key}_show_comment",
        f"{widget_key}_comment_text"
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
