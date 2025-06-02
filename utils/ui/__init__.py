# UI components package

# Import components to make them available when importing from the package
from utils.ui.components import (
    display_answer,
    display_debug_info,
    display_debug_info_for_index,
    display_confidence_meter,
    display_warnings,
    display_uncertain_sections,
    display_term_distinctions,
)

from utils.ui.enhanced_components import (
    ProgressManager,
    QueryProgressTracker,
    display_enhanced_answer,
    display_search_results,
    display_error_message,
    display_system_status,
    create_collapsible_section,
)

from utils.ui.feedback_components import (
    render_feedback_widget,
    render_feedback_analytics_widget,
    render_feedback_summary_card,
    create_feedback_callback,
    clear_feedback_state,
)

from utils.ui.text_processing import (
    smart_wrap,
    extract_confidence_score,
    get_confidence_description,
    safe_highlight_for_streamlit,
    get_confidence_color,
)

from utils.ui.styles import (
    WRAPPER_DIV_STYLE,
    UNCERTAIN_SECTION_STYLE,
    TERM_DISTINCTION_STYLE,
    get_html_wrapper,
)

# Export all components
__all__ = [
    # Original components
    "display_answer",
    "display_debug_info",
    "display_debug_info_for_index",
    "display_confidence_meter",
    "display_warnings",
    "display_uncertain_sections",
    "display_term_distinctions",
    # Enhanced components
    "ProgressManager",
    "QueryProgressTracker",
    "display_enhanced_answer",
    "display_search_results",
    "display_error_message",
    "display_system_status",
    "create_collapsible_section",
    # Feedback components
    "render_feedback_widget",
    "render_feedback_analytics_widget",
    "render_feedback_summary_card",
    "create_feedback_callback",
    "clear_feedback_state",
    # Text processing
    "smart_wrap",
    "extract_confidence_score",
    "get_confidence_description",
    "safe_highlight_for_streamlit",
    "get_confidence_color",
    # Styles
    "WRAPPER_DIV_STYLE",
    "UNCERTAIN_SECTION_STYLE",
    "TERM_DISTINCTION_STYLE",
    "get_html_wrapper",
]
