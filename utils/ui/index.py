# Consolidated UI components for Streamlit display
# This file is deprecated and will be removed in a future version
# Use direct imports from utils.ui instead

from utils.ui.components import (
    display_answer,
    display_debug_info,
    display_debug_info_for_index,
    display_confidence_meter,
    display_warnings,
    display_uncertain_sections,
    display_term_distinctions
)

from utils.ui.text_processing import (
    smart_wrap,
    extract_confidence_score,
    get_confidence_description,
    safe_highlight_for_streamlit,
    get_confidence_color
)

from utils.ui.styles import (
    WRAPPER_DIV_STYLE,
    UNCERTAIN_SECTION_STYLE,
    TERM_DISTINCTION_STYLE,
    get_html_wrapper
)

__all__ = [
    # Components
    'display_answer',
    'display_debug_info',
    'display_debug_info_for_index',
    'display_confidence_meter',
    'display_warnings',
    'display_uncertain_sections',
    'display_term_distinctions',
    
    # Text processing
    'smart_wrap',
    'extract_confidence_score',
    'get_confidence_description',
    'safe_highlight_for_streamlit',
    'get_confidence_color',
    
    # Styles
    'WRAPPER_DIV_STYLE',
    'UNCERTAIN_SECTION_STYLE',
    'TERM_DISTINCTION_STYLE',
    'get_html_wrapper'
]