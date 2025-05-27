# UI components for Streamlit display
import streamlit as st
from typing import Dict, Any, Optional

from utils.ui.text_processing import (
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

def display_warnings(warnings: list) -> None:
    """Display warning messages with appropriate styling
    
    Args:
        warnings: List of warning dictionaries with type and message
    """
    for warning in warnings:
        if warning["type"] == "uncertain":
            st.warning(warning["message"])
        elif warning["type"] == "distinction":
            st.info(warning["message"])
        elif warning["type"] == "confusion":
            st.error(warning["message"])

def display_uncertain_sections(sections: list) -> None:
    """Display uncertain sections with highlighting
    
    Args:
        sections: List of dictionaries containing uncertain text sections
    """
    if sections:
        st.markdown("### Uncertain Sections:")
        for section in sections:
            content = f"<strong>Uncertain:</strong> {section['text']}"
            st.markdown(
                get_html_wrapper(content, UNCERTAIN_SECTION_STYLE),
                unsafe_allow_html=True
            )

def display_term_distinctions(distinctions: list) -> None:
    """Display term distinctions with highlighting
    
    Args:
        distinctions: List of dictionaries containing term distinctions
    """
    if distinctions:
        st.markdown("### Important Term Distinctions:")
        for term in distinctions:
            content = f"<strong>Term Distinction:</strong> {term['text']}"
            st.markdown(
                get_html_wrapper(content, TERM_DISTINCTION_STYLE),
                unsafe_allow_html=True
            )

def display_answer(answer: str) -> Dict[str, Any]:
    """Display the answer with highlighting
    
    Args:
        answer: The formatted answer text
        
    Returns:
        Dictionary with highlighting data
    """
    # Process text for highlighting and warnings
    highlight_data = safe_highlight_for_streamlit(answer)
    
    # Display warnings
    display_warnings(highlight_data["warnings"])
    
    # Display main answer text
    st.markdown(
        get_html_wrapper(highlight_data["text"], WRAPPER_DIV_STYLE),
        unsafe_allow_html=True
    )
    
    # Display uncertain sections and term distinctions
    display_uncertain_sections(highlight_data["uncertain_sections"])
    display_term_distinctions(highlight_data["term_distinctions"])
    
    return highlight_data

def display_confidence_meter(answer: str) -> Optional[int]:
    """Display confidence meter for the answer
    
    Args:
        answer: The formatted answer text
        
    Returns:
        The confidence score if found, None otherwise
    """
    confidence_score = extract_confidence_score(answer)
    
    if confidence_score is not None:
        confidence_color, confidence_message = get_confidence_color(confidence_score)
        
        st.markdown("### Confidence Assessment")
        st.progress(confidence_score/100.0)
        st.markdown(
            f"**{confidence_message}:** {confidence_score}% - {get_confidence_description(confidence_score)}"
        )
        
        # Add uncertainty warning if needed
        highlight_data = safe_highlight_for_streamlit(answer)
        if confidence_score > 80 and len(highlight_data["uncertain_sections"]) > 0:
            st.warning(
                "High confidence score with uncertainty markers detected. "
                "Consider reviewing the highlighted uncertain sections carefully."
            )
    
    return confidence_score

def display_debug_info(data: Dict[str, Any]) -> None:
    """Display debug information in expandable sections
    
    Args:
        data: Dictionary with debug information
    """
    if "retrieval" in data:
        with st.expander("Retrieved Context"):
            st.text_area("Context", data["retrieval"]["context"], height=200)
    
    if "extraction" in data:
        with st.expander("Extraction Prompt"):
            st.code(data["extraction"]["prompt"], language="")
        with st.expander("Raw Answer"):
            st.text_area("Raw", data["extraction"]["raw_answer"], height=150)
        with st.expander("Extraction Metadata"):
            st.json({
                "model": data["extraction"]["response"]["model"],
                "usage": data["extraction"]["response"]["usage"],
                "id": data["extraction"]["response"]["id"]
            })
    
    if "formatting" in data:
        with st.expander("Formatting Prompt"):
            st.code(data["formatting"]["prompt"], language="")
        with st.expander("Final Answer"):
            st.text_area("Final", data["formatting"]["formatted_answer"], height=150)
        with st.expander("Formatting Metadata"):
            st.json({
                "model": data["formatting"]["response"]["model"],
                "usage": data["formatting"]["response"]["usage"],
                "id": data["formatting"]["response"]["id"]
            })

def display_debug_info_for_index(doc_processor, query: str) -> None:
    """Display debug information for the document index
    
    Args:
        doc_processor: Document processor instance
        query: The user's query
    """
    st.subheader("Index Debug Information")
    
    # Display index statistics
    try:
        if hasattr(doc_processor, 'get_index_stats'):
            stats = doc_processor.get_index_stats()
            st.write(f"Index contains {stats.get('num_chunks', 0)} chunks")
        elif hasattr(doc_processor, 'index_manager') and hasattr(doc_processor.index_manager, 'get_stats'):
            stats = doc_processor.index_manager.get_stats()
            st.write(f"Index contains {stats.get('num_chunks', 0)} chunks")
        else:
            st.write("Index statistics not available")
    except Exception as e:
        st.error(f"Error getting index stats: {str(e)}")
    
    # Show top matches for the query
    if query:
        st.write(f"Top matches for query: '{query}'")
        try:
            # Get top matches
            if hasattr(doc_processor, 'get_top_matches'):
                matches = doc_processor.get_top_matches(query, k=5)
            elif hasattr(doc_processor, 'retrieve'):
                matches = doc_processor.retrieve(query, k=5)
            else:
                st.warning("Top matches retrieval not available")
                return
            
            # Display each match with score
            for i, (score, text) in enumerate(matches):
                with st.expander(f"Match {i+1} (Score: {score:.4f})"):
                    st.text_area(f"Content", text, height=150)
        except Exception as e:
            st.error(f"Error retrieving matches: {str(e)}")