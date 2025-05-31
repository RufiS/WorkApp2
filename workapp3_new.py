"""WorkApp2 Streamlit Entry Point (Refactored Version).

Transformed from 745-line monolith to ~100-line orchestrator.
This file remains the entry point for `streamlit run workapp3_new.py`
"""

import streamlit as st
import logging
import argparse
import sys
from typing import Optional

# Configure logging early
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)

# Avoid occasional import bugs
sys.modules["torch.classes"] = None

# Import our extracted components
from core.controllers.ui_controller import UIController
from core.services.app_orchestrator import AppOrchestrator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the application.

    Returns:
        argparse.Namespace: The parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="WorkApp Document QA System")
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Preview index changes without writing to disk"
    )
    return parser.parse_args()


def main() -> None:
    """Main application entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Initialize the application orchestrator
    orchestrator = AppOrchestrator()
    orchestrator.set_dry_run_mode(args.dry_run)
    
    # Initialize the UI controller
    ui_controller = UIController(orchestrator)
    
    # Configure the Streamlit page
    ui_controller.configure_page()
    
    # Render the application header
    ui_controller.render_header()
    
    # Show dry-run indicator if enabled
    ui_controller.render_dry_run_indicator(orchestrator.is_dry_run_mode())
    
    # Initialize services with proper error handling
    try:
        doc_processor, llm_service, retrieval_system = orchestrator.get_services()
        logger.info("Services initialized successfully")
    except Exception as e:
        ui_controller.display_error(
            f"Failed to initialize services: {str(e)}",
            suggestions=[
                "Check your OpenAI API key",
                "Verify that required dependencies are installed",
                "Check the application logs for more details",
            ]
        )
        ui_controller.stop_execution()
        return
    
    # TODO: Phase 1.2 - Extract document upload logic to DocumentController
    # For now, show a placeholder
    st.subheader("Document Upload")
    st.info("📝 Document upload functionality will be extracted in Phase 1.2")
    
    # Show search method status
    performance_config = orchestrator.get_performance_config()
    retrieval_config = orchestrator.get_retrieval_config()
    ui_controller.render_search_method_status(performance_config, retrieval_config)
    
    # TODO: Phase 1.2 - Import and render config sidebar
    # For now, show basic info
    with st.sidebar:
        st.subheader("Configuration")
        st.info("🔧 Configuration sidebar will be integrated in Phase 1.2")
    
    # Render the query form
    query, ask_button_pressed = ui_controller.render_query_form()
    
    # TODO: Phase 1.3 - Extract query processing logic to QueryController
    # For now, show a placeholder
    if query and ask_button_pressed:
        if orchestrator.has_index():
            st.info(f"🔍 Query processing for: '{query}' will be extracted in Phase 1.3")
            logger.info(f"Query received: {query[:50]}...")
        else:
            ui_controller.display_warning("Please upload documents first to build an index")
    elif ask_button_pressed and not query:
        ui_controller.display_warning("Please enter a question")
    
    # Show current architecture status
    with st.expander("🏗️ Refactoring Progress", expanded=False):
        st.markdown("""
        **Phase 1.1 Complete:** ✅ UI Controller Foundation
        - Extracted UIController (150 lines)
        - Extracted AppOrchestrator (160 lines)  
        - Reduced workapp3.py from 745 → ~100 lines
        
        **Next Steps:**
        - Phase 1.2: Extract DocumentController
        - Phase 1.3: Extract QueryController (with async processing)
        - Phase 1.4: Complete monolith breakdown
        """)
    
    logger.info("Application main function completed")


if __name__ == "__main__":
    main()
