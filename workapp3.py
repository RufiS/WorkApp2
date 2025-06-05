"""WorkApp2 Streamlit Entry Point (Refactored).

Transformed from 745-line monolith to ~100-line orchestrator using extracted controllers.
This file remains the entry point for `streamlit run workapp3.py`
"""

import streamlit as st
import logging
import argparse
import sys

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
from core.controllers.document_controller import DocumentController
from core.controllers.query_controller import QueryController
from core.controllers.testing_controller import TestingController
from core.services.app_orchestrator import AppOrchestrator

# Import config sidebar for Phase 1.4 integration
from utils.ui.config_sidebar import render_config_sidebar  # type: ignore[import] # TODO: Add proper UI types


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the application.

    Returns:
        argparse.Namespace: The parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="KTI Document QA System")
    
    # Dry-run flag
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview index changes without writing to disk"
    )
    
    # Simplified positional arguments
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["production", "development"],
        default="development",
        help="Application mode: 'production' for minimal UI, 'development' for full features (default: development)"
    )
    
    parser.add_argument(
        "features",
        nargs="*",
        help="Optional features: 'splade' for experimental sparse+dense hybrid retrieval"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main application entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Determine application mode from new positional arguments
    production_mode = args.mode == "production"
    development_mode = args.mode == "development"

    # Initialize the application orchestrator
    orchestrator = AppOrchestrator()
    orchestrator.set_dry_run_mode(args.dry_run)
    
    # Set SPLADE mode if specified in features
    splade_enabled = "splade" in args.features
    if splade_enabled:
        orchestrator.set_splade_mode(True)
        logger.info("🧪 SPLADE mode enabled via positional argument")

    # Initialize controllers with mode information
    ui_controller = UIController(orchestrator, production_mode=production_mode)
    document_controller = DocumentController(orchestrator, production_mode=production_mode)
    query_controller = QueryController(orchestrator, production_mode=production_mode)
    testing_controller = TestingController(orchestrator, production_mode=production_mode)

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
        
        # Warm up models for faster response times (only once per session)
        if "models_warmed_up" not in st.session_state:
            with st.spinner("🔥 Warming up models for faster responses..."):
                orchestrator.ensure_models_preloaded()
                st.session_state["models_warmed_up"] = True
                logger.info("✅ Models warmed up successfully for this session")
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

    # Conditionally render sections based on mode
    debug_mode = False
    
    if not production_mode:
        # Development mode: Show all features
        # Render document upload section
        ui_config = orchestrator.get_ui_config()
        uploads = document_controller.render_upload_section(ui_config, orchestrator.is_dry_run_mode())

        # Process uploaded files if any
        if uploads:
            success, message = document_controller.process_uploaded_files(uploads, orchestrator.is_dry_run_mode())
            if success:
                # Display index statistics after successful upload
                document_controller.display_index_statistics(ui_config)
            else:
                ui_controller.display_error(message)
        elif orchestrator.has_index():
            # Display existing index statistics
            document_controller.display_index_statistics(ui_config)

        # Show search method status
        performance_config = orchestrator.get_performance_config()
        retrieval_config = orchestrator.get_retrieval_config()
        ui_controller.render_search_method_status(performance_config, retrieval_config)

        # Render configuration sidebar
        debug_mode = render_config_sidebar(args, doc_processor, orchestrator)
    else:
        # Production mode: Ensure index is loaded silently
        if orchestrator.has_index():
            try:
                retrieval_config = orchestrator.get_retrieval_config()
                doc_processor.load_index(retrieval_config.index_path)
                logger.info("Index loaded for production mode")
            except Exception as e:
                ui_controller.display_error(f"Error loading documents: Please contact administrator")
                return

    # Render the query form
    query, ask_button_pressed = ui_controller.render_query_form()

    # Add systematic testing section (development mode only)
    if not production_mode:
        testing_controller.render_testing_section()

    # Always try to restore cached answers first (handles feedback interactions)
    if not ask_button_pressed:
        # Check for cached answers to restore after feedback interactions
        # Pass empty query to allow cache restoration for feedback interactions
        query_controller.process_query("", debug_mode=debug_mode)
    
    # Process query if submitted
    if ask_button_pressed:
        # Validate query inputs
        error_message = query_controller.validate_query_inputs(query, orchestrator.has_index())
        if error_message:
            ui_controller.display_warning(error_message)
        else:
            # Ensure index is loaded before processing
            if doc_processor.index is None or doc_processor.texts is None:
                try:
                    retrieval_config = orchestrator.get_retrieval_config()
                    doc_processor.load_index(retrieval_config.index_path)
                    logger.info("Loaded index for query processing")
                except Exception as e:
                    ui_controller.display_error(f"Error loading index: {str(e)}")
                    return

            # Process the query using the query controller
            query_controller.process_query(query, debug_mode=debug_mode)

    # Show debugging roadmap progress (development mode only)
    if not production_mode:
        with st.expander("🏗️ Debugging Roadmap - Phase B Complete", expanded=False):
            st.markdown("""
            **Phase B Complete:** ✅ Systematic Testing Framework
            - ✅ Multi-engine testing UI implemented
            - ✅ Configuration comparison system built
            - ✅ Results matrix generation and analysis
            - ✅ A/B testing framework for parameter optimization
            - ✅ TestingController integrated into main app

            **Testing Capabilities Now Available:**
            - 🔬 Single-click testing across all engine configurations
            - 📊 Automated performance comparison with statistical analysis
            - 🎯 Clear identification of best-performing settings
            - 💡 Actionable recommendations for optimization

            **Next Phase:** Phase C - Configuration Audit & Resolution
            """)

    logger.info("Application main function completed")


if __name__ == "__main__":
    main()
