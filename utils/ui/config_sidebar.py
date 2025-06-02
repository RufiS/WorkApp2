"""Configuration sidebar component for the Streamlit application"""

import streamlit as st
from typing import Any

from core.config import (
    model_config,
    retrieval_config,
    performance_config,
    config_manager,
)
from utils.ui import display_debug_info_for_index


def render_config_sidebar(args, doc_processor) -> bool:
    """
    Render the configuration sidebar and return debug mode status

    Args:
        args: Command line arguments
        doc_processor: Document processor instance

    Returns:
        bool: Debug mode status
    """
    # Hide sidebar completely in production mode
    if hasattr(args, 'production') and args.production:
        return False
        
    with st.sidebar:
        st.header("Settings")

        # Display dry-run mode indicator in sidebar if enabled
        if args.dry_run:
            st.warning("⚠️ DRY RUN MODE")

        # Helper function to sync pending config with current config
        def sync_pending_config_with_current():
            """Force sync pending config with actual loaded config"""
            st.session_state.pending_config = {
                "retrieval_k": retrieval_config.top_k,
                "similarity_threshold": retrieval_config.similarity_threshold,
                "extraction_model": model_config.extraction_model,
                "formatting_model": model_config.formatting_model,
                "chunk_size": retrieval_config.chunk_size,
                "chunk_overlap": retrieval_config.chunk_overlap,
                "use_gpu": performance_config.use_gpu_for_faiss,
                "enable_reranking": performance_config.enable_reranking,
                "enhanced_mode": retrieval_config.enhanced_mode,
                "vector_weight": retrieval_config.vector_weight,
            }

        # Helper function to generate config hash for change detection
        def get_config_hash():
            """Generate hash of current config for change detection"""
            config_str = f"{retrieval_config.top_k}-{retrieval_config.similarity_threshold}-{retrieval_config.enhanced_mode}-{model_config.extraction_model}"
            return hash(config_str)

        # Check for config changes and sync if needed
        current_config_hash = get_config_hash()
        if ("config_hash" not in st.session_state or 
            st.session_state.config_hash != current_config_hash or 
            "pending_config" not in st.session_state):
            sync_pending_config_with_current()
            st.session_state.config_hash = current_config_hash
            if "config_hash" in st.session_state:  # Only show message after first load
                st.info("🔄 Configuration synchronized with updated settings")

        # Manual sync button
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🔄 Sync with Config", help="Reload settings from config files"):
                sync_pending_config_with_current()
                st.session_state.config_hash = get_config_hash()
                st.success("✅ Settings synchronized!")
                st.rerun()

        # Configuration status display
        with st.expander("📊 Configuration Status", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Current (Active)")
                st.write(f"**Top K**: {retrieval_config.top_k}")
                st.write(f"**Threshold**: {retrieval_config.similarity_threshold}")
                st.write(f"**Enhanced**: {retrieval_config.enhanced_mode}")
                st.write(f"**Extraction**: {model_config.extraction_model}")
            with col2:
                st.subheader("Pending (Sidebar)")
                st.write(f"**Top K**: {st.session_state.pending_config['retrieval_k']}")
                st.write(f"**Threshold**: {st.session_state.pending_config['similarity_threshold']}")
                st.write(f"**Enhanced**: {st.session_state.pending_config['enhanced_mode']}")
                st.write(f"**Extraction**: {st.session_state.pending_config['extraction_model']}")

        # Parameter sweep recommendations
        with st.expander("🎯 Optimal Settings (from Parameter Sweep)", expanded=False):
            st.info("Based on recent parameter sweep testing:")
            st.write("• **Similarity Threshold**: 0.35 (optimal for text messaging queries)")
            st.write("• **Top K**: 15 (optimal retrieval count)")
            st.write("• **Enhanced Mode**: True (recommended after chunking fixes)")
            st.write("• **Expected Coverage**: >50% (vs 28.57% with old settings)")
            
            if st.button("Apply Optimal Settings"):
                st.session_state.pending_config.update({
                    "similarity_threshold": 0.35,
                    "retrieval_k": 15,
                    "enhanced_mode": True
                })
                st.success("✅ Optimal settings applied to sidebar!")
                st.rerun()

        # Configuration section
        st.subheader("Configuration")

        # Check if there are pending changes
        has_pending_changes = (
            st.session_state.pending_config["retrieval_k"] != retrieval_config.top_k or
            st.session_state.pending_config["similarity_threshold"] != retrieval_config.similarity_threshold or
            st.session_state.pending_config["extraction_model"] != model_config.extraction_model or
            st.session_state.pending_config["formatting_model"] != model_config.formatting_model or
            st.session_state.pending_config["chunk_size"] != retrieval_config.chunk_size or
            st.session_state.pending_config["chunk_overlap"] != retrieval_config.chunk_overlap or
            st.session_state.pending_config["use_gpu"] != performance_config.use_gpu_for_faiss or
            st.session_state.pending_config["enable_reranking"] != performance_config.enable_reranking or
            st.session_state.pending_config["enhanced_mode"] != retrieval_config.enhanced_mode or
            st.session_state.pending_config["vector_weight"] != retrieval_config.vector_weight
        )

        # Show pending changes indicator
        if has_pending_changes:
            st.warning("⚠️ Pending configuration changes")

        # Basic configuration options
        retrieval_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=100,
            value=st.session_state.pending_config["retrieval_k"],
            help="Number of document chunks to retrieve for each query",
        )
        st.session_state.pending_config["retrieval_k"] = retrieval_k

        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=-1.0,
            max_value=1.0,
            value=st.session_state.pending_config["similarity_threshold"],
            step=0.05,
            help="Minimum similarity score for retrieved chunks (higher = more strict)",
        )
        st.session_state.pending_config["similarity_threshold"] = similarity_threshold

        # Apply and Reset buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Apply Settings", help="Apply all pending configuration changes"):
                # Apply all pending changes
                retrieval_config.top_k = st.session_state.pending_config["retrieval_k"]
                retrieval_config.similarity_threshold = st.session_state.pending_config["similarity_threshold"]
                model_config.extraction_model = st.session_state.pending_config["extraction_model"]
                model_config.formatting_model = st.session_state.pending_config["formatting_model"]
                retrieval_config.chunk_size = st.session_state.pending_config["chunk_size"]
                retrieval_config.chunk_overlap = st.session_state.pending_config["chunk_overlap"]
                performance_config.use_gpu_for_faiss = st.session_state.pending_config["use_gpu"]
                performance_config.enable_reranking = st.session_state.pending_config["enable_reranking"]
                retrieval_config.enhanced_mode = st.session_state.pending_config["enhanced_mode"]
                retrieval_config.vector_weight = st.session_state.pending_config["vector_weight"]

                # Update config files
                config_updates = {
                    "retrieval": {
                        "top_k": st.session_state.pending_config["retrieval_k"],
                        "similarity_threshold": st.session_state.pending_config["similarity_threshold"],
                        "chunk_size": st.session_state.pending_config["chunk_size"],
                        "chunk_overlap": st.session_state.pending_config["chunk_overlap"],
                        "enhanced_mode": st.session_state.pending_config["enhanced_mode"],
                        "vector_weight": st.session_state.pending_config["vector_weight"],
                    },
                    "model": {
                        "extraction_model": st.session_state.pending_config["extraction_model"],
                        "formatting_model": st.session_state.pending_config["formatting_model"],
                    },
                    "performance": {
                        "use_gpu_for_faiss": st.session_state.pending_config["use_gpu"],
                        "enable_reranking": st.session_state.pending_config["enable_reranking"],
                    }
                }
                config_manager.update_config(config_updates)
                st.success("✅ Configuration applied successfully!")
                st.rerun()

        with col2:
            if st.button("↩️ Reset", help="Reset all changes to current values"):
                # Reset pending config to current values
                st.session_state.pending_config = {
                    "retrieval_k": retrieval_config.top_k,
                    "similarity_threshold": retrieval_config.similarity_threshold,
                    "extraction_model": model_config.extraction_model,
                    "formatting_model": model_config.formatting_model,
                    "chunk_size": retrieval_config.chunk_size,
                    "chunk_overlap": retrieval_config.chunk_overlap,
                    "use_gpu": performance_config.use_gpu_for_faiss,
                    "enable_reranking": performance_config.enable_reranking,
                    "enhanced_mode": retrieval_config.enhanced_mode,
                    "vector_weight": retrieval_config.vector_weight,
                }
                st.info("🔄 Settings reset to current values")
                st.rerun()

        # Advanced configuration section in an expander
        with st.expander("Advanced Configuration"):
            # Model selection
            extraction_model = st.selectbox(
                "Extraction model",
                options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                index=(
                    0
                    if st.session_state.pending_config["extraction_model"] == "gpt-3.5-turbo"
                    else 1 if st.session_state.pending_config["extraction_model"] == "gpt-4" else 2
                ),
                help="Model used for extracting answers from context",
            )
            st.session_state.pending_config["extraction_model"] = extraction_model

            formatting_model = st.selectbox(
                "Formatting model",
                options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                index=(
                    0
                    if st.session_state.pending_config["formatting_model"] == "gpt-3.5-turbo"
                    else 1 if st.session_state.pending_config["formatting_model"] == "gpt-4" else 2
                ),
                help="Model used for formatting the final answer",
            )
            st.session_state.pending_config["formatting_model"] = formatting_model

            # Chunking parameters
            chunk_size = st.number_input(
                "Chunk size",
                min_value=100,
                max_value=2000,
                value=st.session_state.pending_config["chunk_size"],
                step=100,
                help="Size of document chunks in characters",
            )
            st.session_state.pending_config["chunk_size"] = chunk_size

            chunk_overlap = st.number_input(
                "Chunk overlap",
                min_value=0,
                max_value=500,
                value=st.session_state.pending_config["chunk_overlap"],
                step=50,
                help="Overlap between document chunks in characters",
            )
            st.session_state.pending_config["chunk_overlap"] = chunk_overlap

            # Performance settings
            st.subheader("Performance Settings")

            use_gpu = st.checkbox(
                "Use GPU for embeddings",
                value=st.session_state.pending_config["use_gpu"],
                help="Use GPU for FAISS index operations (if available)",
            )
            st.session_state.pending_config["use_gpu"] = use_gpu

            enable_reranking = st.checkbox(
                "Enable reranking",
                value=st.session_state.pending_config["enable_reranking"],
                help="Use a reranker model to improve retrieval quality (slower)",
            )
            st.session_state.pending_config["enable_reranking"] = enable_reranking

            # Enhanced retrieval settings
            st.subheader("Enhanced Retrieval")

            enhanced_mode = st.checkbox(
                "Enable hybrid search",
                value=st.session_state.pending_config["enhanced_mode"],
                help="Combine vector search with keyword search for better results",
            )
            st.session_state.pending_config["enhanced_mode"] = enhanced_mode

            # Only show vector weight slider if enhanced mode is enabled
            if enhanced_mode:
                vector_weight = st.slider(
                    "Vector search weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.pending_config["vector_weight"],
                    step=0.1,
                    help="Weight for vector search in hybrid mode (0.0 = keyword only, 1.0 = vector only)",
                )
                st.session_state.pending_config["vector_weight"] = vector_weight

            # Add a toggle for raw context display
            st.subheader("Display Options")

            show_raw_context = st.checkbox(
                "Show raw context",
                value=st.session_state.get("show_raw_context", False),
                help="Display the raw context used to generate the answer",
            )
            if show_raw_context:
                st.info(
                    "Raw context will be displayed in a collapsible section below each answer, including retrieval scores."
                )

            if (
                "show_raw_context" not in st.session_state
                or show_raw_context != st.session_state.get("show_raw_context", False)
            ):
                st.session_state["show_raw_context"] = show_raw_context

        # Debug mode toggle
        st.subheader("Debug Options")
        debug_mode = st.checkbox("Debug Mode", value=False, help="Show detailed debug information")

        # Add a debug index button if in debug mode
        if debug_mode:
            st.subheader("Debug Tools")
            debug_query = st.text_input(
                "Debug Query", placeholder="Enter a query to debug the index"
            )
            if st.button("Debug Index"):
                if debug_query:
                    display_debug_info_for_index(doc_processor, debug_query)
                else:
                    st.warning("Please enter a query to debug the index")

    return debug_mode
