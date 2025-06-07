"""Application Orchestrator for WorkApp2.

Coordinates the initialization and interaction between all application services.
Extracted from workapp3.py to enable better separation of concerns.
"""

import logging
import asyncio
import time
import streamlit as st
from typing import Tuple, Optional, Any

# Type hints for internal modules are allowed to use # type: ignore with reason
from core.config import (  # type: ignore[import] # TODO: Add proper config types
    app_config,
    model_config,
    retrieval_config,
    ui_config,
    performance_config,
)
from core.document_processor import DocumentProcessor  # type: ignore[import] # TODO: Add proper types
from llm.services.llm_service import LLMService  # type: ignore[import] # TODO: Add proper types
from retrieval.retrieval_system import UnifiedRetrievalSystem  # type: ignore[import] # TODO: Add proper types
from utils.error_handling.enhanced_decorators import with_advanced_retry, with_error_tracking  # type: ignore[import] # TODO: Add proper types
from llm.services.model_preloader import ModelPreloader, PreloadConfig, set_global_preloader  # type: ignore[import] # TODO: Add proper types

logger = logging.getLogger(__name__)


class AppOrchestrator:
    """Main application orchestrator that coordinates all services and business logic."""

    def __init__(self) -> None:
        """Initialize the application orchestrator."""
        self.doc_processor: Optional[DocumentProcessor] = None
        self.llm_service: Optional[LLMService] = None
        self.retrieval_system: Optional[UnifiedRetrievalSystem] = None
        self.model_preloader: Optional[ModelPreloader] = None
        self.logger = logger
        self._services_initialized = False
        self._models_preloaded = False

    @with_advanced_retry(max_attempts=3, backoff_factor=2.0)
    @with_error_tracking()
    def initialize_services(self) -> Tuple[DocumentProcessor, LLMService, UnifiedRetrievalSystem]:
        """Initialize the application services.

        Returns:
            Tuple containing document processor, LLM service, and retrieval system

        Raises:
            RuntimeError: If service initialization fails
        """
        try:
            # Use the unified document processor
            doc_processor = DocumentProcessor()

            # Use the consolidated LLM service
            llm_service = LLMService(app_config.api_keys.get("openai", ""))

            # Use the unified retrieval system with enhanced error handling
            retrieval_system = UnifiedRetrievalSystem(doc_processor)
            
            # CRITICAL: Verify SPLADE engine status after initialization
            splade_available = retrieval_system.splade_engine is not None
            logger.info(f"Core services initialized successfully - SPLADE available: {splade_available}")
            
            if not splade_available:
                logger.warning("SPLADE engine not available - check transformers/torch dependencies")

            # Ensure index is loaded if it exists
            if doc_processor.has_index() and (
                doc_processor.index is None or doc_processor.texts is None
            ):
                try:
                    doc_processor.load_index(retrieval_config.index_path)
                    logger.info("Loaded existing index")
                except Exception as e:
                    logger.warning(f"Failed to load existing index: {str(e)}")
                    # Continue without index - user can upload files

            return doc_processor, llm_service, retrieval_system

        except Exception as e:
            error_msg = f"Failed to initialize services: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def _initialize_model_preloader(self) -> None:
        """Initialize the model preloader after services are ready."""
        if self._models_preloaded or not self._services_initialized:
            return
            
        try:
            # Create preload configuration
            preload_config = PreloadConfig(
                enable_llm_preload=True,
                preload_extraction_model=True,
                preload_formatting_model=True,
                preload_timeout_seconds=60,
                enable_embedding_preload=True,
                embedding_preload_timeout_seconds=30,
                llm_warmup_queries=[
                    "What is the main phone number?",
                    "How do I handle customer concerns?",
                    "Test warmup query for model preloading"
                ],
                embedding_warmup_texts=[
                    "phone number contact information",
                    "customer service workflow",
                    "warmup embedding test for preloading"
                ]
            )
            
            # Initialize model preloader with our services
            # Use shared global embedding service for warm-up (fixes 42s embedding delay!)
            from core.embeddings.embedding_service import embedding_service
            self.model_preloader = ModelPreloader(
                llm_service=self.llm_service,
                embedding_service=embedding_service,
                config=preload_config
            )
            
            # Set as global preloader
            set_global_preloader(self.model_preloader)
            
            logger.info("🚀 Model preloader initialized - ready for warm-up")
            
        except Exception as e:
            logger.error(f"Failed to initialize model preloader: {e}")
            # Don't fail the whole app if preloader fails
            self.model_preloader = None

    async def preload_models(self) -> dict:
        """Preload all models for faster response times."""
        if self._models_preloaded:
            logger.info("Models already preloaded, skipping...")
            return {"status": "already_preloaded"}
            
        if not self.model_preloader:
            logger.warning("Model preloader not initialized, skipping preload")
            return {"status": "no_preloader", "error": "Preloader not initialized"}
            
        try:
            logger.info("🔥 Starting model preloading for faster response times...")
            
            # Show progress in Streamlit if available
            if hasattr(st, 'empty'):
                progress_placeholder = st.empty()
                progress_placeholder.info("🔥 Warming up models for faster responses...")
            
            # Phase 1: Run LLM and embedding model preloading
            preload_results = await self.model_preloader.preload_all_models()
            
            # Phase 2: Add FAISS vector index warm-up (THIS WAS THE MISSING 35+ SECOND BOTTLENECK!)
            if self.doc_processor and self.doc_processor.has_index():
                try:
                    vector_start = time.time()
                    logger.info("🔥 Warming up FAISS vector index (the missing piece for 49s delays!)...")
                    
                    # Force load the FAISS index into memory - this was taking 35+ seconds on first query!
                    if self.doc_processor.index is None or self.doc_processor.texts is None:
                        retrieval_config = self.get_retrieval_config()
                        logger.info(f"📂 Loading FAISS index from {retrieval_config.index_path}...")
                        self.doc_processor.load_index(retrieval_config.index_path)
                        index_size = len(self.doc_processor.texts) if self.doc_processor.texts else 0
                        logger.info(f"✅ FAISS index loaded: {index_size} chunks")
                    
                    # Warm up retrieval system with test query to initialize search infrastructure
                    if self.retrieval_system:
                        test_query = "test warmup query for FAISS vector system"
                        try:
                            logger.info("🔍 Performing test FAISS search to warm up infrastructure...")
                            # Perform a lightweight retrieval to warm up the FAISS search pipeline
                            search_results = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.retrieval_system.retrieve(test_query, top_k=5)[0]  # Get just the context
                            )
                            vector_time = time.time() - vector_start
                            preload_results["faiss_index_preload"] = {
                                "success": True,
                                "time_seconds": round(vector_time, 2),
                                "chunks_loaded": index_size if 'index_size' in locals() else 0,
                                "test_results": len(search_results) if search_results else 0
                            }
                            logger.info(f"✅ FAISS index and search pipeline fully warmed up in {vector_time:.1f}s")
                        except Exception as retrieval_e:
                            logger.warning(f"FAISS search warm-up encountered issue: {retrieval_e}")
                            vector_time = time.time() - vector_start
                            preload_results["faiss_index_preload"] = {
                                "success": False,
                                "time_seconds": round(vector_time, 2),
                                "error": str(retrieval_e)
                            }
                    
                except Exception as e:
                    logger.warning(f"FAISS index warm-up failed: {e}")
                    preload_results["faiss_index_preload"] = {
                        "success": False,
                        "time_seconds": 0,
                        "error": str(e)
                    }
            else:
                logger.info("📋 No FAISS index found - skipping vector index warm-up")
                preload_results["faiss_index_preload"] = {
                    "success": True,
                    "time_seconds": 0,
                    "note": "No index to warm up"
                }
            
            self._models_preloaded = True
            
            # Update progress
            if 'progress_placeholder' in locals():
                if preload_results.get("total_preload_time", 0) > 0:
                    progress_placeholder.success(
                        f"✅ Models warmed up in {preload_results['total_preload_time']:.1f}s - ready for fast responses!"
                    )
                else:
                    progress_placeholder.success("✅ Models are now warmed up and ready!")
            
            logger.info(f"🎯 Model preloading completed successfully in {preload_results.get('total_preload_time', 0):.1f}s")
            
            return preload_results
            
        except Exception as e:
            logger.error(f"Model preloading failed: {e}")
            if 'progress_placeholder' in locals():
                progress_placeholder.warning("⚠️ Model warm-up had issues, but app will continue normally")
            return {"status": "failed", "error": str(e)}

    def get_services(self) -> Tuple[DocumentProcessor, LLMService, UnifiedRetrievalSystem]:
        """Get the initialized services, initializing them if needed.

        Returns:
            Tuple containing document processor, LLM service, and retrieval system

        Raises:
            RuntimeError: If service initialization fails
        """
        if not self._services_initialized:
            try:
                self.doc_processor, self.llm_service, self.retrieval_system = self.initialize_services()
                self._services_initialized = True
                self.logger.info("Services retrieved and cached successfully")
                
                # Initialize model preloader after services are ready
                self._initialize_model_preloader()
                
            except Exception as e:
                self.logger.error(f"Failed to get services: {str(e)}")
                raise

        return self.doc_processor, self.llm_service, self.retrieval_system

    def ensure_models_preloaded(self) -> None:
        """Ensure models are preloaded, trigger preloading if not done yet."""
        if not self._models_preloaded and self.model_preloader:
            try:
                # Run async preloading in sync context
                if hasattr(asyncio, 'run'):
                    # Python 3.7+
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # We're in an async context, create a task
                            asyncio.create_task(self.preload_models())
                        else:
                            # We can run directly
                            asyncio.run(self.preload_models())
                    except RuntimeError:
                        # No event loop, create one
                        asyncio.run(self.preload_models())
                else:
                    # Use modern asyncio.run() for Python 3.7+
                    asyncio.run(self.preload_models())
                        
            except Exception as e:
                logger.error(f"Failed to run model preloading: {e}")
                # Don't block the app if preloading fails

    def get_preload_status(self) -> dict:
        """Get the current model preloading status."""
        if not self.model_preloader:
            return {"preloader_available": False, "models_preloaded": False}
            
        status = self.model_preloader.get_preload_status()
        status["preloader_available"] = True
        status["app_models_preloaded"] = self._models_preloaded
        
        return status

    def get_document_processor(self) -> DocumentProcessor:
        """Get the document processor service.

        Returns:
            The document processor instance
        """
        if not self._services_initialized:
            self.get_services()
        return self.doc_processor

    def get_llm_service(self) -> LLMService:
        """Get the LLM service.

        Returns:
            The LLM service instance
        """
        if not self._services_initialized:
            self.get_services()
        return self.llm_service

    def get_retrieval_system(self) -> UnifiedRetrievalSystem:
        """Get the retrieval system service.

        Returns:
            The retrieval system instance
        """
        if not self._services_initialized:
            self.get_services()
        return self.retrieval_system

    def has_index(self) -> bool:
        """Check if the document processor has an index.

        Returns:
            True if an index exists, False otherwise
        """
        try:
            doc_processor = self.get_document_processor()
            return doc_processor.has_index()
        except Exception as e:
            self.logger.error(f"Error checking index status: {str(e)}")
            return False

    def get_config(self, config_name: str) -> Any:
        """Get a configuration object by name.

        Args:
            config_name: Name of the configuration to retrieve

        Returns:
            The requested configuration object

        Raises:
            ValueError: If the configuration name is not recognized
        """
        config_map = {
            "app": app_config,
            "model": model_config,
            "retrieval": retrieval_config,
            "ui": ui_config,
            "performance": performance_config,
        }

        if config_name not in config_map:
            raise ValueError(f"Unknown configuration: {config_name}")

        return config_map[config_name]

    def get_app_config(self) -> Any:
        """Get the application configuration.

        Returns:
            The application configuration object
        """
        return app_config

    def get_ui_config(self) -> Any:
        """Get the UI configuration.

        Returns:
            The UI configuration object
        """
        return ui_config

    def get_performance_config(self) -> Any:
        """Get the performance configuration.

        Returns:
            The performance configuration object
        """
        return performance_config

    def get_retrieval_config(self) -> Any:
        """Get the retrieval configuration.

        Returns:
            The retrieval configuration object
        """
        return retrieval_config

    def is_dry_run_mode(self) -> bool:
        """Check if the application is running in dry-run mode.

        Returns:
            True if in dry-run mode, False otherwise
        """
        # This will be updated when we extract the args parsing
        # For now, check if it exists in session state or return False
        return getattr(st.session_state, 'dry_run_mode', False)

    def set_dry_run_mode(self, dry_run: bool) -> None:
        """Set the dry-run mode for the application.

        Args:
            dry_run: Whether to enable dry-run mode
        """
        st.session_state.dry_run_mode = dry_run
        self.logger.info(f"Dry-run mode set to: {dry_run}")

    def set_splade_mode(self, use_splade: bool) -> None:
        """Set the SPLADE mode for the retrieval system.

        Args:
            use_splade: Whether to enable SPLADE retrieval
        """
        st.session_state.use_splade = use_splade
        if self.retrieval_system:
            self.retrieval_system.use_splade = use_splade
            self.logger.info(f"SPLADE mode set to: {use_splade}")
            if use_splade and self.retrieval_system.splade_engine:
                self.logger.info("🧪 EXPERIMENTAL: SPLADE retrieval engine activated")
            elif use_splade and not self.retrieval_system.splade_engine:
                self.logger.warning("⚠️ SPLADE mode requested but engine not available - falling back to standard retrieval")

    def is_splade_mode_enabled(self) -> bool:
        """Check if SPLADE mode is currently enabled.

        Returns:
            True if SPLADE mode is enabled, False otherwise
        """
        return getattr(st.session_state, 'use_splade', False)

    def cleanup(self) -> None:
        """Clean up resources and services."""
        self.logger.info("Cleaning up orchestrator resources")
        
        # Cleanup model preloader
        if self.model_preloader:
            try:
                asyncio.run(self.model_preloader.cleanup())
            except Exception as e:
                logger.debug(f"Error during model preloader cleanup: {e}")
        
        self.doc_processor = None
        self.llm_service = None
        self.retrieval_system = None
        self.model_preloader = None
        self._services_initialized = False
        self._models_preloaded = False
