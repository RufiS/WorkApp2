"""Application Orchestrator for WorkApp2.

Coordinates the initialization and interaction between all application services.
Extracted from workapp3.py to enable better separation of concerns.
"""

import logging
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

logger = logging.getLogger(__name__)


class AppOrchestrator:
    """Main application orchestrator that coordinates all services and business logic."""

    def __init__(self) -> None:
        """Initialize the application orchestrator."""
        self.doc_processor: Optional[DocumentProcessor] = None
        self.llm_service: Optional[LLMService] = None
        self.retrieval_system: Optional[UnifiedRetrievalSystem] = None
        self.logger = logger
        self._services_initialized = False

    @st.cache_resource
    @with_advanced_retry(max_attempts=3, backoff_factor=2.0)
    @with_error_tracking()
    def initialize_services(_self) -> Tuple[DocumentProcessor, LLMService, UnifiedRetrievalSystem]:
        """Initialize and cache the application services.

        Note: Using _self to work with st.cache_resource decorator

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

            # Use the unified retrieval system
            retrieval_system = UnifiedRetrievalSystem(doc_processor)
            logger.info("Services initialized successfully")

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
            except Exception as e:
                self.logger.error(f"Failed to get services: {str(e)}")
                raise

        return self.doc_processor, self.llm_service, self.retrieval_system

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

    def cleanup(self) -> None:
        """Clean up resources and services."""
        self.logger.info("Cleaning up orchestrator resources")
        self.doc_processor = None
        self.llm_service = None
        self.retrieval_system = None
        self._services_initialized = False
