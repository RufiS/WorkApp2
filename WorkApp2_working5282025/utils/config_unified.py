# Unified configuration module for all application settings
# WorkApp2 Progress: Last completed Error-2
# Next pending Error-4 (File Path Issues)
import os
import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    """Configuration for document retrieval"""
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 50
    similarity_threshold: float = 0.0
    max_context_length: int = 10000
    index_path: str = os.path.join(".", "data", "index")
    enhanced_mode: bool = False
    vector_weight: float = 0.7  # Weight for vector search in hybrid mode (0.0 to 1.0)
    # enable_keyword_fallback moved to PerformanceConfig to avoid duplication

@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    extraction_model: str = "gpt-3.5-turbo"
    formatting_model: str = "gpt-3.5-turbo"
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: int = 60
    retry_attempts: int = 3
    retry_backoff: float = 2.0

@dataclass
class UIConfig:
    """Configuration for the user interface"""
    theme: str = "light"
    font_size: int = 14
    show_sources: bool = True
    show_metrics: bool = False
    max_history: int = 10
    enable_feedback: bool = True
    show_debug_mode: bool = True
    show_clear_index_button: bool = True
    show_index_statistics: bool = True
    show_debug_button: bool = True
    show_document_upload: bool = True

@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations"""
    # Caching settings
    enable_query_cache: bool = True
    query_cache_size: int = 100
    enable_chunk_cache: bool = True
    chunk_cache_size: int = 50
    
    # Batch processing settings
    embedding_batch_size: int = 32
    llm_batch_size: int = 5
    
    # FAISS optimization settings
    enable_faiss_optimization: bool = True
    use_gpu_for_faiss: bool = True
    
    # PDF processing settings
    extract_pdf_hyperlinks: bool = True
    
    # Advanced retrieval settings
    enable_reranking: bool = False
    enable_keyword_fallback: bool = True
    
    # Logging and instrumentation
    log_query_metrics: bool = True
    log_zero_hit_queries: bool = True

@dataclass
class AppConfig:
    """Main application configuration"""
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    data_dir: str = os.path.join(".", "data")
    log_level: str = "INFO"
    error_log_path: str = os.path.join(".", "logs", "workapp_errors.log")
    api_keys: Dict[str, str] = field(default_factory=dict)
    page_title: str = "Document QA"
    version: str = "0.4.0"
    debug_default: bool = False
    ui_settings: Dict[str, Any] = field(default_factory=dict)
    # Additional attributes for UI
    icon: str = "📚"
    subtitle: str = "Ask questions about your documents"

class ConfigManager:
    """Manager for loading and saving configuration"""
    
    def __init__(self, config_path: str = "config.json", performance_config_path: str = "performance_config.json"):
        """
        Initialize the configuration manager
        
        Args:
            config_path: Path to the main configuration file
            performance_config_path: Path to the performance configuration file
        """
        self.config_path = config_path
        self.performance_config_path = performance_config_path
        self.config = self._load_config()
        self.lock = threading.RLock()  # Reentrant lock for thread safety
    
    def _load_config(self) -> AppConfig:
        """
        Load configuration from files
        
        Returns:
            AppConfig instance
        """
        # Initialize with default values
        app_config = AppConfig()
        
        # Load main config if it exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    try:
                        config_data = json.load(f)
                    except json.JSONDecodeError as jde:
                        logger.error(f"Malformed JSON in configuration file {self.config_path}: {str(jde)}")
                        logger.warning(f"Using default configuration values due to malformed config file")
                        return app_config
                
                # Update retrieval config
                if "retrieval" in config_data:
                    for k, v in config_data["retrieval"].items():
                        if hasattr(app_config.retrieval, k):
                            setattr(app_config.retrieval, k, v)
                
                # Update model config
                if "model" in config_data:
                    for k, v in config_data["model"].items():
                        if hasattr(app_config.model, k):
                            setattr(app_config.model, k, v)
                
                # Update UI config
                if "ui" in config_data:
                    for k, v in config_data["ui"].items():
                        if hasattr(app_config.ui, k):
                            setattr(app_config.ui, k, v)
                
                # Update app-level config
                for k, v in config_data.items():
                    if k not in ["retrieval", "model", "ui", "performance"] and hasattr(app_config, k):
                        setattr(app_config, k, v)
                
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {self.config_path}: {str(e)}")
                logger.warning(f"Using default configuration values due to error")
        
        # Load performance config if it exists
        if os.path.exists(self.performance_config_path):
            try:
                with open(self.performance_config_path, "r") as f:
                    try:
                        performance_data = json.load(f)
                    except json.JSONDecodeError as jde:
                        logger.error(f"Malformed JSON in performance configuration file {self.performance_config_path}: {str(jde)}")
                        logger.warning(f"Using default performance configuration values due to malformed config file")
                        return app_config
                
                # Update performance config
                for k, v in performance_data.items():
                    if hasattr(app_config.performance, k):
                        setattr(app_config.performance, k, v)
                
                logger.info(f"Loaded performance configuration from {self.performance_config_path}")
            except Exception as e:
                logger.error(f"Error loading performance configuration from {self.performance_config_path}: {str(e)}")
                logger.warning(f"Using default performance configuration values due to error")
        
        return app_config
    
    def save_config(self) -> None:
        """
        Save configuration to files
        """
        with self.lock:  # Acquire lock before saving
            # Save main config
            try:
                # Convert to dictionary
                config_dict = asdict(self.config)
                
                # Remove performance config (saved separately)
                performance_dict = config_dict.pop("performance")
                
                # Save main config
                with open(self.config_path, "w") as f:
                    json.dump(config_dict, f, indent=2)
                
                logger.info(f"Saved configuration to {self.config_path}")
                
                # Save performance config
                with open(self.performance_config_path, "w") as f:
                    json.dump(performance_dict, f, indent=2)
                
                logger.info(f"Saved performance configuration to {self.performance_config_path}")
            except Exception as e:
                logger.error(f"Error saving configuration: {str(e)}")
                # Attempt to create directory if it doesn't exist
                try:
                    config_dir = os.path.dirname(self.config_path)
                    if config_dir and not os.path.exists(config_dir):
                        os.makedirs(config_dir)
                        logger.info(f"Created directory {config_dir} for configuration files")
                        # Try saving again
                        self.save_config()
                except Exception as dir_error:
                    logger.error(f"Error creating directory for configuration: {str(dir_error)}")
    
    def get_config(self) -> AppConfig:
        """
        Get the current configuration
        
        Returns:
            AppConfig instance
        """
        with self.lock:  # Acquire lock before reading
            return self.config
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values
        
        Args:
            config_updates: Dictionary with configuration updates
        """
        with self.lock:  # Acquire lock before updating
            # Update nested configs
            for section in ["retrieval", "model", "ui", "performance"]:
                if section in config_updates:
                    section_config = getattr(self.config, section)
                    for k, v in config_updates[section].items():
                        if hasattr(section_config, k):
                            setattr(section_config, k, v)
                        else:
                            logger.warning(f"Unknown configuration key: {section}.{k}")
            
            # Update app-level config
            for k, v in config_updates.items():
                if k not in ["retrieval", "model", "ui", "performance"] and hasattr(self.config, k):
                    setattr(self.config, k, v)
                elif k not in ["retrieval", "model", "ui", "performance"]:
                    logger.warning(f"Unknown configuration key: {k}")
            
            # Save updated config
            self.save_config()

# Create global configuration manager
config_manager = ConfigManager(config_path="config.json", performance_config_path="performance_config.json")

# Path resolution utility function
def resolve_path(path: str, create_dir: bool = False) -> str:
    """
    Resolve a path relative to the application root directory
    
    Args:
        path: Path to resolve (can be relative or absolute)
        create_dir: If True, create the directory if it doesn't exist
        
    Returns:
        Resolved absolute path
    """
    # If path is already absolute, return it as is
    if os.path.isabs(path):
        resolved_path = path
    else:
        # Get the application root directory (parent of the utils directory)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Join with the relative path using os.path.join for cross-platform compatibility
        resolved_path = os.path.join(root_dir, path)
    
    # Create directory if requested and it doesn't exist
    if create_dir and not os.path.exists(os.path.dirname(resolved_path)):
        try:
            os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
            logger.info(f"Created directory: {os.path.dirname(resolved_path)}")
        except Exception as e:
            logger.error(f"Error creating directory {os.path.dirname(resolved_path)}: {str(e)}")
    
    return resolved_path

# Export configuration objects for easy access
app_config = config_manager.config
retrieval_config = app_config.retrieval
model_config = app_config.model
ui_config = app_config.ui
performance_config = app_config.performance