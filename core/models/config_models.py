"""Configuration Data Models for WorkApp2.

Pydantic models for application configuration, replacing dictionary usage.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict, validator


class AppConfig(BaseModel):
    """Model for application configuration."""
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        validate_default=True
    )

    page_title: str = Field("WorkApp Document QA", description="Application page title")
    icon: str = Field("📚", description="Application icon")
    subtitle: Optional[str] = Field("AI-powered document analysis", description="Application subtitle")
    version: str = Field("0.4.0", description="Application version")
    api_keys: Dict[str, str] = Field(default_factory=dict, description="API keys for external services")

    @validator('version')
    def validate_version(cls, v):
        """Validate version format."""
        if not v or len(v.split('.')) != 3:
            raise ValueError('Version must be in format X.Y.Z')
        return v


class RetrievalConfig(BaseModel):
    """Model for retrieval system configuration."""
    model_config = ConfigDict(validate_assignment=True)

    # Core retrieval settings
    top_k: int = Field(5, description="Default number of results to retrieve", ge=1, le=50)
    similarity_threshold: float = Field(0.0, description="Minimum similarity threshold", ge=0.0, le=1.0)
    max_context_length: int = Field(8000, description="Maximum context length in characters", ge=100)

    # Index settings
    index_path: str = Field("data/index", description="Path to store the search index")
    chunk_size: int = Field(500, description="Default chunk size for text splitting", ge=50, le=2000)
    chunk_overlap: int = Field(50, description="Overlap between chunks", ge=0, le=500)

    # Hybrid search settings
    enhanced_mode: bool = Field(False, description="Enable hybrid search mode")
    vector_weight: float = Field(0.7, description="Weight for vector search in hybrid mode", ge=0.0, le=1.0)

    # Reranking settings
    reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Model name for reranking"
    )

    @validator('chunk_overlap')
    def validate_chunk_overlap(cls, v, values):
        """Ensure chunk overlap is less than chunk size."""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('Chunk overlap must be less than chunk size')
        return v


class PerformanceConfig(BaseModel):
    """Model for performance-related configuration."""
    model_config = ConfigDict(validate_assignment=True)

    # Performance toggles
    enable_reranking: bool = Field(False, description="Enable reranking for better quality")
    enable_caching: bool = Field(True, description="Enable caching for better performance")
    enable_async_processing: bool = Field(True, description="Enable async processing")

    # Resource limits
    max_concurrent_requests: int = Field(10, description="Maximum concurrent requests", ge=1, le=100)
    request_timeout: float = Field(30.0, description="Request timeout in seconds", ge=1.0, le=300.0)
    memory_limit_mb: int = Field(2048, description="Memory limit in MB", ge=512)

    # Caching settings
    cache_ttl: int = Field(3600, description="Cache TTL in seconds", ge=60)
    cache_max_size: int = Field(1000, description="Maximum cache entries", ge=10)

    # GPU settings
    use_gpu: bool = Field(False, description="Use GPU for processing if available")
    gpu_memory_fraction: float = Field(0.5, description="Fraction of GPU memory to use", ge=0.1, le=1.0)


class UIConfig(BaseModel):
    """Model for UI configuration."""
    model_config = ConfigDict(validate_assignment=True)

    # Display settings
    show_document_upload: bool = Field(True, description="Show document upload section")
    show_index_statistics: bool = Field(True, description="Show index statistics")
    show_clear_index_button: bool = Field(True, description="Show clear index button")
    show_raw_context: bool = Field(False, description="Show raw context by default")
    show_debug_info: bool = Field(False, description="Show debug information")

    # Layout settings
    sidebar_width: str = Field("wide", description="Sidebar width setting")
    theme: str = Field("auto", description="UI theme (auto, light, dark)")

    # Custom styling
    custom_css: Optional[str] = Field(None, description="Custom CSS styling")
    custom_header: Optional[str] = Field(None, description="Custom header HTML")

    # Progress tracking
    show_progress_bars: bool = Field(True, description="Show progress bars during processing")
    progress_update_interval: float = Field(0.1, description="Progress update interval in seconds")


class ModelConfig(BaseModel):
    """Model for AI model configuration."""
    model_config = ConfigDict(validate_assignment=True)

    # OpenAI settings
    openai_model: str = Field("gpt-3.5-turbo", description="OpenAI model name")
    openai_temperature: float = Field(0.1, description="OpenAI temperature", ge=0.0, le=2.0)
    openai_max_tokens: int = Field(1000, description="Maximum tokens for response", ge=10, le=4000)

    # Embedding settings
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Embedding model name")
    embedding_batch_size: int = Field(100, description="Batch size for embeddings", ge=1, le=1000)

    # Processing settings
    enable_parallel_processing: bool = Field(True, description="Enable parallel model calls")
    max_retries: int = Field(3, description="Maximum retry attempts", ge=0, le=10)
    retry_delay: float = Field(1.0, description="Delay between retries in seconds", ge=0.1)


class LoggingConfig(BaseModel):
    """Model for logging configuration."""
    model_config = ConfigDict(validate_assignment=True)

    # Log levels
    app_log_level: str = Field("INFO", description="Application log level")
    openai_log_level: str = Field("WARNING", description="OpenAI library log level")

    # Log files
    log_directory: str = Field("logs", description="Directory for log files")
    error_log_file: str = Field("workapp_errors.log", description="Error log filename")
    query_log_file: str = Field("query_metrics.log", description="Query metrics log filename")

    # Log rotation
    max_log_size_mb: int = Field(50, description="Maximum log file size in MB", ge=1)
    max_log_files: int = Field(5, description="Maximum number of log files to keep", ge=1)

    # Debug settings
    enable_query_logging: bool = Field(True, description="Enable query logging")
    enable_performance_logging: bool = Field(True, description="Enable performance logging")
    log_context_snippets: bool = Field(False, description="Log context snippets (privacy concern)")

    @validator('app_log_level', 'openai_log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()


class WorkAppConfig(BaseModel):
    """Main configuration model that combines all config sections."""
    model_config = ConfigDict(validate_assignment=True)

    app: AppConfig = Field(default_factory=AppConfig, description="Application configuration")
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig, description="Retrieval configuration")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance configuration")
    ui: UIConfig = Field(default_factory=UIConfig, description="UI configuration")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")

    @classmethod
    def load_from_file(cls, config_path: str) -> 'WorkAppConfig':
        """Load configuration from a JSON file."""
        from ..file_operations import JsonHandler

        config_data = JsonHandler.load_from_file(config_path)
        if config_data is None:
            # Return default configuration if file doesn't exist or is invalid
            return cls()

        return cls(**config_data)

    def save_to_file(self, config_path: str) -> bool:
        """Save configuration to a JSON file."""
        from ..file_operations import JsonHandler

        return JsonHandler.save_to_file(self.model_dump(), config_path, pretty=True)

    def get_section(self, section_name: str) -> BaseModel:
        """Get a specific configuration section."""
        section_map = {
            "app": self.app,
            "retrieval": self.retrieval,
            "performance": self.performance,
            "ui": self.ui,
            "model": self.model,
            "logging": self.logging,
        }

        if section_name not in section_map:
            raise ValueError(f"Unknown configuration section: {section_name}")

        return section_map[section_name]
