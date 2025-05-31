"""Configuration matrix for systematic engine testing.

Defines different test configurations to compare retrieval engine performance.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class TestConfiguration:
    """Represents a single test configuration."""
    name: str
    description: str
    config_overrides: Dict[str, Any]
    expected_behavior: str
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.name or not self.config_overrides:
            raise ValueError("Configuration must have name and config_overrides")


class ConfigurationMatrix:
    """Manages the matrix of test configurations for engine comparison."""
    
    def __init__(self):
        """Initialize the configuration matrix."""
        self._configurations = self._build_test_configurations()
    
    def _build_test_configurations(self) -> List[TestConfiguration]:
        """Build the standard set of test configurations.
        
        Returns:
            List of TestConfiguration objects
        """
        return [
            TestConfiguration(
                name="vector_only",
                description="Pure vector search (baseline)",
                config_overrides={
                    "enhanced_mode": False,
                    "enable_reranking": False,
                    "top_k": 100
                },
                expected_behavior="Fast, basic similarity matching"
            ),
            
            TestConfiguration(
                name="hybrid_balanced",
                description="Balanced hybrid search (50/50 vector/keyword)",
                config_overrides={
                    "enhanced_mode": True,
                    "enable_reranking": False,
                    "vector_weight": 0.5,
                    "top_k": 100
                },
                expected_behavior="Balanced vector and keyword matching"
            ),
            
            TestConfiguration(
                name="hybrid_vector_heavy",
                description="Vector-heavy hybrid search (90/10 vector/keyword)",
                config_overrides={
                    "enhanced_mode": True,
                    "enable_reranking": False,
                    "vector_weight": 0.9,
                    "top_k": 100
                },
                expected_behavior="Mostly vector with keyword boost"
            ),
            
            TestConfiguration(
                name="hybrid_keyword_heavy",
                description="Keyword-heavy hybrid search (30/70 vector/keyword)",
                config_overrides={
                    "enhanced_mode": True,
                    "enable_reranking": False,
                    "vector_weight": 0.3,
                    "top_k": 100
                },
                expected_behavior="Keyword-focused with vector support"
            ),
            
            TestConfiguration(
                name="reranking_enabled",
                description="Full reranking pipeline (highest quality)",
                config_overrides={
                    "enhanced_mode": True,
                    "enable_reranking": True,
                    "top_k": 100
                },
                expected_behavior="Highest quality, slower performance"
            ),
            
            TestConfiguration(
                name="low_threshold",
                description="Permissive similarity threshold",
                config_overrides={
                    "enhanced_mode": True,
                    "enable_reranking": False,
                    "similarity_threshold": 0.3,
                    "top_k": 100
                },
                expected_behavior="More results, potentially lower quality"
            ),
            
            TestConfiguration(
                name="high_threshold",
                description="Strict similarity threshold",
                config_overrides={
                    "enhanced_mode": True,
                    "enable_reranking": False,
                    "similarity_threshold": 0.8,
                    "top_k": 100
                },
                expected_behavior="Fewer, higher quality results"
            ),
            
            TestConfiguration(
                name="small_topk",
                description="Small result set (focused)",
                config_overrides={
                    "enhanced_mode": True,
                    "enable_reranking": False,
                    "top_k": 10
                },
                expected_behavior="Fast, focused results"
            ),
            
            TestConfiguration(
                name="large_topk",
                description="Large result set (comprehensive)",
                config_overrides={
                    "enhanced_mode": True,
                    "enable_reranking": False,
                    "top_k": 200
                },
                expected_behavior="Comprehensive results, slower"
            ),
            
            TestConfiguration(
                name="optimal_candidate",
                description="Optimized configuration candidate",
                config_overrides={
                    "enhanced_mode": True,
                    "enable_reranking": True,
                    "vector_weight": 0.7,
                    "similarity_threshold": 0.5,
                    "top_k": 50
                },
                expected_behavior="Balanced performance and quality"
            )
        ]
    
    def get_all_configurations(self) -> List[TestConfiguration]:
        """Get all test configurations.
        
        Returns:
            List of all test configurations
        """
        return self._configurations.copy()
    
    def get_configuration_by_name(self, name: str) -> TestConfiguration:
        """Get a specific configuration by name.
        
        Args:
            name: Name of the configuration
            
        Returns:
            TestConfiguration object
            
        Raises:
            ValueError: If configuration not found
        """
        for config in self._configurations:
            if config.name == name:
                return config
        raise ValueError(f"Configuration '{name}' not found")
    
    def get_configuration_names(self) -> List[str]:
        """Get list of all configuration names.
        
        Returns:
            List of configuration names
        """
        return [config.name for config in self._configurations]
    
    def get_baseline_configurations(self) -> List[TestConfiguration]:
        """Get baseline configurations for comparison.
        
        Returns:
            List of baseline configurations
        """
        baseline_names = ["vector_only", "hybrid_balanced", "reranking_enabled"]
        return [self.get_configuration_by_name(name) for name in baseline_names]
    
    def get_parameter_sweep_configurations(self) -> List[TestConfiguration]:
        """Get configurations for parameter sweeping.
        
        Returns:
            List of parameter sweep configurations
        """
        sweep_names = [
            "hybrid_balanced", "hybrid_vector_heavy", "hybrid_keyword_heavy",
            "low_threshold", "high_threshold", "small_topk", "large_topk"
        ]
        return [self.get_configuration_by_name(name) for name in sweep_names]


# Global instance for easy access
TEST_CONFIGURATIONS = ConfigurationMatrix()
