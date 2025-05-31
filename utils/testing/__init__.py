"""Testing utilities for WorkApp2.

Provides systematic testing and comparison capabilities for different retrieval engine configurations.
"""

from .engine_comparison import EngineComparisonService
from .configuration_matrix import TEST_CONFIGURATIONS, ConfigurationMatrix
from .test_runner import TestRunner

__all__ = [
    "EngineComparisonService",
    "TEST_CONFIGURATIONS", 
    "ConfigurationMatrix",
    "TestRunner"
]
