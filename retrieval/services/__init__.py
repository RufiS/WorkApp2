"""Retrieval Services package for WorkApp2.

This package contains support services for the retrieval system,
including metrics aggregation and advanced search operations.
"""

__version__ = "0.1.0"

from .metrics_service import MetricsService

__all__ = ["MetricsService"]
