"""Centralized metrics collection for the application"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MetricEntry:
    """Individual metric entry"""
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Centralized metrics collection and aggregation"""

    def __init__(self, max_entries_per_metric: int = 1000):
        """
        Initialize the metrics collector

        Args:
            max_entries_per_metric: Maximum number of entries to keep per metric
        """
        self.metrics: Dict[str, List[MetricEntry]] = defaultdict(list)
        self.max_entries_per_metric = max_entries_per_metric
        self._counters: Dict[str, int] = defaultdict(int)

    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a metric value

        Args:
            name: Metric name
            value: Metric value
            metadata: Optional metadata
        """
        entry = MetricEntry(value=value, metadata=metadata or {})
        self.metrics[name].append(entry)

        # Trim if needed
        if len(self.metrics[name]) > self.max_entries_per_metric:
            self.metrics[name] = self.metrics[name][-self.max_entries_per_metric:]

    def increment_counter(self, name: str, amount: int = 1) -> None:
        """
        Increment a counter metric

        Args:
            name: Counter name
            amount: Amount to increment by
        """
        self._counters[name] += amount

    def get_metric_stats(self, name: str) -> Dict[str, Any]:
        """
        Get statistics for a metric

        Args:
            name: Metric name

        Returns:
            Dictionary with metric statistics
        """
        if name not in self.metrics or not self.metrics[name]:
            return {"count": 0}

        values = [entry.value for entry in self.metrics[name]]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0,
            "total": sum(values)
        }

    def get_counter_value(self, name: str) -> int:
        """
        Get counter value

        Args:
            name: Counter name

        Returns:
            Counter value
        """
        return self._counters.get(name, 0)

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics and counters

        Returns:
            Dictionary with all metrics
        """
        result = {
            "metrics": {},
            "counters": dict(self._counters)
        }

        for name in self.metrics:
            result["metrics"][name] = self.get_metric_stats(name)

        return result

    def clear_metrics(self) -> None:
        """Clear all metrics"""
        self.metrics.clear()
        self._counters.clear()
        logger.info("All metrics cleared")


# Global metrics collector instance
metrics_collector = MetricsCollector()
