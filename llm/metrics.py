"""
LLM service metrics tracking

Extracted from llm/llm_service.py
"""
import time
from typing import Dict, Any, List


class MetricsTracker:
    """Tracks LLM service performance metrics"""

    def __init__(self, max_request_times: int = 100):
        """
        Initialize metrics tracker

        Args:
            max_request_times: Maximum number of request times to store
        """
        self.total_requests = 0
        self.total_tokens = 0
        self.request_times: List[float] = []
        self.max_request_times = max_request_times

    def update_metrics(self, start_time: float, response: Dict[str, Any]) -> None:
        """
        Update request metrics

        Args:
            start_time: Request start time
            response: Response from the API
        """
        # Update request count
        self.total_requests += 1

        # Update token count
        if "usage" in response and "total_tokens" in response["usage"]:
            self.total_tokens += response["usage"]["total_tokens"]

        # Update request times
        request_time = time.time() - start_time
        self.request_times.append(request_time)

        # Trim request times if needed
        if len(self.request_times) > self.max_request_times:
            self.request_times = self.request_times[-self.max_request_times :]

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics

        Returns:
            Dictionary with service metrics
        """
        metrics = {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
        }

        # Calculate average request time
        if self.request_times:
            metrics["avg_request_time"] = sum(self.request_times) / len(self.request_times)
            metrics["min_request_time"] = min(self.request_times)
            metrics["max_request_time"] = max(self.request_times)
        else:
            metrics["avg_request_time"] = 0.0
            metrics["min_request_time"] = 0.0
            metrics["max_request_time"] = 0.0

        return metrics

    def get_comprehensive_metrics(self, cache_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive metrics including cache statistics

        Args:
            cache_stats: Cache statistics from CacheManager

        Returns:
            Dictionary with comprehensive service metrics
        """
        metrics = self.get_metrics()
        metrics.update(cache_stats)
        return metrics

    def reset_metrics(self) -> None:
        """Reset all metrics to initial state"""
        self.total_requests = 0
        self.total_tokens = 0
        self.request_times.clear()

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a performance summary with additional calculated metrics

        Returns:
            Dictionary with performance summary
        """
        metrics = self.get_metrics()

        # Calculate tokens per request
        if self.total_requests > 0:
            metrics["avg_tokens_per_request"] = self.total_tokens / self.total_requests
        else:
            metrics["avg_tokens_per_request"] = 0.0

        # Calculate requests per minute based on recent request times
        if len(self.request_times) >= 2:
            # Estimate based on recent activity
            recent_times = self.request_times[-min(10, len(self.request_times)):]
            if recent_times:
                avg_time_between_requests = sum(recent_times) / len(recent_times)
                if avg_time_between_requests > 0:
                    metrics["estimated_requests_per_minute"] = 60 / avg_time_between_requests
                else:
                    metrics["estimated_requests_per_minute"] = 0.0
            else:
                metrics["estimated_requests_per_minute"] = 0.0
        else:
            metrics["estimated_requests_per_minute"] = 0.0

        return metrics
