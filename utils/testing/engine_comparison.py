"""Engine comparison service for systematic testing.

Provides capabilities to test the same query across different retrieval engine configurations
and compare their performance systematically.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .configuration_matrix import TestConfiguration
from utils.logging.retrieval_logger import retrieval_logger

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Results from a single configuration test."""
    configuration_name: str
    query: str
    success: bool
    context: str
    context_length: int
    chunk_count: int
    similarity_scores: List[float]
    retrieval_time: float
    context_quality_score: float
    session_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error_message: Optional[str] = None

    @property
    def average_similarity(self) -> float:
        """Calculate average similarity score."""
        return sum(self.similarity_scores) / len(self.similarity_scores) if self.similarity_scores else 0.0

    @property
    def max_similarity(self) -> float:
        """Get maximum similarity score."""
        return max(self.similarity_scores) if self.similarity_scores else 0.0

    @property
    def success_score(self) -> float:
        """Calculate overall success score (0.0 to 1.0)."""
        if not self.success:
            return 0.0

        # Combine context quality and similarity scores
        quality_weight = 0.6
        similarity_weight = 0.4

        return (
            quality_weight * self.context_quality_score +
            similarity_weight * self.average_similarity
        )


@dataclass
class ComparisonResults:
    """Results from comparing multiple configurations."""
    query: str
    test_results: List[TestResult]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_ranked_results(self) -> List[TestResult]:
        """Get results ranked by success score."""
        return sorted(self.test_results, key=lambda r: r.success_score, reverse=True)

    def get_best_result(self) -> Optional[TestResult]:
        """Get the best performing result."""
        ranked = self.get_ranked_results()
        return ranked[0] if ranked else None

    def get_success_rate_by_config(self) -> Dict[str, float]:
        """Get success rate for each configuration."""
        success_rates = {}
        for result in self.test_results:
            config_name = result.configuration_name
            success_rates[config_name] = result.success_score
        return success_rates


class EngineComparisonService:
    """Service for comparing retrieval engine configurations."""

    def __init__(self, retrieval_system):
        """Initialize the comparison service.

        Args:
            retrieval_system: UnifiedRetrievalSystem instance
        """
        self.retrieval_system = retrieval_system
        self.logger = logger

        # Store original configurations to restore later
        self._original_configs = {}

    def run_single_test(
        self,
        query: str,
        configuration: TestConfiguration
    ) -> TestResult:
        """Run a single test with a specific configuration.

        Args:
            query: Query to test
            configuration: Test configuration to use

        Returns:
            TestResult object with test results
        """
        self.logger.info(f"Testing configuration '{configuration.name}' with query: '{query[:50]}...'")

        try:
            # Apply configuration temporarily
            self._apply_configuration(configuration)

            # Run the retrieval test
            start_time = time.time()
            context, retrieval_time, chunk_count, similarity_scores = self.retrieval_system.retrieve(
                query=query,
                top_k=configuration.config_overrides.get("top_k", 100)
            )

            # Assess context quality
            context_quality = retrieval_logger.assess_context_quality(
                query=query,
                retrieved_context=context,
                chunk_scores=similarity_scores
            )

            # Determine success based on context quality and content
            success = self._assess_success(context, context_quality)

            # Create test result
            result = TestResult(
                configuration_name=configuration.name,
                query=query,
                success=success,
                context=context,
                context_length=len(context),
                chunk_count=chunk_count,
                similarity_scores=similarity_scores,
                retrieval_time=retrieval_time,
                context_quality_score=context_quality,
                session_id=f"test_{configuration.name}_{int(time.time())}"
            )

            self.logger.info(
                f"Test '{configuration.name}' completed: "
                f"success={success}, quality={context_quality:.2f}, "
                f"chunks={chunk_count}, time={retrieval_time:.2f}s"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error testing configuration '{configuration.name}': {str(e)}")

            return TestResult(
                configuration_name=configuration.name,
                query=query,
                success=False,
                context="",
                context_length=0,
                chunk_count=0,
                similarity_scores=[],
                retrieval_time=0.0,
                context_quality_score=0.0,
                session_id=f"test_error_{configuration.name}_{int(time.time())}",
                error_message=str(e)
            )
        finally:
            # Restore original configuration
            self._restore_original_configuration()

    def run_comparison_test(
        self,
        query: str,
        configurations: List[TestConfiguration]
    ) -> ComparisonResults:
        """Run comparison test across multiple configurations.

        Args:
            query: Query to test
            configurations: List of configurations to test

        Returns:
            ComparisonResults with all test results
        """
        self.logger.info(f"Running comparison test with {len(configurations)} configurations")

        test_results = []

        for config in configurations:
            result = self.run_single_test(query, config)
            test_results.append(result)

            # Small delay between tests to avoid overwhelming the system
            time.sleep(0.1)

        comparison_results = ComparisonResults(
            query=query,
            test_results=test_results
        )

        # Log summary
        best_result = comparison_results.get_best_result()
        if best_result:
            self.logger.info(
                f"Comparison complete. Best configuration: '{best_result.configuration_name}' "
                f"(score: {best_result.success_score:.2f})"
            )

        return comparison_results

    def _apply_configuration(self, configuration: TestConfiguration) -> None:
        """Apply a test configuration temporarily.

        Args:
            configuration: Configuration to apply
        """
        try:
            # Store original configurations before modification
            from core.config import retrieval_config, performance_config

            self._original_configs = {
                "retrieval": {},
                "performance": {}
            }

            # Apply configuration overrides
            for key, value in configuration.config_overrides.items():
                if key in ["enhanced_mode", "similarity_threshold", "vector_weight", "top_k"]:
                    # Store original value
                    if hasattr(retrieval_config, key):
                        self._original_configs["retrieval"][key] = getattr(retrieval_config, key)
                        setattr(retrieval_config, key, value)

                elif key in ["enable_reranking"]:
                    # Store original value
                    if hasattr(performance_config, key):
                        self._original_configs["performance"][key] = getattr(performance_config, key)
                        setattr(performance_config, key, value)

            self.logger.debug(f"Applied configuration: {configuration.config_overrides}")

        except Exception as e:
            self.logger.error(f"Error applying configuration: {str(e)}")
            raise

    def _restore_original_configuration(self) -> None:
        """Restore the original configuration after testing."""
        try:
            from core.config import retrieval_config, performance_config

            # Restore retrieval config
            for key, value in self._original_configs.get("retrieval", {}).items():
                if hasattr(retrieval_config, key):
                    setattr(retrieval_config, key, value)

            # Restore performance config
            for key, value in self._original_configs.get("performance", {}).items():
                if hasattr(performance_config, key):
                    setattr(performance_config, key, value)

            self.logger.debug("Original configuration restored")

        except Exception as e:
            self.logger.error(f"Error restoring configuration: {str(e)}")

    def _assess_success(self, context: str, context_quality: float) -> bool:
        """Assess whether a test was successful.

        Args:
            context: Retrieved context
            context_quality: Quality score of the context

        Returns:
            True if test was successful
        """
        # Basic success criteria
        if not context or len(context.strip()) < 10:
            return False

        # Check for error messages indicating failure
        failure_indicators = [
            "No relevant information found",
            "Answer not found",
            "Please contact a manager",
            "Error:",
            "Failed to"
        ]

        context_lower = context.lower()
        for indicator in failure_indicators:
            if indicator.lower() in context_lower:
                return False

        # Quality threshold
        return context_quality > 0.3

    def get_performance_summary(self, results: ComparisonResults) -> Dict[str, Any]:
        """Generate performance summary from comparison results.

        Args:
            results: Comparison results to analyze

        Returns:
            Dictionary with performance summary
        """
        if not results.test_results:
            return {"error": "No test results to analyze"}

        # Calculate statistics
        successful_tests = [r for r in results.test_results if r.success]
        total_tests = len(results.test_results)

        summary = {
            "query": results.query,
            "total_configurations": total_tests,
            "successful_configurations": len(successful_tests),
            "overall_success_rate": len(successful_tests) / total_tests if total_tests > 0 else 0.0,
            "best_configuration": None,
            "performance_rankings": [],
            "recommendations": []
        }

        # Get ranked results
        ranked_results = results.get_ranked_results()

        if ranked_results:
            best_result = ranked_results[0]
            summary["best_configuration"] = {
                "name": best_result.configuration_name,
                "success_score": best_result.success_score,
                "context_quality": best_result.context_quality_score,
                "retrieval_time": best_result.retrieval_time,
                "chunk_count": best_result.chunk_count
            }

            # Create performance rankings
            for i, result in enumerate(ranked_results[:5]):  # Top 5
                ranking = {
                    "rank": i + 1,
                    "configuration": result.configuration_name,
                    "success_score": result.success_score,
                    "quality_score": result.context_quality_score,
                    "retrieval_time": result.retrieval_time,
                    "success": result.success
                }
                summary["performance_rankings"].append(ranking)

        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations(results)

        return summary

    def _generate_recommendations(self, results: ComparisonResults) -> List[str]:
        """Generate optimization recommendations based on test results.

        Args:
            results: Comparison results to analyze

        Returns:
            List of recommendation strings
        """
        recommendations = []

        ranked_results = results.get_ranked_results()
        if not ranked_results:
            return ["No successful configurations found. Check your query and document index."]

        best_result = ranked_results[0]

        # Configuration-specific recommendations
        if best_result.configuration_name == "reranking_enabled":
            recommendations.append("✅ Reranking provides the best results - consider enabling it by default")
        elif best_result.configuration_name == "vector_only":
            recommendations.append("⚡ Vector search is performing well - hybrid may be unnecessary")
        elif "hybrid" in best_result.configuration_name:
            recommendations.append(f"🔄 Hybrid search with current settings works well")

        # Performance recommendations
        avg_time = sum(r.retrieval_time for r in ranked_results) / len(ranked_results)
        if best_result.retrieval_time > avg_time * 2:
            recommendations.append("⚠️ Best configuration is slow - consider performance trade-offs")

        # Quality recommendations
        if best_result.context_quality_score < 0.6:
            recommendations.append("📊 Context quality could be improved - review similarity thresholds")

        return recommendations
