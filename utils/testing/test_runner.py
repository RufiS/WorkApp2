"""Test runner for systematic engine comparison.

Coordinates the execution of test suites and manages test workflows.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .configuration_matrix import TEST_CONFIGURATIONS, TestConfiguration
from .engine_comparison import EngineComparisonService, ComparisonResults

logger = logging.getLogger(__name__)


class TestRunner:
    """Orchestrates systematic testing workflows."""
    
    def __init__(self, retrieval_system):
        """Initialize the test runner.
        
        Args:
            retrieval_system: UnifiedRetrievalSystem instance
        """
        self.retrieval_system = retrieval_system
        self.comparison_service = EngineComparisonService(retrieval_system)
        self.logger = logger
    
    def run_baseline_comparison(self, query: str) -> ComparisonResults:
        """Run baseline comparison with essential configurations.
        
        Args:
            query: Query to test
            
        Returns:
            ComparisonResults for baseline configurations
        """
        self.logger.info(f"Running baseline comparison for query: '{query[:50]}...'")
        
        baseline_configs = TEST_CONFIGURATIONS.get_baseline_configurations()
        return self.comparison_service.run_comparison_test(query, baseline_configs)
    
    def run_full_comparison(self, query: str) -> ComparisonResults:
        """Run comprehensive comparison with all configurations.
        
        Args:
            query: Query to test
            
        Returns:
            ComparisonResults for all configurations
        """
        self.logger.info(f"Running full comparison for query: '{query[:50]}...'")
        
        all_configs = TEST_CONFIGURATIONS.get_all_configurations()
        return self.comparison_service.run_comparison_test(query, all_configs)
    
    def run_parameter_sweep(self, query: str) -> ComparisonResults:
        """Run parameter sweep test with parameter variations.
        
        Args:
            query: Query to test
            
        Returns:
            ComparisonResults for parameter sweep configurations
        """
        self.logger.info(f"Running parameter sweep for query: '{query[:50]}...'")
        
        sweep_configs = TEST_CONFIGURATIONS.get_parameter_sweep_configurations()
        return self.comparison_service.run_comparison_test(query, sweep_configs)
    
    def run_custom_comparison(
        self, 
        query: str, 
        configuration_names: List[str]) -> ComparisonResults:
        """Run comparison with custom selection of configurations.
        
        Args:
            query: Query to test
            configuration_names: List of configuration names to test
            
        Returns:
            ComparisonResults for selected configurations
        """
        self.logger.info(
            f"Running custom comparison for query: '{query[:50]}...' "
            f"with {len(configuration_names)} configurations"
        )
        
        configs = []
        for name in configuration_names:
            try:
                config = TEST_CONFIGURATIONS.get_configuration_by_name(name)
                configs.append(config)
            except ValueError:
                self.logger.warning(f"Configuration '{name}' not found, skipping")
        
        if not configs:
            raise ValueError("No valid configurations found")
        
        return self.comparison_service.run_comparison_test(query, configs)
    
    def get_available_test_suites(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available test suites.
        
        Returns:
            Dictionary describing available test suites
        """
        return {
            "baseline": {
                "name": "Baseline Comparison",
                "description": "Compare essential configurations (vector, hybrid, reranking)",
                "configurations": [c.name for c in TEST_CONFIGURATIONS.get_baseline_configurations()],
                "estimated_time": "30-60 seconds"
            },
            "full": {
                "name": "Full Comparison",
                "description": "Test all available configurations",
                "configurations": TEST_CONFIGURATIONS.get_configuration_names(),
                "estimated_time": "2-5 minutes"
            },
            "parameter_sweep": {
                "name": "Parameter Sweep",
                "description": "Compare different parameter settings",
                "configurations": [c.name for c in TEST_CONFIGURATIONS.get_parameter_sweep_configurations()],
                "estimated_time": "1-3 minutes"
            }
        }
    
    def validate_test_environment(self) -> Dict[str, Any]:
        """Validate that the test environment is ready.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": []
        }
        
        try:
            # Check if retrieval system is available
            if not self.retrieval_system:
                validation_results["valid"] = False
                validation_results["issues"].append("Retrieval system not available")
                return validation_results
            
            # Check if document processor has index
            if not hasattr(self.retrieval_system, 'document_processor'):
                validation_results["valid"] = False
                validation_results["issues"].append("Document processor not available")
                return validation_results
            
            doc_processor = self.retrieval_system.document_processor
            
            # Check if index is loaded
            if not doc_processor.has_index():
                validation_results["valid"] = False
                validation_results["issues"].append("No document index loaded - upload documents first")
                return validation_results
            
            # Check index size
            try:
                if hasattr(doc_processor, 'texts') and doc_processor.texts is not None:
                    index_size = len(doc_processor.texts)
                    if index_size < 10:
                        validation_results["warnings"].append(
                            f"Small index size ({index_size} documents) may produce unreliable test results"
                        )
                    elif index_size > 1000:
                        validation_results["warnings"].append(
                            f"Large index size ({index_size} documents) may cause slower tests"
                        )
            except Exception as e:
                validation_results["warnings"].append(f"Could not determine index size: {str(e)}")
            
            # Check engine availability
            engines_available = []
            if hasattr(self.retrieval_system, 'vector_engine'):
                engines_available.append("vector")
            if hasattr(self.retrieval_system, 'hybrid_engine'):
                engines_available.append("hybrid")
            if hasattr(self.retrieval_system, 'reranking_engine'):
                engines_available.append("reranking")
            
            if len(engines_available) < 3:
                validation_results["warnings"].append(
                    f"Only {len(engines_available)} engines available: {engines_available}"
                )
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def generate_test_report(self, results: ComparisonResults) -> Dict[str, Any]:
        """Generate comprehensive test report.
        
        Args:
            results: Comparison results to analyze
            
        Returns:
            Dictionary with detailed test report
        """
        # Get performance summary
        summary = self.comparison_service.get_performance_summary(results)
        
        # Add additional analysis
        report = {
            "metadata": {
                "query": results.query,
                "timestamp": results.timestamp,
                "total_configurations": len(results.test_results),
                "test_duration": self._calculate_test_duration(results)
            },
            "summary": summary,
            "detailed_results": [],
            "insights": self._generate_insights(results),
            "next_steps": self._suggest_next_steps(results)
        }
        
        # Add detailed results for each configuration
        for result in results.get_ranked_results():
            detailed_result = {
                "configuration": result.configuration_name,
                "rank": len(report["detailed_results"]) + 1,
                "success": result.success,
                "scores": {
                    "overall": result.success_score,
                    "context_quality": result.context_quality_score,
                    "average_similarity": result.average_similarity,
                    "max_similarity": result.max_similarity
                },
                "performance": {
                    "retrieval_time": result.retrieval_time,
                    "chunk_count": result.chunk_count,
                    "context_length": result.context_length
                },
                "context_preview": result.context[:200] + "..." if len(result.context) > 200 else result.context,
                "error_message": result.error_message
            }
            report["detailed_results"].append(detailed_result)
        
        return report
    
    def _calculate_test_duration(self, results: ComparisonResults) -> float:
        """Calculate total test duration from results.
        
        Args:
            results: Comparison results
            
        Returns:
            Total duration in seconds
        """
        return sum(result.retrieval_time for result in results.test_results)
    
    def _generate_insights(self, results: ComparisonResults) -> List[str]:
        """Generate insights from test results.
        
        Args:
            results: Comparison results to analyze
            
        Returns:
            List of insight strings
        """
        insights = []
        
        ranked_results = results.get_ranked_results()
        if not ranked_results:
            return ["No successful configurations - review query and document content"]
        
        # Performance insights
        successful_results = [r for r in ranked_results if r.success]
        if successful_results:
            avg_quality = sum(r.context_quality_score for r in successful_results) / len(successful_results)
            if avg_quality > 0.7:
                insights.append(f"📈 Good overall context quality (avg: {avg_quality:.2f})")
            elif avg_quality < 0.4:
                insights.append(f"📉 Low context quality detected (avg: {avg_quality:.2f})")
        
        # Configuration insights
        best_result = ranked_results[0]
        worst_result = ranked_results[-1]
        
        performance_gap = best_result.success_score - worst_result.success_score
        if performance_gap > 0.5:
            insights.append(f"⚡ Large performance gap ({performance_gap:.2f}) between configurations")
        
        # Engine type analysis
        engine_types = {}
        for result in successful_results:
            engine_type = self._categorize_engine(result.configuration_name)
            if engine_type not in engine_types:
                engine_types[engine_type] = []
            engine_types[engine_type].append(result.success_score)
        
        for engine_type, scores in engine_types.items():
            avg_score = sum(scores) / len(scores)
            insights.append(f"🔧 {engine_type.title()} engines average score: {avg_score:.2f}")
        
        return insights
    
    def _suggest_next_steps(self, results: ComparisonResults) -> List[str]:
        """Suggest next steps based on test results.
        
        Args:
            results: Comparison results to analyze
            
        Returns:
            List of suggested next steps
        """
        next_steps = []
        
        best_result = results.get_best_result()
        if not best_result:
            return ["Upload documents and try again with a different query"]
        
        # Configuration recommendations
        if best_result.success_score > 0.8:
            next_steps.append(f"✅ Deploy '{best_result.configuration_name}' configuration in production")
        elif best_result.success_score > 0.6:
            next_steps.append(f"🔧 Fine-tune '{best_result.configuration_name}' configuration parameters")
        else:
            next_steps.append("🔍 Investigate document content and query relevance")
        
        # Testing recommendations
        successful_configs = [r for r in results.test_results if r.success]
        if len(successful_configs) > 1:
            next_steps.append("📊 Test with more diverse queries to validate consistency")
        
        if best_result.retrieval_time > 2.0:
            next_steps.append("⚡ Optimize for performance if speed is critical")
        
        return next_steps
    
    def _categorize_engine(self, config_name: str) -> str:
        """Categorize engine type from configuration name.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Engine category string
        """
        if "vector" in config_name and "hybrid" not in config_name:
            return "vector"
        elif "hybrid" in config_name:
            return "hybrid"
        elif "reranking" in config_name:
            return "reranking"
        else:
            return "other"
