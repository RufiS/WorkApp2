"""Answer Quality Parameter Sweep - Systematic Optimization for User Success

Comprehensive testing framework to find optimal retrieval parameters that maximize
user task completion and answer completeness. Uses GPU acceleration for efficient
testing across multiple threshold and top_k combinations.
"""

import logging
import time
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
import itertools

logger = logging.getLogger(__name__)


class AnswerQualityParameterSweep:
    """Systematic parameter optimization for retrieval quality."""
    
    def __init__(self, retrieval_system, llm_service):
        """
        Initialize parameter sweep testing.
        
        Args:
            retrieval_system: UnifiedRetrievalSystem instance
            llm_service: LLM service for answer generation
        """
        self.retrieval_system = retrieval_system
        self.llm_service = llm_service
        self.logger = logger
        
        # Import Answer Quality Analyzer
        from utils.testing.answer_quality_analyzer import AnswerQualityAnalyzer
        self.analyzer = AnswerQualityAnalyzer(retrieval_system, llm_service)
    
    def run_comprehensive_sweep(
        self,
        query: str = "How do I respond to a text message",
        expected_chunks: List[int] = None,
        expected_content_areas: List[str] = None,
        threshold_values: List[float] = None,
        top_k_values: List[int] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive parameter sweep to find optimal retrieval settings.
        
        Args:
            query: Test query to analyze
            expected_chunks: List of chunk IDs that should be retrieved for complete answer
            expected_content_areas: Content areas that should be covered
            threshold_values: Similarity thresholds to test
            top_k_values: Top_k values to test
            save_results: Whether to save detailed results to file
            
        Returns:
            Dictionary with sweep results and optimal configuration
        """
        self.logger.info(f"Starting comprehensive parameter sweep for: '{query}'")
        
        # Default test parameters based on analysis insights
        if threshold_values is None:
            threshold_values = [0.30, 0.35, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60]
        
        if top_k_values is None:
            top_k_values = [10, 15, 20, 25, 30, 40, 50]
        
        if expected_chunks is None:
            expected_chunks = [10, 11, 12, 56, 58, 59, 60]
        
        if expected_content_areas is None:
            expected_content_areas = [
                "RingCentral Texting", "Text Response", "SMS format", 
                "Text Tickets", "Freshdesk ticket", "Field Engineer contact"
            ]
        
        sweep_start_time = time.time()
        results = {
            "sweep_metadata": {
                "query": query,
                "expected_chunks": expected_chunks,
                "expected_content_areas": expected_content_areas,
                "threshold_values": threshold_values,
                "top_k_values": top_k_values,
                "total_combinations": len(threshold_values) * len(top_k_values),
                "start_time": datetime.now().isoformat(),
            },
            "test_results": [],
            "optimal_configuration": None,
            "performance_summary": {}
        }
        
        total_tests = len(threshold_values) * len(top_k_values)
        test_count = 0
        
        self.logger.info(f"Testing {total_tests} combinations: {len(threshold_values)} thresholds × {len(top_k_values)} top_k values")
        
        # Test all combinations
        for threshold, top_k in itertools.product(threshold_values, top_k_values):
            test_count += 1
            self.logger.info(f"Test {test_count}/{total_tests}: threshold={threshold}, top_k={top_k}")
            
            try:
                # Run single test configuration
                test_result = self._run_single_test(
                    query, expected_chunks, expected_content_areas, threshold, top_k
                )
                
                results["test_results"].append(test_result)
                
                # Log progress every 10 tests
                if test_count % 10 == 0:
                    elapsed = time.time() - sweep_start_time
                    avg_time = elapsed / test_count
                    eta = avg_time * (total_tests - test_count)
                    self.logger.info(f"Progress: {test_count}/{total_tests} tests completed. ETA: {eta:.1f}s")
                
            except Exception as e:
                self.logger.error(f"Error in test {test_count}: threshold={threshold}, top_k={top_k}: {str(e)}")
                continue
        
        # Analyze results and find optimal configuration
        self._analyze_sweep_results(results)
        
        sweep_time = time.time() - sweep_start_time
        results["sweep_metadata"]["total_time"] = sweep_time
        results["sweep_metadata"]["end_time"] = datetime.now().isoformat()
        
        self.logger.info(f"Parameter sweep completed in {sweep_time:.1f}s")
        self.logger.info(f"Optimal configuration: {results['optimal_configuration']}")
        
        # Save results if requested
        if save_results:
            self._save_sweep_results(results)
        
        return results
    
    def _run_single_test(
        self, 
        query: str, 
        expected_chunks: List[int], 
        expected_content_areas: List[str],
        threshold: float, 
        top_k: int
    ) -> Dict[str, Any]:
        """Run a single parameter test configuration."""
        test_start_time = time.time()
        
        # Configure retrieval system with test parameters
        original_config = self._backup_current_config()
        self._apply_test_config(threshold, top_k)
        
        try:
            # Run answer quality analysis with new configuration
            analysis_result = self.analyzer.analyze_answer_completeness(
                query, expected_chunks, expected_content_areas
            )
            
            # Extract key metrics for comparison
            test_result = {
                "configuration": {
                    "threshold": threshold,
                    "top_k": top_k
                },
                "test_id": f"threshold_{threshold}_topk_{top_k}",
                "analysis_time": time.time() - test_start_time,
                "user_metrics": self._extract_user_metrics(analysis_result, expected_chunks),
                "performance_metrics": self._extract_performance_metrics(analysis_result),
                "retrieval_details": self._extract_retrieval_details(analysis_result),
                "user_success_score": self._calculate_user_success_score(analysis_result, expected_chunks)
            }
            
            return test_result
            
        finally:
            # Restore original configuration
            self._restore_config(original_config)
    
    def _extract_user_metrics(self, analysis_result: Dict, expected_chunks: List[int]) -> Dict:
        """Extract user-focused success metrics."""
        gap_analysis = analysis_result.get("gap_analysis", {})
        user_impact = analysis_result.get("user_impact", {})
        retrieval_analysis = analysis_result.get("retrieval_analysis", {})
        
        retrieved_chunk_ids = retrieval_analysis.get("retrieved_chunk_ids", [])
        expected_chunks_retrieved = [cid for cid in expected_chunks if cid in retrieved_chunk_ids]
        
        return {
            "expected_chunks_coverage": len(expected_chunks_retrieved) / len(expected_chunks) * 100,
            "expected_chunks_retrieved": expected_chunks_retrieved,
            "expected_chunks_missing": [cid for cid in expected_chunks if cid not in retrieved_chunk_ids],
            "task_completion": user_impact.get("can_complete_task", False),
            "success_probability": user_impact.get("task_success_probability", 0.0),
            "completion_confidence": user_impact.get("completion_confidence", 0.0),
            "frustration_risk": user_impact.get("user_frustration_risk", "unknown"),
            "coverage_percentage": gap_analysis.get("coverage_percentage", 0.0)
        }
    
    def _extract_performance_metrics(self, analysis_result: Dict) -> Dict:
        """Extract performance and efficiency metrics."""
        retrieval_analysis = analysis_result.get("retrieval_analysis", {})
        
        return {
            "chunk_count": retrieval_analysis.get("chunk_count", 0),
            "context_length": retrieval_analysis.get("context_length", 0),
            "retrieval_time": retrieval_analysis.get("retrieval_time", 0.0),
            "total_chunks_available": retrieval_analysis.get("total_chunks_available", 0)
        }
    
    def _extract_retrieval_details(self, analysis_result: Dict) -> Dict:
        """Extract detailed retrieval information."""
        retrieval_analysis = analysis_result.get("retrieval_analysis", {})
        
        return {
            "similarity_scores": retrieval_analysis.get("similarity_scores", []),
            "min_similarity": min(retrieval_analysis.get("similarity_scores", [0])) if retrieval_analysis.get("similarity_scores") else 0,
            "max_similarity": max(retrieval_analysis.get("similarity_scores", [0])) if retrieval_analysis.get("similarity_scores") else 0,
            "retrieved_chunk_ids": retrieval_analysis.get("retrieved_chunk_ids", [])
        }
    
    def _calculate_user_success_score(self, analysis_result: Dict, expected_chunks: List[int]) -> float:
        """Calculate overall user success score (0-1) combining multiple factors."""
        user_metrics = self._extract_user_metrics(analysis_result, expected_chunks)
        
        # Weighted scoring favoring user task completion
        score_components = {
            "expected_coverage": user_metrics["expected_chunks_coverage"] / 100.0,  # 40% weight
            "task_completion": 1.0 if user_metrics["task_completion"] else 0.0,      # 30% weight  
            "success_probability": user_metrics["success_probability"],              # 20% weight
            "completion_confidence": user_metrics["completion_confidence"]           # 10% weight
        }
        
        weights = {
            "expected_coverage": 0.4,
            "task_completion": 0.3, 
            "success_probability": 0.2,
            "completion_confidence": 0.1
        }
        
        total_score = sum(score_components[key] * weights[key] for key in weights)
        return total_score
    
    def _analyze_sweep_results(self, results: Dict) -> None:
        """Analyze sweep results and identify optimal configuration."""
        test_results = results["test_results"]
        
        if not test_results:
            self.logger.warning("No test results to analyze")
            return
        
        # Sort by user success score (descending)
        sorted_results = sorted(test_results, key=lambda x: x["user_success_score"], reverse=True)
        
        # Find optimal configuration
        optimal_result = sorted_results[0]
        results["optimal_configuration"] = {
            "threshold": optimal_result["configuration"]["threshold"],
            "top_k": optimal_result["configuration"]["top_k"],
            "user_success_score": optimal_result["user_success_score"],
            "expected_coverage": optimal_result["user_metrics"]["expected_chunks_coverage"],
            "task_completion": optimal_result["user_metrics"]["task_completion"],
            "success_probability": optimal_result["user_metrics"]["success_probability"]
        }
        
        # Performance summary
        results["performance_summary"] = {
            "total_tests": len(test_results),
            "best_user_success_score": optimal_result["user_success_score"],
            "best_expected_coverage": optimal_result["user_metrics"]["expected_chunks_coverage"],
            "configurations_with_100_percent_coverage": len([
                r for r in test_results 
                if r["user_metrics"]["expected_chunks_coverage"] >= 100.0
            ]),
            "configurations_with_task_completion": len([
                r for r in test_results 
                if r["user_metrics"]["task_completion"]
            ]),
            "top_5_configurations": [
                {
                    "threshold": r["configuration"]["threshold"],
                    "top_k": r["configuration"]["top_k"],
                    "user_success_score": r["user_success_score"],
                    "coverage": r["user_metrics"]["expected_chunks_coverage"]
                }
                for r in sorted_results[:5]
            ]
        }
    
    def _backup_current_config(self) -> Dict:
        """Backup current retrieval configuration."""
        # This would backup current similarity_threshold and top_k settings
        # Implementation depends on how configuration is stored
        return {
            "threshold": getattr(self.retrieval_system, 'similarity_threshold', 0.5),
            "top_k": getattr(self.retrieval_system, 'top_k', 100)
        }
    
    def _apply_test_config(self, threshold: float, top_k: int) -> None:
        """Apply test configuration to retrieval system."""
        # This would set the test parameters on the retrieval system
        # Implementation depends on how configuration is applied
        if hasattr(self.retrieval_system, 'similarity_threshold'):
            self.retrieval_system.similarity_threshold = threshold
        if hasattr(self.retrieval_system, 'top_k'):
            self.retrieval_system.top_k = top_k
    
    def _restore_config(self, original_config: Dict) -> None:
        """Restore original retrieval configuration."""
        if hasattr(self.retrieval_system, 'similarity_threshold'):
            self.retrieval_system.similarity_threshold = original_config["threshold"]
        if hasattr(self.retrieval_system, 'top_k'):
            self.retrieval_system.top_k = original_config["top_k"]
    
    def _save_sweep_results(self, results: Dict) -> str:
        """Save sweep results to timestamped file."""
        timestamp = int(time.time())
        filename = f"answer_quality_parameter_sweep_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Sweep results saved to: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving sweep results: {str(e)}")
            return ""


def run_parameter_sweep_analysis(retrieval_system, llm_service) -> Dict[str, Any]:
    """
    Convenience function to run comprehensive parameter sweep analysis.
    
    Args:
        retrieval_system: UnifiedRetrievalSystem instance
        llm_service: LLM service for answer generation
        
    Returns:
        Dictionary with sweep results and optimal configuration
    """
    sweep = AnswerQualityParameterSweep(retrieval_system, llm_service)
    
    return sweep.run_comprehensive_sweep(
        query="How do I respond to a text message",
        expected_chunks=[10, 11, 12, 56, 58, 59, 60],
        expected_content_areas=[
            "RingCentral Texting", "Text Response", "SMS format", 
            "Text Tickets", "Freshdesk ticket", "Field Engineer contact"
        ]
    )


def run_focused_threshold_sweep(retrieval_system, llm_service) -> Dict[str, Any]:
    """
    Run focused sweep around the identified problem area (threshold 0.35-0.50).
    
    Args:
        retrieval_system: UnifiedRetrievalSystem instance
        llm_service: LLM service for answer generation
        
    Returns:
        Dictionary with focused sweep results
    """
    sweep = AnswerQualityParameterSweep(retrieval_system, llm_service)
    
    # Focused testing based on analysis insights
    return sweep.run_comprehensive_sweep(
        query="How do I respond to a text message",
        expected_chunks=[10, 11, 12, 56, 58, 59, 60],
        expected_content_areas=[
            "RingCentral Texting", "Text Response", "SMS format", 
            "Text Tickets", "Freshdesk ticket", "Field Engineer contact"
        ],
        threshold_values=[0.35, 0.37, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50],  # Focused range
        top_k_values=[15, 20, 25, 30]  # Reasonable range
    )
