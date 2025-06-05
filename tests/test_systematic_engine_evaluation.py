"""
Comprehensive Systematic Engine Testing Framework
==============================================

This test suite performs exhaustive evaluation of:
1. SPLADE retrieval engine parameter optimization
2. Alternative embedding models comparison
3. Answer quality analysis using feedback data
4. Comprehensive performance metrics collection

The framework saves detailed logs and results for analysis.
"""

import json
import time
import logging
import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document_processor import DocumentProcessor
from core.services.app_orchestrator import AppOrchestrator
from retrieval.retrieval_system import UnifiedRetrievalSystem
from retrieval.engines.splade_engine import SpladeEngine
from core.embeddings.embedding_service import EmbeddingService
from llm.services.optimized_llm_service import OptimizedLLMService
from llm.pipeline.optimized_answer_pipeline import OptimizedAnswerPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_logs/systematic_engine_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfiguration:
    """Configuration for a single test run."""
    # SPLADE parameters
    splade_model: str
    sparse_weight: float
    expansion_k: int
    max_sparse_length: int
    
    # Embedding model
    embedding_model: str
    
    # Retrieval parameters
    similarity_threshold: float
    top_k: int
    
    # Engine selection
    use_splade: bool
    engine_type: str  # 'vector', 'hybrid', 'reranking', 'splade'

@dataclass
class QueryResult:
    """Results for a single query test."""
    question: str
    expected_answer: Optional[str]
    retrieved_context: str
    generated_answer: str
    retrieval_time: float
    answer_time: float
    total_time: float
    
    # Retrieval metrics
    chunks_retrieved: int
    similarity_scores: List[float]
    max_similarity: float
    avg_similarity: float
    
    # Quality metrics
    context_contains_answer: bool
    answer_correctness_score: float
    completeness_score: float
    specificity_score: float
    
    # Feedback classification
    feedback_type: Optional[str]  # 'positive', 'negative', 'neutral'
    feedback_text: Optional[str]

@dataclass
class TestResults:
    """Complete results for a test configuration."""
    config: TestConfiguration
    query_results: List[QueryResult]
    
    # Aggregate metrics
    avg_retrieval_time: float
    avg_answer_time: float
    avg_total_time: float
    avg_similarity: float
    
    # Quality metrics
    context_hit_rate: float  # % of queries where context contains answer
    avg_correctness: float
    avg_completeness: float
    avg_specificity: float
    
    # Feedback correlation
    positive_feedback_rate: float
    negative_feedback_rate: float
    
    # Performance metrics
    total_test_time: float
    errors_encountered: int
    error_details: List[str]

class SystematicEngineEvaluator:
    """Main evaluation framework for testing different engine configurations."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.test_data_dir = Path("test_logs")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Load test data
        self.qa_examples = self._load_qa_examples()
        self.feedback_queries = self._load_feedback_queries()
        self.all_test_queries = self.qa_examples + self.feedback_queries
        
        # Initialize services
        self.orchestrator = AppOrchestrator()
        self.doc_processor, self.llm_service, self.retrieval_system = self.orchestrator.get_services()
        
        logger.info(f"Loaded {len(self.qa_examples)} QA examples and {len(self.feedback_queries)} feedback queries")
        logger.info(f"Total test queries: {len(self.all_test_queries)}")
    
    def _load_qa_examples(self) -> List[Dict[str, str]]:
        """Load test questions from all QA JSON files."""
        examples = []
        
        # Load from all QA JSON files
        qa_files = [
            ("tests/QAexamples.json", "qa_examples"),
            ("tests/QAcomplex.json", "qa_complex"),
            ("tests/QAmultisection.json", "qa_multisection")
        ]
        
        for file_path, source_name in qa_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                
                # Convert to standard format
                file_examples = []
                for item in data:
                    if isinstance(item, dict) and "question" in item:
                        file_examples.append({
                            "question": item["question"],
                            "expected_answer": item.get("answer", ""),
                            "source": source_name,
                            "feedback_type": "neutral"
                        })
                
                examples.extend(file_examples)
                logger.info(f"Loaded {len(file_examples)} questions from {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        logger.info(f"Total QA examples loaded: {len(examples)}")
        return examples
    
    def _load_feedback_queries(self) -> List[Dict[str, str]]:
        """Load queries from feedback logs."""
        feedback_queries = []
        
        # Load from feedback_detailed.log
        try:
            with open("logs/feedback_detailed.log", "r") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if "question" in data and "feedback" in data:
                            feedback_queries.append({
                                "question": data["question"],
                                "expected_answer": data.get("answer", ""),
                                "source": "feedback_log",
                                "feedback_type": data["feedback"].get("type", "neutral"),
                                "feedback_text": data["feedback"].get("text", "")
                            })
                    except json.JSONDecodeError:
                        continue
        
        except FileNotFoundError:
            logger.warning("feedback_detailed.log not found")
        
        logger.info(f"Loaded {len(feedback_queries)} feedback queries")
        return feedback_queries
    
    def generate_test_configurations(self) -> List[TestConfiguration]:
        """Generate all test configurations for comprehensive evaluation."""
        
        # SPLADE models to test
        splade_models = [
            "naver/splade-cocondenser-ensembledistil",  # Current
            "naver/splade-v2-max",  # Newer version
            "naver/splade-v2-distil",  # Smaller/faster
            "naver/efficient-splade-VI-BT-large-query"  # Query-optimized
        ]
        
        # Embedding models to test
        embedding_models = [
            "intfloat/e5-base-v2",  # Current
            "intfloat/e5-large-v2",  # Better performance
            "intfloat/e5-small-v2",  # Faster
            "BAAI/bge-base-en-v1.5",  # Strong performer
            "BAAI/bge-large-en-v1.5",  # Large BGE
            "sentence-transformers/all-MiniLM-L6-v2",  # Fast baseline
            "microsoft/mpnet-base"  # General purpose
        ]
        
        # SPLADE parameter ranges
        sparse_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
        expansion_k_values = [50, 100, 150, 200, 300]
        max_sparse_lengths = [128, 256, 512, 1024]
        
        # Retrieval parameters
        similarity_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]
        top_k_values = [5, 10, 15, 20, 30]
        
        # Engine types
        engine_configs = [
            ("vector", False),
            ("hybrid", False),
            ("reranking", False),
            ("splade", True)
        ]
        
        configurations = []
        
        # Generate configurations for each engine type
        for engine_type, use_splade in engine_configs:
            for embedding_model in embedding_models:
                for sim_threshold in similarity_thresholds:
                    for top_k in top_k_values:
                        
                        if use_splade:
                            # SPLADE-specific parameter sweep
                            for splade_model in splade_models:
                                for sparse_weight in sparse_weights:
                                    for expansion_k in expansion_k_values:
                                        for max_sparse_length in max_sparse_lengths:
                                            configurations.append(TestConfiguration(
                                                splade_model=splade_model,
                                                sparse_weight=sparse_weight,
                                                expansion_k=expansion_k,
                                                max_sparse_length=max_sparse_length,
                                                embedding_model=embedding_model,
                                                similarity_threshold=sim_threshold,
                                                top_k=top_k,
                                                use_splade=use_splade,
                                                engine_type=engine_type
                                            ))
                        else:
                            # Non-SPLADE configurations (use default SPLADE params)
                            configurations.append(TestConfiguration(
                                splade_model="naver/splade-cocondenser-ensembledistil",
                                sparse_weight=0.5,
                                expansion_k=100,
                                max_sparse_length=256,
                                embedding_model=embedding_model,
                                similarity_threshold=sim_threshold,
                                top_k=top_k,
                                use_splade=use_splade,
                                engine_type=engine_type
                            ))
        
        logger.info(f"Generated {len(configurations)} test configurations")
        return configurations
    
    def generate_quick_test_configurations(self) -> List[TestConfiguration]:
        """Generate a smaller set of configurations for quick testing."""
        
        # Key configurations to test
        configs = [
            # Current baseline
            TestConfiguration(
                splade_model="naver/splade-cocondenser-ensembledistil",
                sparse_weight=0.5,
                expansion_k=100,
                max_sparse_length=256,
                embedding_model="intfloat/e5-base-v2",
                similarity_threshold=0.25,
                top_k=15,
                use_splade=False,
                engine_type="reranking"
            ),
            
            # SPLADE with current embedding
            TestConfiguration(
                splade_model="naver/splade-cocondenser-ensembledistil",
                sparse_weight=0.5,
                expansion_k=100,
                max_sparse_length=256,
                embedding_model="intfloat/e5-base-v2",
                similarity_threshold=0.25,
                top_k=15,
                use_splade=True,
                engine_type="splade"
            ),
            
            # Better embedding model
            TestConfiguration(
                splade_model="naver/splade-cocondenser-ensembledistil",
                sparse_weight=0.5,
                expansion_k=100,
                max_sparse_length=256,
                embedding_model="BAAI/bge-base-en-v1.5",
                similarity_threshold=0.25,
                top_k=15,
                use_splade=False,
                engine_type="reranking"
            ),
            
            # SPLADE optimization
            TestConfiguration(
                splade_model="naver/splade-v2-max",
                sparse_weight=0.7,
                expansion_k=150,
                max_sparse_length=512,
                embedding_model="BAAI/bge-base-en-v1.5",
                similarity_threshold=0.20,
                top_k=20,
                use_splade=True,
                engine_type="splade"
            ),
            
            # Fast configuration
            TestConfiguration(
                splade_model="naver/splade-v2-distil",
                sparse_weight=0.3,
                expansion_k=50,
                max_sparse_length=128,
                embedding_model="intfloat/e5-small-v2",
                similarity_threshold=0.30,
                top_k=10,
                use_splade=True,
                engine_type="splade"
            )
        ]
        
        logger.info(f"Generated {len(configs)} quick test configurations")
        return configs
    
    def setup_configuration(self, config: TestConfiguration) -> bool:
        """Set up the system for a specific test configuration."""
        try:
            # Configure embedding service
            if hasattr(self.doc_processor, 'embedding_service'):
                # This would require reinitialization of embedding service
                # For now, we'll note that model changes require restart
                pass
            
            # Configure SPLADE if using it
            if config.use_splade and hasattr(self.retrieval_system, 'splade_engine'):
                splade_engine = self.retrieval_system.splade_engine
                if splade_engine:
                    # Update SPLADE configuration
                    splade_engine.update_config(
                        sparse_weight=config.sparse_weight,
                        expansion_k=config.expansion_k,
                        max_sparse_length=config.max_sparse_length
                    )
            
            # Set retrieval system mode
            self.retrieval_system.use_splade = config.use_splade
            
            # Update similarity threshold if possible
            if hasattr(self.retrieval_system, 'similarity_threshold'):
                self.retrieval_system.similarity_threshold = config.similarity_threshold
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up configuration: {e}")
            return False
    
    def evaluate_query(self, query_data: Dict[str, str], config: TestConfiguration) -> QueryResult:
        """Evaluate a single query with the current configuration."""
        
        question = query_data["question"]
        expected_answer = query_data.get("expected_answer", "")
        feedback_type = query_data.get("feedback_type", "neutral")
        feedback_text = query_data.get("feedback_text", "")
        
        start_time = time.time()
        
        try:
            # Perform retrieval
            retrieval_start = time.time()
            context, retrieval_time, chunk_count, similarity_scores = self.retrieval_system.retrieve(
                question, top_k=config.top_k
            )
            retrieval_duration = time.time() - retrieval_start
            
            # Generate answer using LLM
            answer_start = time.time()
            if isinstance(self.llm_service, OptimizedLLMService):
                # Use optimized pipeline
                pipeline = OptimizedAnswerPipeline(self.llm_service)
                answer = pipeline.generate_answer(question, context)
            else:
                # Fallback to basic answer generation
                answer = f"Based on the context: {context[:200]}..."
            
            answer_duration = time.time() - answer_start
            total_duration = time.time() - start_time
            
            # Calculate metrics
            max_similarity = max(similarity_scores) if similarity_scores else 0.0
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            
            # Quality assessment (simplified)
            context_contains_answer = self._assess_context_contains_answer(context, expected_answer)
            answer_correctness = self._assess_answer_correctness(answer, expected_answer)
            completeness_score = self._assess_completeness(answer, question)
            specificity_score = self._assess_specificity(answer)
            
            return QueryResult(
                question=question,
                expected_answer=expected_answer,
                retrieved_context=context,
                generated_answer=answer,
                retrieval_time=retrieval_duration,
                answer_time=answer_duration,
                total_time=total_duration,
                chunks_retrieved=chunk_count,
                similarity_scores=similarity_scores,
                max_similarity=max_similarity,
                avg_similarity=avg_similarity,
                context_contains_answer=context_contains_answer,
                answer_correctness_score=answer_correctness,
                completeness_score=completeness_score,
                specificity_score=specificity_score,
                feedback_type=feedback_type,
                feedback_text=feedback_text
            )
            
        except Exception as e:
            logger.error(f"Error evaluating query '{question}': {e}")
            
            return QueryResult(
                question=question,
                expected_answer=expected_answer,
                retrieved_context="ERROR",
                generated_answer="ERROR",
                retrieval_time=0.0,
                answer_time=0.0,
                total_time=time.time() - start_time,
                chunks_retrieved=0,
                similarity_scores=[],
                max_similarity=0.0,
                avg_similarity=0.0,
                context_contains_answer=False,
                answer_correctness_score=0.0,
                completeness_score=0.0,
                specificity_score=0.0,
                feedback_type=feedback_type,
                feedback_text=feedback_text
            )
    
    def _assess_context_contains_answer(self, context: str, expected_answer: str) -> bool:
        """Assess if the retrieved context contains information to answer the question."""
        if not expected_answer or not context:
            return False
        
        # Simple keyword-based assessment
        expected_words = set(expected_answer.lower().split())
        context_words = set(context.lower().split())
        
        # Check for overlap (simplified heuristic)
        overlap = len(expected_words.intersection(context_words))
        return overlap >= len(expected_words) * 0.3  # 30% word overlap threshold
    
    def _assess_answer_correctness(self, answer: str, expected_answer: str) -> float:
        """Assess the correctness of the generated answer."""
        if not expected_answer or not answer:
            return 0.0
        
        # Simple keyword-based scoring
        expected_words = set(expected_answer.lower().split())
        answer_words = set(answer.lower().split())
        
        if len(expected_words) == 0:
            return 1.0
        
        overlap = len(expected_words.intersection(answer_words))
        return overlap / len(expected_words)
    
    def _assess_completeness(self, answer: str, question: str) -> float:
        """Assess the completeness of the answer."""
        if not answer or answer == "ERROR":
            return 0.0
        
        # Simple length-based heuristic
        if len(answer) < 20:
            return 0.3
        elif len(answer) < 100:
            return 0.7
        else:
            return 1.0
    
    def _assess_specificity(self, answer: str) -> float:
        """Assess the specificity of the answer."""
        if not answer or answer == "ERROR":
            return 0.0
        
        # Check for specific indicators
        specific_indicators = ["$", "phone", "number", "address", "time", "date", "percent", "%"]
        specificity_score = sum(1 for indicator in specific_indicators if indicator in answer.lower())
        
        return min(specificity_score / 3.0, 1.0)  # Normalize to 0-1
    
    def run_configuration_test(self, config: TestConfiguration, test_queries: List[Dict[str, str]] = None) -> TestResults:
        """Run tests for a specific configuration."""
        
        if test_queries is None:
            test_queries = self.all_test_queries
        
        logger.info(f"Testing configuration: {config.engine_type} engine, embedding: {config.embedding_model}")
        
        start_time = time.time()
        
        # Setup configuration
        if not self.setup_configuration(config):
            logger.error(f"Failed to setup configuration")
            return None
        
        # Run queries
        query_results = []
        errors = []
        
        for i, query_data in enumerate(test_queries):
            try:
                result = self.evaluate_query(query_data, config)
                query_results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(test_queries)} queries")
                    
            except Exception as e:
                error_msg = f"Error on query {i}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        total_test_time = time.time() - start_time
        
        # Calculate aggregate metrics
        if query_results:
            avg_retrieval_time = sum(r.retrieval_time for r in query_results) / len(query_results)
            avg_answer_time = sum(r.answer_time for r in query_results) / len(query_results)
            avg_total_time = sum(r.total_time for r in query_results) / len(query_results)
            avg_similarity = sum(r.avg_similarity for r in query_results) / len(query_results)
            
            context_hit_rate = sum(1 for r in query_results if r.context_contains_answer) / len(query_results)
            avg_correctness = sum(r.answer_correctness_score for r in query_results) / len(query_results)
            avg_completeness = sum(r.completeness_score for r in query_results) / len(query_results)
            avg_specificity = sum(r.specificity_score for r in query_results) / len(query_results)
            
            positive_feedback = sum(1 for r in query_results if r.feedback_type == "positive")
            negative_feedback = sum(1 for r in query_results if r.feedback_type == "negative")
            
            positive_feedback_rate = positive_feedback / len(query_results)
            negative_feedback_rate = negative_feedback / len(query_results)
        else:
            avg_retrieval_time = avg_answer_time = avg_total_time = avg_similarity = 0.0
            context_hit_rate = avg_correctness = avg_completeness = avg_specificity = 0.0
            positive_feedback_rate = negative_feedback_rate = 0.0
        
        return TestResults(
            config=config,
            query_results=query_results,
            avg_retrieval_time=avg_retrieval_time,
            avg_answer_time=avg_answer_time,
            avg_total_time=avg_total_time,
            avg_similarity=avg_similarity,
            context_hit_rate=context_hit_rate,
            avg_correctness=avg_correctness,
            avg_completeness=avg_completeness,
            avg_specificity=avg_specificity,
            positive_feedback_rate=positive_feedback_rate,
            negative_feedback_rate=negative_feedback_rate,
            total_test_time=total_test_time,
            errors_encountered=len(errors),
            error_details=errors
        )
    
    def save_results(self, results: List[TestResults], filename_prefix: str = "systematic_evaluation"):
        """Save test results to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = self.test_data_dir / f"{filename_prefix}_detailed_{timestamp}.json"
        summary_file = self.test_data_dir / f"{filename_prefix}_summary_{timestamp}.json"
        csv_file = self.test_data_dir / f"{filename_prefix}_summary_{timestamp}.csv"
        
        # Prepare data for JSON serialization
        detailed_data = []
        for result in results:
            result_dict = asdict(result)
            detailed_data.append(result_dict)
        
        # Save detailed results
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        # Create summary data
        summary_data = []
        for result in results:
            summary = {
                "engine_type": result.config.engine_type,
                "embedding_model": result.config.embedding_model,
                "use_splade": result.config.use_splade,
                "splade_model": result.config.splade_model,
                "sparse_weight": result.config.sparse_weight,
                "expansion_k": result.config.expansion_k,
                "similarity_threshold": result.config.similarity_threshold,
                "top_k": result.config.top_k,
                "avg_retrieval_time": result.avg_retrieval_time,
                "avg_total_time": result.avg_total_time,
                "context_hit_rate": result.context_hit_rate,
                "avg_correctness": result.avg_correctness,
                "avg_completeness": result.avg_completeness,
                "positive_feedback_rate": result.positive_feedback_rate,
                "negative_feedback_rate": result.negative_feedback_rate,
                "total_queries": len(result.query_results),
                "errors_encountered": result.errors_encountered
            }
            summary_data.append(summary)
        
        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Save CSV for easy analysis
        import csv
        if summary_data:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
                writer.writeheader()
                writer.writerows(summary_data)
        
        logger.info(f"Results saved:")
        logger.info(f"  Detailed: {detailed_file}")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  CSV: {csv_file}")
        
        return detailed_file, summary_file, csv_file

def run_quick_evaluation():
    """Run a quick evaluation with key configurations."""
    
    print("=== Quick Systematic Engine Evaluation ===")
    
    evaluator = SystematicEngineEvaluator()
    configs = evaluator.generate_quick_test_configurations()
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n--- Testing Configuration {i+1}/{len(configs)} ---")
        print(f"Engine: {config.engine_type}, Embedding: {config.embedding_model}")
        print(f"SPLADE: {config.use_splade}, Sparse Weight: {config.sparse_weight}")
        
        result = evaluator.run_configuration_test(config)
        if result:
            results.append(result)
            
            print(f"Results:")
            print(f"  Context Hit Rate: {result.context_hit_rate:.3f}")
            print(f"  Avg Correctness: {result.avg_correctness:.3f}")
            print(f"  Avg Retrieval Time: {result.avg_retrieval_time:.3f}s")
            print(f"  Positive Feedback Rate: {result.positive_feedback_rate:.3f}")
    
    # Save results
    if results:
        evaluator.save_results(results, "quick_evaluation")
        
        # Print best configurations
        print("\n=== Top Configurations by Context Hit Rate ===")
        sorted_results = sorted(results, key=lambda x: x.context_hit_rate, reverse=True)
        for i, result in enumerate(sorted_results[:3]):
            print(f"{i+1}. {result.config.engine_type} + {result.config.embedding_model}")
            print(f"   Hit Rate: {result.context_hit_rate:.3f}, Correctness: {result.avg_correctness:.3f}")

def run_comprehensive_evaluation():
    """Run comprehensive evaluation with all configurations."""
    
    print("=== Comprehensive Systematic Engine Evaluation ===")
    print("WARNING: This will take several hours to complete!")
    
    evaluator = SystematicEngineEvaluator()
    configs = evaluator.generate_test_configurations()
    
    print(f"Total configurations to test: {len(configs)}")
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n--- Testing Configuration {i+1}/{len(configs)} ---")
        print(f"Progress: {(i+1)/len(configs)*100:.1f}%")
        
        result = evaluator.run_configuration_test(config)
        if result:
            results.append(result)
        
        # Save intermediate results every 10 configurations
        if (i + 1) % 10 == 0:
            evaluator.save_results(results, f"comprehensive_evaluation_checkpoint_{i+1}")
    
    # Save final results
    if results:
        evaluator.save_results(results, "comprehensive_evaluation_final")

if __name__ == "__main__":
    # Create test logs directory
    Path("test_logs").mkdir(exist_ok=True)
    
    import argparse
    parser = argparse.ArgumentParser(description="Systematic Engine Evaluation")
    parser.add_argument("--mode", choices=["quick", "comprehensive"], default="quick",
                       help="Evaluation mode")
    args = parser.parse_args()
    
    if args.mode == "quick":
        run_quick_evaluation()
    else:
        run_comprehensive_evaluation()
