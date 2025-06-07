#!/usr/bin/env python3
"""
COMPREHENSIVE Systematic Engine Testing Framework
===============================================

This framework performs EXHAUSTIVE evaluation of ALL parameter combinations:
1. Multiple embedding models with index regeneration 
2. Multiple SPLADE models and parameter sweeps
3. Multiple reranker models
4. All engine types and retrieval parameters
5. Proper model switching and configuration management
6. Comprehensive logging and progress tracking

This is the REAL systematic testing framework that tests thousands of configurations.
"""

import json
import time
import logging
import shutil
import os
import csv
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import sys
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.services.app_orchestrator import AppOrchestrator
from core.document_processor import DocumentProcessor
from core.embeddings.embedding_service import EmbeddingService
from retrieval.retrieval_system import UnifiedRetrievalSystem
from retrieval.engines.splade_engine import SpladeEngine
from retrieval.engines.reranking_engine import RerankingEngine
from llm.services.llm_service import LLMService
from llm.pipeline.answer_pipeline import AnswerPipeline

# Configure ULTRA-QUIET logging with controlled console output
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings from libraries

# Ensure test_logs directory exists
Path("test_logs").mkdir(exist_ok=True)

# Set up file-only detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_logs/comprehensive_systematic_evaluation.log', mode='w')
    ]
)

# Silence external library loggers
external_loggers = [
    'transformers', 'sentence_transformers', 'torch', 'tensorflow',
    'faiss', 'numpy', 'sklearn', 'urllib3', 'requests', 'huggingface_hub'
]
for logger_name in external_loggers:
    ext_logger = logging.getLogger(logger_name)
    ext_logger.setLevel(logging.ERROR)
    ext_logger.propagate = False

# Create our test logger with minimal console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler for CRITICAL progress only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.CRITICAL)  # Only critical messages to console
logger.addHandler(console_handler)

def print_progress(message: str):
    """Controlled progress output - minimal console spam."""
    print(f"🔄 {message}")

def print_critical(message: str):
    """Critical status messages."""
    print(f"⚠️  {message}")
    logger.critical(message)

@dataclass
class ComprehensiveTestConfiguration:
    """Configuration for systematic testing with ALL parameters and pipeline variants."""
    
    # Model configurations
    embedding_model: str
    splade_model: str
    reranker_model: str
    
    # Pipeline configuration - THIS IS THE KEY ENHANCEMENT
    pipeline_type: str  # 'pure_splade', 'reranker_then_splade', 'splade_then_reranker', 'vector_baseline'
    
    # SPLADE parameters
    sparse_weight: float
    expansion_k: int
    max_sparse_length: int
    
    # Retrieval parameters
    similarity_threshold: float
    top_k: int
    rerank_top_k: int
    
    # System parameters
    index_regenerated: bool = False
    config_id: str = ""

@dataclass
class ComprehensiveQueryResult:
    """Results for a single query evaluation."""
    question: str
    expected_answer: Optional[str]
    source: str
    retrieved_context: str
    generated_answer: str
    
    # Timing metrics
    retrieval_time: float
    answer_time: float
    total_time: float
    
    # Quality metrics
    chunks_retrieved: int
    similarity_scores: List[float]
    max_similarity: float
    avg_similarity: float
    context_contains_answer: bool
    answer_correctness_score: float
    completeness_score: float
    specificity_score: float
    
    # Feedback data
    feedback_type: Optional[str]
    feedback_text: Optional[str]
    
    # Error tracking
    error_occurred: bool = False
    error_message: str = ""

@dataclass
class ComprehensiveTestResults:
    """Complete results for a configuration test."""
    config: ComprehensiveTestConfiguration
    query_results: List[ComprehensiveQueryResult]
    
    # Aggregate performance metrics
    avg_retrieval_time: float
    avg_answer_time: float
    avg_total_time: float
    avg_similarity: float
    
    # Quality metrics
    context_hit_rate: float
    avg_correctness: float
    avg_completeness: float
    avg_specificity: float
    
    # Feedback correlation
    positive_feedback_rate: float
    negative_feedback_rate: float
    neutral_feedback_rate: float
    
    # Error metrics
    total_test_time: float
    errors_encountered: int
    error_details: List[str]
    success_rate: float
    
    # Model info
    models_used: Dict[str, str]

class ComprehensiveSystematicEvaluator:
    """EXHAUSTIVE systematic evaluation framework with cost-effective dual-track evaluation."""
    
    def __init__(self):
        """Initialize the comprehensive evaluator."""
        self.test_data_dir = Path("test_logs")
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Index backup directory for different embedding models
        self.index_backup_dir = Path("test_index_backups") 
        self.index_backup_dir.mkdir(exist_ok=True)
        
        # Load ALL test data
        self.qa_examples = self._load_qa_examples_with_encoding_fix()
        self.feedback_queries = self._load_feedback_queries()
        self.all_test_queries = self.qa_examples + self.feedback_queries
        
        # Track current configuration
        self.current_embedding_model = None
        self.current_reranker_model = None
        self.current_pipeline_type = None  # Track pipeline type for proper SPLADE scoping
        
        # CRITICAL FIX: Track indexing-time parameters that require index rebuild
        self.current_max_sparse_length = None
        self.current_expansion_k = None
        self.current_splade_model = None  # SPLADE model also affects document indexing
        
        self.orchestrator = None
        self.doc_processor = None
        self.llm_service = None 
        self.retrieval_system = None
        
        # Progress tracking
        self.completed_configs = 0
        self.total_configs = 0
        self.start_time = None
        
        # COST-EFFECTIVE DUAL-TRACK EVALUATION SETUP
        self.ollama_service = None
        self.semantic_evaluator = None
        self.openai_cost_tracker = {"total_calls": 0, "total_cost": 0.0}
        self.semantic_scores = []  # Track all semantic scores for smart sampling
        
        self._setup_cost_effective_evaluation()
        
        logger.info(f"Comprehensive evaluator initialized:")
        logger.info(f"  QA Examples: {len(self.qa_examples)}")
        logger.info(f"  Feedback Queries: {len(self.feedback_queries)}")
        logger.info(f"  Total Test Queries: {len(self.all_test_queries)}")
        logger.info(f"  Cost-effective evaluation: Ollama + E5-base-v2 + OpenAI spot-check")
    
    def _setup_cost_effective_evaluation(self):
        """Setup cost-effective dual-track evaluation system."""
        try:
            print_progress("Setting up cost-effective evaluation system...")
            
            # 1. Setup Ollama LLM Service
            self._setup_ollama_service()
            
            # 2. Setup Semantic Evaluator using existing E5-base-v2
            self._setup_semantic_evaluator()
            
            # 3. Initialize cost tracking
            self.openai_cost_tracker = {
                "total_calls": 0,
                "total_cost": 0.0,
                "spot_check_calls": 0,
                "budget_limit": 15.0  # $15 budget limit
            }
            
            logger.info("✅ Cost-effective evaluation system ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup cost-effective evaluation: {e}")
            # Continue with degraded mode
            self.ollama_service = None
            self.semantic_evaluator = None
    
    def _setup_ollama_service(self):
        """Setup Ollama service for local LLM generation."""
        try:
            print_progress("Connecting to Ollama server...")
            
            # Test connection first
            response = requests.get("http://192.168.254.204:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                if any('qwen2.5:14b-instruct' in name for name in model_names):
                    print_progress("✅ Ollama server connected, qwen2.5:14b-instruct available")
                    self.ollama_service = "connected"  # Simple flag for now
                    logger.info("Ollama service ready for cost-effective generation")
                else:
                    logger.warning(f"qwen2.5:14b-instruct not found. Available: {model_names}")
                    self.ollama_service = None
            else:
                logger.error(f"Ollama server error: {response.status_code}")
                self.ollama_service = None
                
        except Exception as e:
            logger.error(f"Cannot connect to Ollama server: {e}")
            self.ollama_service = None
    
    def _setup_semantic_evaluator(self):
        """Setup semantic evaluator using existing E5-base-v2 model."""
        try:
            print_progress("Loading E5-base-v2 for semantic evaluation...")
            
            # Use the same model that's already configured in the system
            self.semantic_evaluator = SentenceTransformer('intfloat/e5-base-v2')
            
            print_progress("✅ Semantic evaluator ready (E5-base-v2)")
            logger.info("Semantic evaluator initialized with E5-base-v2")
            
        except Exception as e:
            logger.error(f"Failed to setup semantic evaluator: {e}")
            self.semantic_evaluator = None
    
    def _generate_answer_cost_effective(self, question: str, context: str) -> Dict[str, Any]:
        """Generate answer using cost-effective Ollama service."""
        
        if not self.ollama_service:
            # Fallback to original LLM service
            return self._generate_answer_fallback(question, context)
        
        start_time = time.time()
        
        try:
            # Build prompt for Ollama
            prompt = self._build_ollama_prompt(question, context)
            
            # Call Ollama API with increased context window
            payload = {
                "model": "qwen2.5:14b-instruct",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 4096,
                    "num_ctx": 8192,  # Increase context window to 8K tokens
                    "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
                }
            }
            
            response = requests.post(
                "http://192.168.254.204:11434/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                generation_time = time.time() - start_time
                
                return {
                    'content': answer,
                    'model': 'qwen2.5:14b-instruct',
                    'generation_time': generation_time,
                    'cost': 0.0  # Free local generation
                }
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            # Fallback to original service
            return self._generate_answer_fallback(question, context)
    
    def _build_ollama_prompt(self, question: str, context: str) -> str:
        """Build optimized prompt for Ollama model with length management."""
        
        system_prompt = """You are an expert assistant for a dispatch and customer service system. Provide accurate, helpful answers based on the provided context.

Instructions:
- Answer based ONLY on the provided context
- Be specific and actionable 
- If context lacks information, say so clearly
- For procedures, list steps clearly
- Include relevant phone numbers, codes, or specifics
- Keep answers concise but complete"""

        # Estimate token count (rough: 1 token ≈ 4 characters)
        max_tokens = 7500  # Match 8K context window, leave buffer for response
        system_tokens = len(system_prompt) // 4
        question_tokens = len(question) // 4
        overhead_tokens = 50  # For formatting and structure
        
        available_context_tokens = max_tokens - system_tokens - question_tokens - overhead_tokens
        max_context_chars = available_context_tokens * 4
        
        # Truncate context if needed
        if len(context) > max_context_chars:
            # Keep the beginning of context (most relevant chunks typically come first)
            truncated_context = context[:max_context_chars]
            # Try to end at a sentence boundary
            last_period = truncated_context.rfind('.')
            if last_period > max_context_chars * 0.8:  # If we can find a good stopping point
                truncated_context = truncated_context[:last_period + 1]
            
            context = truncated_context + "\n\n[CONTEXT TRUNCATED DUE TO LENGTH]"
            logger.warning(f"Context truncated from {len(context)} to {len(truncated_context)} chars to fit token limit")

        user_prompt = f"""Context from company documents:
{context}

Question: {question}

Answer:"""

        final_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Final safety check
        estimated_tokens = len(final_prompt) // 4
        if estimated_tokens > max_tokens:
            logger.warning(f"Prompt still too long: {estimated_tokens} estimated tokens, further reducing context")
            # Emergency truncation
            emergency_max_context = max_context_chars // 2
            context = context[:emergency_max_context] + "\n\n[CONTEXT HEAVILY TRUNCATED]"
            user_prompt = f"""Context from company documents:
{context}

Question: {question}

Answer:"""
            final_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        return final_prompt
    
    def _generate_answer_fallback(self, question: str, context: str) -> Dict[str, Any]:
        """Fallback to original LLM service."""
        if self.llm_service and hasattr(self.llm_service, 'generate_answer'):
            return self.llm_service.generate_answer(question, context)
        else:
            return {
                'content': f"ERROR: No LLM service available. Context: {context[:200]}...",
                'model': 'fallback',
                'generation_time': 0.0,
                'cost': 0.0
            }
    
    def _evaluate_answer_semantic(self, question: str, expected_answer: str, generated_answer: str, context: str) -> Dict[str, float]:
        """Evaluate answer quality using semantic similarity (FREE)."""
        
        if not self.semantic_evaluator or not generated_answer or generated_answer.startswith("ERROR"):
            return self._get_default_scores()
        
        try:
            # 1. Semantic similarity between expected and generated
            if expected_answer:
                try:
                    expected_emb = self.semantic_evaluator.encode([expected_answer])
                    generated_emb = self.semantic_evaluator.encode([generated_answer])
                    sim_matrix = cosine_similarity(expected_emb, generated_emb)
                    semantic_similarity = float(sim_matrix[0][0]) if sim_matrix.shape == (1, 1) else 0.0
                except (IndexError, ValueError, AttributeError):
                    semantic_similarity = 0.0
            else:
                semantic_similarity = 0.5  # Default when no expected answer
            
            # 2. Question relevance
            try:
                question_emb = self.semantic_evaluator.encode([question])
                generated_emb = self.semantic_evaluator.encode([generated_answer])
                sim_matrix = cosine_similarity(question_emb, generated_emb)
                question_relevance = float(sim_matrix[0][0]) if sim_matrix.shape == (1, 1) else 0.0
            except (IndexError, ValueError, AttributeError):
                question_relevance = 0.0
            
            # 3. Contextual coherence
            if context:
                try:
                    context_emb = self.semantic_evaluator.encode([context])
                    generated_emb = self.semantic_evaluator.encode([generated_answer])
                    sim_matrix = cosine_similarity(context_emb, generated_emb)
                    contextual_coherence = float(sim_matrix[0][0]) if sim_matrix.shape == (1, 1) else 0.0
                except (IndexError, ValueError, AttributeError):
                    contextual_coherence = 0.0
            else:
                contextual_coherence = 0.5
            
            # 4. Length and completeness heuristics
            completeness = min(len(generated_answer) / 100.0, 1.0)  # Normalize by 100 chars
            
            # Combine into overall semantic score
            overall_score = (
                semantic_similarity * 0.4 + 
                question_relevance * 0.3 + 
                contextual_coherence * 0.2 + 
                completeness * 0.1
            )
            
            # Store for smart sampling
            score_data = {
                "overall_score": overall_score,
                "semantic_similarity": semantic_similarity,
                "question_relevance": question_relevance,
                "contextual_coherence": contextual_coherence,
                "completeness": completeness
            }
            self.semantic_scores.append(score_data)
            
            return score_data
            
        except Exception as e:
            logger.error(f"Semantic evaluation failed: {e}")
            return self._get_default_scores()
    
    def _get_default_scores(self) -> Dict[str, float]:
        """Default scores for failed evaluations."""
        return {
            "overall_score": 0.0,
            "semantic_similarity": 0.0,
            "question_relevance": 0.0,
            "contextual_coherence": 0.0,
            "completeness": 0.0
        }
    
    def _should_spot_check_with_openai(self, config_index: int, total_configs: int) -> bool:
        """Smart sampling: decide if this config should get OpenAI spot-check."""
        
        # Budget check
        if self.openai_cost_tracker["total_cost"] >= self.openai_cost_tracker["budget_limit"]:
            return False
        
        # Smart sampling strategy:
        # 1. Top 10% by semantic score
        # 2. Random 5% sample
        # 3. Outliers (very high or low scores)
        
        if len(self.semantic_scores) < 10:  # Not enough data yet
            return config_index % 20 == 0  # Every 20th config initially
        
        recent_scores = self.semantic_scores[-50:]  # Last 50 configs
        avg_score = sum(s["overall_score"] for s in recent_scores) / len(recent_scores) if recent_scores else 0.5
        
        current_score = self.semantic_scores[-1]["overall_score"] if self.semantic_scores else 0.5
        
        # Top performer (above 90th percentile)
        if current_score > avg_score + 0.2:
            return True
        
        # Outlier (very low performance)
        if current_score < avg_score - 0.3:
            return True
        
        # Random sampling (5%)
        if config_index % 20 == 0:
            return True
        
        return False
    
    def _spot_check_with_openai(self, question: str, expected_answer: str, generated_answer: str, context: str) -> Dict[str, float]:
        """Spot-check answer quality with OpenAI (COST: ~$0.002 per call)."""
        
        if self.openai_cost_tracker["total_cost"] >= self.openai_cost_tracker["budget_limit"]:
            logger.warning("OpenAI budget limit reached, skipping spot-check")
            return self._get_default_scores()
        
        try:
            # Import here to avoid issues if not available
            from llm.services.llm_service import LLMService
            
            # Create minimal LLM service for evaluation
            llm = LLMService()
            
            evaluation_prompt = f"""Evaluate this Q&A response on a scale of 0.0-1.0 for each metric:

QUESTION: {question}
EXPECTED ANSWER: {expected_answer}
GENERATED ANSWER: {generated_answer}
CONTEXT: {context[:500]}...

Rate each aspect (0.0-1.0):
1. FACTUAL_ACCURACY: Are the key facts correct?
2. COMPLETENESS: Does it answer the full question?
3. RELEVANCE: Is the answer on-topic and helpful?
4. SPECIFICITY: Does it provide specific details when needed?

Return only JSON: {{"factual_accuracy": X.X, "completeness": X.X, "relevance": X.X, "specificity": X.X}}"""

            result = llm.generate_answer("Evaluate this answer:", evaluation_prompt)
            
            # Track cost (rough estimate)
            self.openai_cost_tracker["total_calls"] += 1
            self.openai_cost_tracker["spot_check_calls"] += 1
            self.openai_cost_tracker["total_cost"] += 0.002  # Rough estimate
            
            # Try to parse JSON response
            if isinstance(result, dict) and 'content' in result:
                try:
                    scores = json.loads(result['content'])
                    return scores
                except json.JSONDecodeError:
                    logger.warning("Could not parse OpenAI evaluation response")
                    return self._get_default_scores()
            else:
                return self._get_default_scores()
                
        except Exception as e:
            logger.error(f"OpenAI spot-check failed: {e}")
            return self._get_default_scores()
    
    def _load_qa_examples_with_encoding_fix(self) -> List[Dict[str, str]]:
        """Load test questions from all QA JSON files with proper encoding."""
        examples = []
        
        qa_files = [
            ("tests/QAexamples.json", "qa_examples"),
            ("tests/QAcomplex.json", "qa_complex"), 
            ("tests/QAmultisection.json", "qa_multisection")
        ]
        
        for file_path, source_name in qa_files:
            try:
                # Try UTF-8 first, then latin-1 for encoding issues
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            data = json.load(f)
                            break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise Exception("Could not decode file with any encoding")
                
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
    
    def generate_splade_focused_configurations(self) -> List[ComprehensiveTestConfiguration]:
        """Generate SPLADE-focused configurations aligned with original plan."""
        
        # SPLADE-FOCUSED: Embedding models to test (REBUILD INDEX FOR EACH)
        embedding_models = [
            "intfloat/e5-base-v2",              # Current baseline
            "intfloat/e5-large-v2",             # Quality upgrade  
            "intfloat/e5-small-v2",             # Speed option
            "BAAI/bge-base-en-v1.5",           # Strong alternative
            "BAAI/bge-large-en-v1.5",          # Large BGE
            "sentence-transformers/all-MiniLM-L6-v2",  # Fast baseline
            "microsoft/mpnet-base"              # General purpose
        ]
        
        # SPLADE models to test
        splade_models = [
            "naver/splade-cocondenser-ensembledistil",
            "naver/splade-v2-max",
            "naver/splade-v2-distil", 
            "naver/efficient-splade-VI-BT-large-query"
        ]
        
        # Reranker models to test
        reranker_models = [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",   # Current default
            "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Larger version
            "cross-encoder/ms-marco-distilbert-base", # Faster
            "cross-encoder/stsb-roberta-large",       # Different domain
            "cross-encoder/nli-deberta-v3-base"       # Advanced model
        ]
        
        # SPLADE parameter ranges (FULL BRUTE FORCE)
        sparse_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
        expansion_k_values = [50, 100, 150, 200, 300]
        max_sparse_lengths = [128, 256, 512, 1024]
        
        # Retrieval parameters (COMPREHENSIVE)
        similarity_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        top_k_values = [5, 10, 15, 20, 25, 30]
        rerank_top_k_values = [15, 30, 45, 60, 90]
        
        # PIPELINE VARIANTS - This is the key enhancement
        pipeline_variants = [
            ("pure_splade", "SPLADE only"),
            ("reranker_then_splade", "Reranker → SPLADE (TODO: implement)"),
            ("splade_then_reranker", "SPLADE → Reranker (TODO: implement)"),
            ("vector_baseline", "Vector baseline for comparison")
        ]
        
        configurations = []
        config_id = 0
        
        # SPLADE-FOCUSED GENERATION
        for embedding_model in embedding_models:
            for reranker_model in reranker_models:
                for pipeline_type, description in pipeline_variants:
                    for sim_threshold in similarity_thresholds:
                        for top_k in top_k_values:
                            for rerank_top_k in rerank_top_k_values:
                                
                                if pipeline_type in ["pure_splade", "reranker_then_splade", "splade_then_reranker"]:
                                    # SPLADE pipelines: test all SPLADE parameters
                                    for splade_model in splade_models:
                                        for sparse_weight in sparse_weights:
                                            for expansion_k in expansion_k_values:
                                                for max_sparse_length in max_sparse_lengths:
                                                    config_id += 1
                                                    configurations.append(ComprehensiveTestConfiguration(
                                                        embedding_model=embedding_model,
                                                        splade_model=splade_model,
                                                        reranker_model=reranker_model,
                                                        pipeline_type=pipeline_type,
                                                        sparse_weight=sparse_weight,
                                                        expansion_k=expansion_k,
                                                        max_sparse_length=max_sparse_length,
                                                        similarity_threshold=sim_threshold,
                                                        top_k=top_k,
                                                        rerank_top_k=rerank_top_k,
                                                        config_id=f"SPLADE_{config_id:06d}"
                                                    ))
                                else:
                                    # Vector baseline: minimal SPLADE params
                                    config_id += 1
                                    configurations.append(ComprehensiveTestConfiguration(
                                        embedding_model=embedding_model,
                                        splade_model="naver/splade-cocondenser-ensembledistil",
                                        reranker_model=reranker_model,
                                        pipeline_type=pipeline_type,
                                        sparse_weight=0.5,
                                        expansion_k=100,
                                        max_sparse_length=256,
                                        similarity_threshold=sim_threshold,
                                        top_k=top_k,
                                        rerank_top_k=rerank_top_k,
                                        config_id=f"VEC_{config_id:06d}"
                                    ))
        
        logger.info(f"Generated {len(configurations)} SPLADE-FOCUSED test configurations")
        logger.info(f"Embedding models: {len(embedding_models)}")
        logger.info(f"Reranker models: {len(reranker_models)}")
        logger.info(f"Pipeline variants: {len(pipeline_variants)}")
        logger.info(f"Expected runtime: {len(configurations) * len(self.all_test_queries) * 0.5 / 3600:.1f} hours")
        
        return configurations

    def generate_comprehensive_configurations(self) -> List[ComprehensiveTestConfiguration]:
        """Alias for SPLADE-focused configurations (aligns with original plan)."""
        return self.generate_splade_focused_configurations()
    
    def generate_focused_configurations(self) -> List[ComprehensiveTestConfiguration]:
        """Generate a focused set for faster testing while still being comprehensive."""
        
        # Reduced but still comprehensive set
        embedding_models = [
            "intfloat/e5-base-v2",
            "BAAI/bge-base-en-v1.5",
            "intfloat/e5-large-v2"
        ]
        
        splade_models = [
            "naver/splade-cocondenser-ensembledistil",
            "naver/splade-v2-max"
        ]
        
        reranker_models = [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2"
        ]
        
        # Key parameter values
        sparse_weights = [0.3, 0.5, 0.7]
        expansion_k_values = [50, 100, 200]
        max_sparse_lengths = [256, 512]
        similarity_thresholds = [0.15, 0.25, 0.35]
        top_k_values = [10, 15, 20]
        rerank_top_k_values = [30, 45]
        
        # Pipeline configurations aligned with enhanced framework
        pipeline_configs = [
            ("vector_baseline", "Vector baseline"),
            ("pure_splade", "SPLADE only")
        ]
        
        configurations = []
        config_id = 0
        
        for embedding_model in embedding_models:
            for reranker_model in reranker_models:
                for pipeline_type, description in pipeline_configs:
                    for sim_threshold in similarity_thresholds:
                        for top_k in top_k_values:
                            for rerank_top_k in rerank_top_k_values:
                                
                                if pipeline_type == "pure_splade":
                                    # SPLADE pipeline: test all SPLADE parameters
                                    for splade_model in splade_models:
                                        for sparse_weight in sparse_weights:
                                            for expansion_k in expansion_k_values:
                                                for max_sparse_length in max_sparse_lengths:
                                                    config_id += 1
                                                    configurations.append(ComprehensiveTestConfiguration(
                                                        embedding_model=embedding_model,
                                                        splade_model=splade_model,
                                                        reranker_model=reranker_model,
                                                        pipeline_type=pipeline_type,
                                                        sparse_weight=sparse_weight,
                                                        expansion_k=expansion_k,
                                                        max_sparse_length=max_sparse_length,
                                                        similarity_threshold=sim_threshold,
                                                        top_k=top_k,
                                                        rerank_top_k=rerank_top_k,
                                                        config_id=f"FOCUSED_{config_id:06d}"
                                                    ))
                                else:
                                    # Vector baseline: minimal SPLADE params
                                    config_id += 1
                                    configurations.append(ComprehensiveTestConfiguration(
                                                        embedding_model=embedding_model,
                                                        splade_model="naver/splade-cocondenser-ensembledistil",
                                                        reranker_model=reranker_model,
                                                        pipeline_type=pipeline_type,
                                                        sparse_weight=0.5,
                                                        expansion_k=100,
                                                        max_sparse_length=256,
                                                        similarity_threshold=sim_threshold,
                                                        top_k=top_k,
                                                        rerank_top_k=rerank_top_k,
                                                        config_id=f"FOCUSED_{config_id:06d}"
                                                    ))
        
        logger.info(f"Generated {len(configurations)} FOCUSED test configurations")
        return configurations
    
    def setup_configuration(self, config: ComprehensiveTestConfiguration) -> bool:
        """Setup system for specific PIPELINE configuration with proper model switching."""
        
        try:
            # CRITICAL FIX: Track current configuration to detect ANY changes - COMPLETE parameter tracking
            current_config_key = f"{config.embedding_model}|{config.splade_model}|{config.reranker_model}|{config.pipeline_type}|{config.sparse_weight}|{config.expansion_k}|{config.max_sparse_length}|{config.similarity_threshold}|{config.top_k}|{config.rerank_top_k}"
            
            if not hasattr(self, 'last_config_key'):
                self.last_config_key = None
            
            # Check if ANY configuration parameter has changed
            config_changed = (self.last_config_key != current_config_key) or (self.orchestrator is None)
            
            if config_changed:
                logger.info(f"🔄 CONFIGURATION CHANGE DETECTED: {config.config_id}")
                logger.info(f"   Previous: {self.last_config_key}")
                logger.info(f"   Current:  {current_config_key}")
                
                # 1. Handle model and indexing-time parameter changes (require index rebuild)
                # CRITICAL FIX: Properly scope SPLADE parameters - only check when using SPLADE
                
                # Check if current or new config uses SPLADE
                current_uses_splade = self.current_pipeline_type in ["pure_splade", "reranker_then_splade", "splade_then_reranker"] if self.current_pipeline_type else False
                new_uses_splade = config.pipeline_type in ["pure_splade", "reranker_then_splade", "splade_then_reranker"]
                
                # Base indexing parameters (always check these)
                base_params_changed = (
                    self.current_embedding_model != config.embedding_model or
                    self.orchestrator is None
                )
                
                # SPLADE parameters (only check when current OR new config uses SPLADE)
                splade_params_changed = False
                if current_uses_splade or new_uses_splade:
                    splade_params_changed = (
                        self.current_max_sparse_length != config.max_sparse_length or
                        self.current_expansion_k != config.expansion_k or
                        self.current_splade_model != config.splade_model
                    )
                    logger.info(f"   SPLADE parameter check (current_uses={current_uses_splade}, new_uses={new_uses_splade}): {splade_params_changed}")
                else:
                    logger.info(f"   SPLADE parameters skipped (current_uses={current_uses_splade}, new_uses={new_uses_splade})")
                
                # Pipeline type change (switching between SPLADE and non-SPLADE requires rebuild)
                pipeline_change = self.current_pipeline_type != config.pipeline_type
                
                indexing_params_changed = base_params_changed or splade_params_changed or pipeline_change
                
                if indexing_params_changed:
                    logger.info(f"🔄 INDEXING-TIME PARAMETER CHANGE DETECTED:")
                    logger.info(f"   Embedding model: {self.current_embedding_model} → {config.embedding_model}")
                    logger.info(f"   SPLADE model: {self.current_splade_model} → {config.splade_model}")
                    logger.info(f"   Max sparse length: {self.current_max_sparse_length} → {config.max_sparse_length}")
                    logger.info(f"   Expansion K: {self.current_expansion_k} → {config.expansion_k}")
                    logger.info(f"   → INDEX REBUILD REQUIRED")
                    
                    # Backup current index if exists
                    if self.current_embedding_model and self.doc_processor:
                        # Create backup key with indexing parameters
                        backup_key = f"{self.current_embedding_model}_{self.current_splade_model}_{self.current_max_sparse_length}_{self.current_expansion_k}"
                        self._backup_current_index(backup_key)
                    
                    # Initialize new orchestrator with new embedding model  
                    self._initialize_orchestrator(config.embedding_model)
                    
                    # Setup index for this parameter combination
                    index_key = f"{config.embedding_model}_{config.splade_model}_{config.max_sparse_length}_{config.expansion_k}"
                    self._setup_index_for_parameter_combination(index_key, config.embedding_model)
                    
                    # Update all indexing-time parameter tracking
                    self.current_embedding_model = config.embedding_model
                    self.current_splade_model = config.splade_model
                    self.current_max_sparse_length = config.max_sparse_length
                    self.current_expansion_k = config.expansion_k
                    self.current_pipeline_type = config.pipeline_type  # CRITICAL: Track pipeline type
                    config.index_regenerated = True
                
                # 2. Handle reranker model changes
                if (self.current_reranker_model != config.reranker_model):
                    logger.info(f"Switching reranker model: {self.current_reranker_model} → {config.reranker_model}")
                    self._update_reranker_model(config.reranker_model)
                    self.current_reranker_model = config.reranker_model
                
                # Update the configuration tracking
                self.last_config_key = current_config_key
            else:
                logger.info(f"✅ Configuration unchanged: {config.config_id} (skipping reconfiguration)")
            
            # 3. Configure pipeline-specific settings
            if config.pipeline_type == "pure_splade":
                # Pure SPLADE: encoder → splade → LLM
                self.retrieval_system.use_splade = True
                # Disable reranking to ensure pure SPLADE path
                import core.config as config_module
                config_module.performance_config.enable_reranking = False
                config_module.retrieval_config.enhanced_mode = False
                
            elif config.pipeline_type == "reranker_then_splade":
                # TODO: Implement reranker → splade pipeline
                logger.warning("reranker_then_splade pipeline not yet implemented - using pure_splade")
                self.retrieval_system.use_splade = True
                import core.config as config_module
                config_module.performance_config.enable_reranking = False
                config_module.retrieval_config.enhanced_mode = False
                
            elif config.pipeline_type == "splade_then_reranker":
                # TODO: Implement splade → reranker pipeline  
                logger.warning("splade_then_reranker pipeline not yet implemented - using pure_splade")
                self.retrieval_system.use_splade = True
                import core.config as config_module
                config_module.performance_config.enable_reranking = False
                config_module.retrieval_config.enhanced_mode = False
                
            elif config.pipeline_type == "vector_baseline":
                # Vector baseline: encoder → vector search → LLM
                self.retrieval_system.use_splade = False
                import core.config as config_module
                config_module.performance_config.enable_reranking = False
                config_module.retrieval_config.enhanced_mode = False
            
            # 4. Configure SPLADE parameters (when using SPLADE)
            if config.pipeline_type in ["pure_splade", "reranker_then_splade", "splade_then_reranker"]:
                if hasattr(self.retrieval_system, 'splade_engine') and self.retrieval_system.splade_engine:
                    self.retrieval_system.splade_engine.update_config(
                        sparse_weight=config.sparse_weight,
                        expansion_k=config.expansion_k,
                        max_sparse_length=config.max_sparse_length
                    )
            
            # 5. Update retrieval parameters in config
            import core.config as config_module
            config_module.retrieval_config.similarity_threshold = config.similarity_threshold
            
            # 6. Configure reranking top_k parameter
            if hasattr(self.retrieval_system, 'reranking_engine') and self.retrieval_system.reranking_engine:
                # Apply rerank_top_k to reranking engine
                self.retrieval_system.reranking_engine.rerank_top_k = config.rerank_top_k
                logger.info(f"Applied rerank_top_k={config.rerank_top_k} to reranking engine")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up configuration {config.config_id}: {e}")
            return False
    
    def _initialize_orchestrator(self, embedding_model: str):
        """Initialize orchestrator with specific embedding model - COMPLETE ISOLATION."""
        logger.info(f"Switching to embedding model: {embedding_model}")
        
        # 1. COMPLETE INDEX AND CACHE CLEANUP
        self._complete_cleanup()
        
        # 2. Update config to use specific embedding model
        import core.config as config_module
        original_model = config_module.retrieval_config.embedding_model
        config_module.retrieval_config.embedding_model = embedding_model
        
        try:
            # 3. Initialize fresh orchestrator
            logger.info("Complete cleanup done - initializing fresh orchestrator")
            self.orchestrator = AppOrchestrator()
            self.doc_processor, self.llm_service, self.retrieval_system = self.orchestrator.get_services()
            
            # 4. FORCE RELOAD AND REBUILD WITH NEW EMBEDDING MODEL
            self._force_document_reload_and_index_rebuild()
            
            logger.info("Model switched and index rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            # Restore original config
            config_module.retrieval_config.embedding_model = original_model
            raise
    
    def _complete_cleanup(self):
        """BULLETPROOF cleanup - remove ALL traces of previous embedding model data."""
        logger.info("Starting BULLETPROOF cleanup - ALL previous model data")
        
        # 1. Delete ALL index directories completely
        import shutil
        index_dirs = [
            Path("./data/index"),
            Path("./current_index"), 
            Path("./.cache"),
            Path("./data/.cache"),
            Path("./.vector_cache"),  # Additional cache locations
            Path("./sentence_transformers_cache"),
            Path("./transformers_cache")
        ]
        
        for index_dir in index_dirs:
            if index_dir.exists():
                try:
                    shutil.rmtree(index_dir)
                    logger.info(f"Removed index directory: {index_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove {index_dir}: {e}")
        
        # 2. Clear document processor completely using proper methods
        if self.doc_processor:
            # Use the proper clear_index method instead of direct assignment
            self.doc_processor.clear_index()
            # Clear ingestion cache
            if hasattr(self.doc_processor, 'ingestion') and self.doc_processor.ingestion:
                self.doc_processor.ingestion.clear_cache()
                self.doc_processor.ingestion.processed_files = set()
        
        # 3. Clear retrieval system engines
        if self.retrieval_system:
            if hasattr(self.retrieval_system, 'vector_engine') and self.retrieval_system.vector_engine:
                if hasattr(self.retrieval_system.vector_engine, 'clear_cache'):
                    self.retrieval_system.vector_engine.clear_cache()
                # Force nullify
                self.retrieval_system.vector_engine = None
            
            if hasattr(self.retrieval_system, 'splade_engine') and self.retrieval_system.splade_engine:
                if hasattr(self.retrieval_system.splade_engine, 'clear_cache'):
                    self.retrieval_system.splade_engine.clear_cache()
                # Force nullify  
                self.retrieval_system.splade_engine = None
                
            if hasattr(self.retrieval_system, 'reranking_engine') and self.retrieval_system.reranking_engine:
                self.retrieval_system.reranking_engine = None
        
        # 4. DEEP MODEL CACHE CLEANUP
        try:
            # Clear sentence-transformers cache
            import sentence_transformers
            if hasattr(sentence_transformers, '_model_cache'):
                sentence_transformers._model_cache.clear()
        except:
            pass
            
        try:
            # Clear transformers cache  
            import transformers
            if hasattr(transformers, 'trainer_utils'):
                if hasattr(transformers.trainer_utils, '_model_cache'):
                    transformers.trainer_utils._model_cache.clear()
        except:
            pass
            
        try:
            # Clear FAISS memory if available
            import faiss
            if hasattr(faiss, 'omp_set_num_threads'):
                faiss.omp_set_num_threads(1)  # Reset threading
        except:
            pass
            
        try:
            # Clear GPU memory if using CUDA
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        
        # 5. Nullify ALL service references  
        self.doc_processor = None
        self.llm_service = None
        self.retrieval_system = None
        self.orchestrator = None
        
        # 6. Force aggressive garbage collection
        import gc
        gc.collect()
        gc.collect()  # Run twice for thoroughness
        
        logger.info("BULLETPROOF cleanup completed - all traces removed")
    
    def _force_document_reload_and_index_rebuild(self):
        """Force complete reload of KTI Dispatch Guide and rebuild index."""
        logger.info("Force reloading KTI_Dispatch_Guide.pdf...")
        
        # 1. Ensure the source document exists
        source_doc = Path("KTI_Dispatch_Guide.pdf")
        if not source_doc.exists():
            raise FileNotFoundError(f"Source document not found: {source_doc}")
        
        # 2. Force document processing with current embedding model
        try:
            # Use the proper process_documents method
            index, chunks = self.doc_processor.process_documents([str(source_doc)])
            
            # Verify the index was built
            if chunks and len(chunks) > 0:
                logger.info(f"Document reloaded - {len(chunks)} chunks indexed")
            else:
                raise RuntimeError("Document reload failed - no chunks indexed")
                
        except Exception as e:
            logger.error(f"Failed to reload document: {e}")
            raise
    
    def _backup_current_index(self, embedding_model: str):
        """Backup current index for later reuse."""
        try:
            source_dir = Path("./data/index")
            backup_dir = self.index_backup_dir / embedding_model.replace("/", "_")
            
            if source_dir.exists() and any(source_dir.iterdir()):
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.copytree(source_dir, backup_dir)
                logger.info(f"Backed up index for {embedding_model}")
        except Exception as e:
            logger.warning(f"Failed to backup index for {embedding_model}: {e}")
    
    def _setup_index_for_embedding_model(self, embedding_model: str):
        """Setup index for specific embedding model."""
        backup_dir = self.index_backup_dir / embedding_model.replace("/", "_")
        index_dir = Path("./data/index")
        
        # Try to restore from backup first
        if backup_dir.exists() and any(backup_dir.iterdir()):
            try:
                if index_dir.exists():
                    shutil.rmtree(index_dir)
                shutil.copytree(backup_dir, index_dir)
                logger.info(f"Restored index from backup for {embedding_model}")
                return
            except Exception as e:
                logger.warning(f"Failed to restore backup for {embedding_model}: {e}")
        
        # Build new index if no backup
        logger.info(f"Building new index for {embedding_model}")
        # The orchestrator should handle this automatically
    
    def _setup_index_for_parameter_combination(self, index_key: str, embedding_model: str):
        """Setup index for specific parameter combination with indexing-time parameters.
        
        Args:
            index_key: Combined key like "model_splade_maxlen_expansionk"
            embedding_model: Base embedding model for fallback
        """
        backup_dir = self.index_backup_dir / index_key.replace("/", "_")
        index_dir = Path("./data/index")
        
        # Try to restore from parameter-specific backup first
        if backup_dir.exists() and any(backup_dir.iterdir()):
            try:
                if index_dir.exists():
                    shutil.rmtree(index_dir)
                shutil.copytree(backup_dir, index_dir)
                logger.info(f"Restored index from parameter backup: {index_key}")
                return
            except Exception as e:
                logger.warning(f"Failed to restore parameter backup {index_key}: {e}")
        
        # Fallback to embedding model backup
        embedding_backup_dir = self.index_backup_dir / embedding_model.replace("/", "_")
        if embedding_backup_dir.exists() and any(embedding_backup_dir.iterdir()):
            try:
                if index_dir.exists():
                    shutil.rmtree(index_dir)
                shutil.copytree(embedding_backup_dir, index_dir)
                logger.info(f"Restored from embedding backup: {embedding_model}")
                return
            except Exception as e:
                logger.warning(f"Failed to restore embedding backup {embedding_model}: {e}")
        
        # Build completely new index if no backups
        logger.info(f"Building new index for parameter combination: {index_key}")
        # The orchestrator will handle this automatically during initialization
    
    def _update_reranker_model(self, reranker_model: str):
        """Update reranker model in the system."""
        try:
            if hasattr(self.retrieval_system, 'reranking_engine'):
                reranking_engine = self.retrieval_system.reranking_engine
                if reranking_engine:
                    # Force reload with new model
                    reranking_engine._reranker_model = reranker_model
                    reranking_engine._cross_encoder = None  # Force reload
                    logger.info(f"Updated reranker model to {reranker_model}")
        except Exception as e:
            logger.error(f"Failed to update reranker model: {e}")
    
    def evaluate_query(self, query_data: Dict[str, str], config: ComprehensiveTestConfiguration) -> ComprehensiveQueryResult:
        """Evaluate a single query with current configuration."""
        
        question = query_data["question"]
        expected_answer = query_data.get("expected_answer", "")
        source = query_data.get("source", "unknown")
        feedback_type = query_data.get("feedback_type", "neutral")
        feedback_text = query_data.get("feedback_text", "")
        
        # CRITICAL: Validate question format for SPLADE
        if not isinstance(question, str) or not question.strip():
            logger.error(f"Invalid question format: {type(question)} - '{question}'")
            question = str(question).strip() if question else "default question"
        
        logger.info(f"🔍 EVALUATING: '{question[:50]}...' [Expected: '{expected_answer[:30]}...']")
        
        start_time = time.time()
        
        try:
            # CRITICAL FIX: Ensure retrieval system is properly initialized
            if self.retrieval_system is None:
                logger.error("Retrieval system is None - attempting to reinitialize")
                if self.orchestrator is None:
                    logger.error("Orchestrator is also None - cannot proceed")
                    raise RuntimeError("Both orchestrator and retrieval_system are None")
                
                # Attempt to get services again
                self.doc_processor, self.llm_service, self.retrieval_system = self.orchestrator.get_services()
                
                if self.retrieval_system is None:
                    raise RuntimeError("Failed to reinitialize retrieval_system")
                    
                logger.info("Successfully reinitialized retrieval_system")
            
            # Perform retrieval with input validation
            retrieval_start = time.time()
            logger.info(f"🚀 RETRIEVAL START: Pipeline={config.pipeline_type}, Question='{question[:50]}...'")
            
            # Use appropriate retrieval method based on pipeline type
            if config.pipeline_type == "pure_splade":
                logger.info("Using PURE SPLADE pipeline")
                context, retrieval_time, chunk_count, similarity_scores = self.retrieval_system.retrieve(
                    question, top_k=config.top_k
                )
            elif config.pipeline_type in ["reranker_then_splade", "splade_then_reranker"]:
                logger.info(f"Using COMBINED pipeline: {config.pipeline_type}")
                context, retrieval_time, chunk_count, similarity_scores = self.retrieval_system.retrieve(
                    question, top_k=config.top_k
                )
            elif config.pipeline_type == "vector_baseline":
                logger.info("Using VECTOR BASELINE pipeline")
                context, retrieval_time, chunk_count, similarity_scores = self.retrieval_system.retrieve(
                    question, top_k=config.top_k
                )
            else:
                logger.info(f"Using DEFAULT pipeline: {config.pipeline_type}")
                context, retrieval_time, chunk_count, similarity_scores = self.retrieval_system.retrieve(
                    question, top_k=config.top_k
                )
            
            retrieval_duration = time.time() - retrieval_start
            logger.info(f"✅ RETRIEVAL COMPLETE: {chunk_count} chunks, {len(context)} chars, {retrieval_duration:.3f}s")
            
            # Generate answer with COST-EFFECTIVE Ollama service
            answer_start = time.time()
            logger.info(f"🤖 COST-EFFECTIVE GENERATION START: Using Ollama qwen2.5:14b-instruct")
            
            # Use cost-effective Ollama generation
            answer_result = self._generate_answer_cost_effective(question, context)
            
            # Extract content from the returned dictionary
            if isinstance(answer_result, dict) and 'content' in answer_result:
                answer = answer_result['content']
                logger.info(f"✅ OLLAMA ANSWER GENERATED: '{answer[:100]}...' (Cost: ${answer_result.get('cost', 0):.4f})")
            else:
                logger.error(f"❌ UNEXPECTED ANSWER FORMAT: {type(answer_result)} - {str(answer_result)[:200]}...")
                answer = f"ERROR: Unexpected answer format from cost-effective service: {type(answer_result)}"
            
            answer_duration = time.time() - answer_start
            total_duration = time.time() - start_time
            
            logger.info(f"⏱️  TIMING: Retrieval={retrieval_duration:.3f}s, Answer={answer_duration:.3f}s, Total={total_duration:.3f}s")
            
            # Calculate metrics
            max_similarity = max(similarity_scores) if similarity_scores else 0.0
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            
            # Quality assessment with detailed logging
            logger.info(f"🔍 QUALITY ASSESSMENT START:")
            logger.info(f"   Expected Answer: '{expected_answer[:100]}...'")
            logger.info(f"   Generated Answer: '{answer[:100]}...'")
            logger.info(f"   Context Length: {len(context)} chars")
            
            context_contains_answer = self._assess_context_contains_answer(context, expected_answer)
            answer_correctness = self._assess_answer_correctness(answer, expected_answer)
            completeness_score = self._assess_completeness(answer, question)
            specificity_score = self._assess_specificity(answer)
            
            logger.info(f"📊 ASSESSMENT RESULTS:")
            logger.info(f"   Context Contains Answer: {context_contains_answer}")
            logger.info(f"   Answer Correctness: {answer_correctness:.3f}")
            logger.info(f"   Completeness Score: {completeness_score:.3f}")
            logger.info(f"   Specificity Score: {specificity_score:.3f}")
            logger.info(f"   Similarity Stats: max={max_similarity:.3f}, avg={avg_similarity:.3f}")
            
            return ComprehensiveQueryResult(
                question=question,
                expected_answer=expected_answer,
                source=source,
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
                feedback_text=feedback_text,
                error_occurred=False
            )
            
        except Exception as e:
            logger.error(f"Error evaluating query '{question[:50]}...': {e}")
            
            return ComprehensiveQueryResult(
                question=question,
                expected_answer=expected_answer,
                source=source,
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
                feedback_text=feedback_text,
                error_occurred=True,
                error_message=str(e)
            )
    
    def _assess_context_contains_answer(self, context: str, expected_answer: str) -> bool:
        """Assess if context contains answer information."""
        if not expected_answer or not context:
            return False
        
        try:
            # CRITICAL TYPE SAFETY: Ensure strings before string operations
            if not isinstance(expected_answer, str) or not isinstance(context, str):
                logger.warning(f"Non-string inputs to context assessment: expected_answer={type(expected_answer)}, context={type(context)}")
                expected_answer = str(expected_answer) if expected_answer else ""
                context = str(context) if context else ""
            
            expected_words = set(expected_answer.lower().split())
            context_words = set(context.lower().split())
            
            if len(expected_words) == 0:
                return False
                
            overlap = len(expected_words.intersection(context_words))
            return overlap >= len(expected_words) * 0.3
        except Exception as e:
            logger.error(f"Error in context assessment: {e}")
            return False
    
    def _assess_answer_correctness(self, answer: str, expected_answer: str) -> float:
        """Assess answer correctness."""
        if not expected_answer or not answer or answer == "ERROR":
            return 0.0
        
        try:
            expected_words = set(expected_answer.lower().split())
            answer_words = set(answer.lower().split())
            
            if len(expected_words) == 0:
                return 1.0
            
            overlap = len(expected_words.intersection(answer_words))
            return overlap / len(expected_words) if len(expected_words) > 0 else 0.0
        except Exception as e:
            logger.error(f"Error in answer correctness assessment: {e}")
            return 0.0
    
    def _assess_completeness(self, answer: str, question: str) -> float:
        """Assess answer completeness with adjusted thresholds for comprehensive LLM responses."""
        if not answer or answer == "ERROR":
            return 0.0
        
        if len(answer) < 100:     # Very brief
            return 0.3
        elif len(answer) < 250:   # Adequate  
            return 0.7
        else:                     # Comprehensive (250+ chars)
            return 1.0
    
    def _assess_specificity(self, answer: str) -> float:
        """Assess answer specificity."""
        if not answer or answer == "ERROR":
            return 0.0
        
        specific_indicators = ["$", "phone", "number", "address", "time", "date", "percent", "%"]
        specificity_score = sum(1 for indicator in specific_indicators if indicator in answer.lower())
        return min(specificity_score / 3.0, 1.0)
    
    def run_configuration_test(self, config: ComprehensiveTestConfiguration) -> ComprehensiveTestResults:
        """Run complete test for a configuration."""
        
        config_start_time = time.time()
        
        logger.info(f"=== Testing Configuration {config.config_id} ===")
        logger.info(f"  Embedding: {config.embedding_model}")
        logger.info(f"  Pipeline: {config.pipeline_type}")
        logger.info(f"  Reranker: {config.reranker_model}")
        if config.pipeline_type in ["pure_splade", "reranker_then_splade", "splade_then_reranker"]:
            logger.info(f"  SPLADE Model: {config.splade_model}")
            logger.info(f"  Sparse Weight: {config.sparse_weight}")
            logger.info(f"  Expansion K: {config.expansion_k}")
        
        # Setup configuration - CRITICAL BUG FIX
        if not self.setup_configuration(config):
            logger.error(f"Failed to setup configuration {config.config_id}")
            # Return empty result instead of None to prevent crashes
            return ComprehensiveTestResults(
                config=config,
                query_results=[],
                avg_retrieval_time=0.0,
                avg_answer_time=0.0,
                avg_total_time=0.0,
                avg_similarity=0.0,
                context_hit_rate=0.0,
                avg_correctness=0.0,
                avg_completeness=0.0,
                avg_specificity=0.0,
                positive_feedback_rate=0.0,
                negative_feedback_rate=0.0,
                neutral_feedback_rate=0.0,
                total_test_time=0.0,
                errors_encountered=1,
                error_details=[f"Configuration setup failed for {config.config_id}"],
                success_rate=0.0,
                models_used={"error": "setup_failed"}
            )
        
        # Run all queries
        query_results = []
        errors = []
        
        for i, query_data in enumerate(self.all_test_queries):
            try:
                result = self.evaluate_query(query_data, config)
                query_results.append(result)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"  Completed {i + 1}/{len(self.all_test_queries)} queries")
                    
            except Exception as e:
                error_msg = f"Error on query {i}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        total_test_time = time.time() - config_start_time
        
        # Calculate aggregate metrics
        if query_results:
            successful_results = [r for r in query_results if not r.error_occurred]
            
            if successful_results:
                avg_retrieval_time = sum(r.retrieval_time for r in successful_results) / len(successful_results)
                avg_answer_time = sum(r.answer_time for r in successful_results) / len(successful_results)
                avg_total_time = sum(r.total_time for r in successful_results) / len(successful_results)
                avg_similarity = sum(r.avg_similarity for r in successful_results) / len(successful_results)
                
                context_hit_rate = sum(1 for r in successful_results if r.context_contains_answer) / len(successful_results)
                avg_correctness = sum(r.answer_correctness_score for r in successful_results) / len(successful_results)
                avg_completeness = sum(r.completeness_score for r in successful_results) / len(successful_results)
                avg_specificity = sum(r.specificity_score for r in successful_results) / len(successful_results)
                
                # Feedback analysis
                positive_feedback = sum(1 for r in successful_results if r.feedback_type == "positive")
                negative_feedback = sum(1 for r in successful_results if r.feedback_type == "negative")
                neutral_feedback = sum(1 for r in successful_results if r.feedback_type == "neutral")
                
                total_feedback = len(successful_results)
                positive_feedback_rate = positive_feedback / total_feedback if total_feedback > 0 else 0.0
                negative_feedback_rate = negative_feedback / total_feedback if total_feedback > 0 else 0.0
                neutral_feedback_rate = neutral_feedback / total_feedback if total_feedback > 0 else 0.0
                
                success_rate = len(successful_results) / len(query_results)
            else:
                # All queries failed
                avg_retrieval_time = avg_answer_time = avg_total_time = avg_similarity = 0.0
                context_hit_rate = avg_correctness = avg_completeness = avg_specificity = 0.0
                positive_feedback_rate = negative_feedback_rate = neutral_feedback_rate = 0.0
                success_rate = 0.0
        else:
            # No results at all
            avg_retrieval_time = avg_answer_time = avg_total_time = avg_similarity = 0.0
            context_hit_rate = avg_correctness = avg_completeness = avg_specificity = 0.0
            positive_feedback_rate = negative_feedback_rate = neutral_feedback_rate = 0.0
            success_rate = 0.0
        
        # Track models used
        models_used = {
            "embedding_model": config.embedding_model,
            "splade_model": config.splade_model if config.pipeline_type in ["pure_splade", "reranker_then_splade", "splade_then_reranker"] else "none",
            "reranker_model": config.reranker_model,
            "pipeline_type": config.pipeline_type
        }
        
        logger.info(f"Configuration {config.config_id} completed:")
        logger.info(f"  Success rate: {success_rate:.3f}")
        logger.info(f"  Context hit rate: {context_hit_rate:.3f}")
        logger.info(f"  Avg correctness: {avg_correctness:.3f}")
        logger.info(f"  Avg retrieval time: {avg_retrieval_time:.3f}s")
        
        return ComprehensiveTestResults(
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
            neutral_feedback_rate=neutral_feedback_rate,
            total_test_time=total_test_time,
            errors_encountered=len(errors),
            error_details=errors,
            success_rate=success_rate,
            models_used=models_used
        )
    
    def run_comprehensive_evaluation(self, configurations: List[ComprehensiveTestConfiguration]) -> List[ComprehensiveTestResults]:
        """Run comprehensive evaluation across all configurations."""
        
        self.total_configs = len(configurations)
        self.start_time = time.time()
        self.completed_configs = 0
        
        logger.info(f"=== STARTING COMPREHENSIVE SYSTEMATIC EVALUATION ===")
        logger.info(f"Total configurations: {self.total_configs}")
        logger.info(f"Questions per config: {len(self.all_test_queries)}")
        logger.info(f"Total evaluations: {self.total_configs * len(self.all_test_queries)}")
        
        results = []
        
        for i, config in enumerate(configurations):
            try:
                # Progress tracking
                elapsed_time = time.time() - self.start_time
                if self.completed_configs > 0:
                    avg_time_per_config = elapsed_time / max(self.completed_configs, 1)
                    eta_seconds = avg_time_per_config * (self.total_configs - self.completed_configs)
                    eta_hours = eta_seconds / 3600
                    
                    logger.info(f"=== PROGRESS: {self.completed_configs}/{self.total_configs} ({(self.completed_configs/self.total_configs)*100:.1f}%) ===")
                    logger.info(f"Elapsed: {elapsed_time/3600:.1f}h, ETA: {eta_hours:.1f}h")
                
                # Run configuration test
                result = self.run_configuration_test(config)
                if result:
                    results.append(result)
                
                self.completed_configs += 1
                
                # Save checkpoint every 10 configurations
                if self.completed_configs % 10 == 0:
                    self.save_checkpoint_results(results, self.completed_configs)
                
            except Exception as e:
                logger.error(f"Critical error testing configuration {config.config_id}: {e}")
                continue
        
        logger.info(f"=== COMPREHENSIVE EVALUATION COMPLETE ===")
        logger.info(f"Configurations tested: {len(results)}")
        logger.info(f"Total time: {(time.time() - self.start_time)/3600:.1f} hours")
        
        return results
    
    def save_checkpoint_results(self, results: List[ComprehensiveTestResults], checkpoint_num: int):
        """Save intermediate results as checkpoint with comprehensive logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint_file = self.test_data_dir / f"comprehensive_checkpoint_{checkpoint_num}_{timestamp}.json"
        
        try:
            # Prepare data for JSON serialization
            checkpoint_data = {
                "checkpoint_info": {
                    "configs_completed": checkpoint_num,
                    "total_configs": self.total_configs,
                    "timestamp": timestamp,
                    "elapsed_hours": (time.time() - self.start_time) / 3600,
                    "completion_percentage": (checkpoint_num / self.total_configs) * 100,
                    "estimated_remaining_hours": ((time.time() - self.start_time) / checkpoint_num) * (self.total_configs - checkpoint_num) / 3600 if checkpoint_num > 0 else 0
                },
                "results": [asdict(result) for result in results]
            }
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Generate progress summary markdown
            self._generate_progress_summary(checkpoint_data, checkpoint_num)
            
            # Generate intermediate analysis
            self._generate_intermediate_analysis(results, checkpoint_num)
            
            # Track model switching efficiency
            self._log_model_transitions(checkpoint_num)
            
            logger.info(f"Checkpoint {checkpoint_num} saved with comprehensive logs")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _generate_progress_summary(self, checkpoint_data: Dict, checkpoint_num: int):
        """Generate human-readable progress summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        progress_file = self.test_data_dir / f"progress_summary_{timestamp}.md"
        
        info = checkpoint_data["checkpoint_info"]
        
        progress_md = f"""# Comprehensive SPLADE Testing Progress Report
        
**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Overall Progress
- **Configurations Completed**: {info['configs_completed']:,} / {info['total_configs']:,}
- **Completion Percentage**: {info['completion_percentage']:.1f}%
- **Elapsed Time**: {info['elapsed_hours']:.1f} hours
- **Estimated Remaining**: {info['estimated_remaining_hours']:.1f} hours
- **Estimated Total Runtime**: {info['elapsed_hours'] + info['estimated_remaining_hours']:.1f} hours

## 🏃‍♂️ Performance Summary
- **Average Time per Configuration**: {(info['elapsed_hours'] * 3600) / info['configs_completed']:.1f} seconds
- **Configurations per Hour**: {info['configs_completed'] / info['elapsed_hours']:.1f}

## 📈 Quality Overview
"""
        
        # Add quality metrics if we have results
        if checkpoint_data["results"]:
            results = checkpoint_data["results"]
            avg_hit_rate = sum(r['context_hit_rate'] for r in results) / len(results)
            avg_correctness = sum(r['avg_correctness'] for r in results) / len(results)
            avg_success_rate = sum(r['success_rate'] for r in results) / len(results)
            
            progress_md += f"""
- **Average Context Hit Rate**: {avg_hit_rate:.3f}
- **Average Correctness**: {avg_correctness:.3f}  
- **Average Success Rate**: {avg_success_rate:.3f}

## 🔬 Configuration Breakdown
- **Embedding Models Tested**: {len(set(r['models_used']['embedding_model'] for r in results))}
- **Pipeline Types Tested**: {len(set(r['config']['pipeline_type'] for r in results))}
- **Total Queries Processed**: {sum(len(r['query_results']) for r in results):,}

## ⚡ Next Milestone
- **Next Checkpoint**: Configuration {checkpoint_num + 10}
- **Expected Time**: {info['estimated_remaining_hours'] * (10 / (info['total_configs'] - info['configs_completed'])):.1f} hours

---
*This is an automated progress report. Check `comprehensive_systematic_evaluation.log` for detailed execution logs.*
"""
        
        with open(progress_file, 'w') as f:
            f.write(progress_md)
    
    def _generate_intermediate_analysis(self, results: List[ComprehensiveTestResults], checkpoint_num: int):
        """Generate intermediate analysis of results so far."""
        if not results:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = self.test_data_dir / f"intermediate_analysis_{checkpoint_num}_{timestamp}.json"
        
        # Analyze results so far
        summary_data = []
        for result in results:
            summary = {
                "config_id": result.config.config_id,
                "embedding_model": result.config.embedding_model,
                "pipeline_type": result.config.pipeline_type,
                "context_hit_rate": result.context_hit_rate,
                "avg_correctness": result.avg_correctness,
                "avg_retrieval_time": result.avg_retrieval_time,
                "success_rate": result.success_rate
            }
            summary_data.append(summary)
        
        # Find current best configurations - QUALITY FIRST
        best_overall = max(summary_data, key=lambda x: (x['context_hit_rate'] * 0.4) + (x['avg_correctness'] * 0.4) + (x['success_rate'] * 0.2))
        highest_quality = max(summary_data, key=lambda x: (x['context_hit_rate'] + x['avg_correctness']) / 2)
        most_accurate = max(summary_data, key=lambda x: x['avg_correctness'])
        most_reliable = max(summary_data, key=lambda x: x['success_rate'])
        
        analysis = {
            "checkpoint_info": {
                "checkpoint": checkpoint_num,
                "configs_analyzed": len(results),
                "timestamp": timestamp
            },
            "current_leaders": {
                "best_overall": best_overall,
                "highest_quality": highest_quality,
                "most_accurate": most_accurate,
                "most_reliable": most_reliable
            },
            "summary_stats": {
                "avg_context_hit_rate": sum(s['context_hit_rate'] for s in summary_data) / len(summary_data) if summary_data else 0.0,
                "avg_correctness": sum(s['avg_correctness'] for s in summary_data) / len(summary_data) if summary_data else 0.0,
                "avg_retrieval_time": sum(s['avg_retrieval_time'] for s in summary_data) / len(summary_data) if summary_data else 0.0,
                "avg_success_rate": sum(s['success_rate'] for s in summary_data) / len(summary_data) if summary_data else 0.0
            }
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def _log_model_transitions(self, checkpoint_num: int):
        """Log model transition efficiency."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transition_file = self.test_data_dir / f"model_transitions_{timestamp}.json"
        
        transition_log = {
            "checkpoint": checkpoint_num,
            "current_embedding_model": self.current_embedding_model,
            "current_reranker_model": self.current_reranker_model,
            "timestamp": timestamp,
            "elapsed_hours": (time.time() - self.start_time) / 3600 if self.start_time else 0
        }
        
        # Append to existing log or create new
        if transition_file.exists():
            with open(transition_file, 'r') as f:
                existing_log = json.load(f)
            existing_log.append(transition_log)
        else:
            existing_log = [transition_log]
        
        with open(transition_file, 'w') as f:
            json.dump(existing_log, f, indent=2)
    
    def save_comprehensive_results(self, results: List[ComprehensiveTestResults], filename_prefix: str = "comprehensive_systematic"):
        """Save complete comprehensive results."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = self.test_data_dir / f"{filename_prefix}_detailed_{timestamp}.json"
        summary_file = self.test_data_dir / f"{filename_prefix}_summary_{timestamp}.json"
        csv_file = self.test_data_dir / f"{filename_prefix}_summary_{timestamp}.csv"
        
        # Prepare detailed data
        detailed_data = {
            "evaluation_info": {
                "total_configurations": len(results),
                "total_queries_per_config": len(self.all_test_queries),
                "total_evaluations": len(results) * len(self.all_test_queries),
                "start_time": self.start_time,
                "end_time": time.time(),
                "total_hours": (time.time() - self.start_time) / 3600,
                "timestamp": timestamp
            },
            "results": [asdict(result) for result in results]
        }
        
        # Save detailed results
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        
        # Create summary data for analysis
        summary_data = []
        for result in results:
            summary = {
                "config_id": result.config.config_id,
                "embedding_model": result.config.embedding_model,
                "pipeline_type": result.config.pipeline_type,
                "reranker_model": result.config.reranker_model,
                "splade_model": result.config.splade_model,
                "sparse_weight": result.config.sparse_weight,
                "expansion_k": result.config.expansion_k,
                "max_sparse_length": result.config.max_sparse_length,
                "similarity_threshold": result.config.similarity_threshold,
                "top_k": result.config.top_k,
                "rerank_top_k": result.config.rerank_top_k,
                "index_regenerated": result.config.index_regenerated,
                
                # Performance metrics
                "avg_retrieval_time": result.avg_retrieval_time,
                "avg_answer_time": result.avg_answer_time,
                "avg_total_time": result.avg_total_time,
                "avg_similarity": result.avg_similarity,
                
                # Quality metrics
                "context_hit_rate": result.context_hit_rate,
                "avg_correctness": result.avg_correctness,
                "avg_completeness": result.avg_completeness,
                "avg_specificity": result.avg_specificity,
                
                # Feedback metrics
                "positive_feedback_rate": result.positive_feedback_rate,
                "negative_feedback_rate": result.negative_feedback_rate,
                "neutral_feedback_rate": result.neutral_feedback_rate,
                
                # Reliability metrics
                "success_rate": result.success_rate,
                "errors_encountered": result.errors_encountered,
                "total_queries": len(result.query_results),
                "total_test_time": result.total_test_time
            }
            summary_data.append(summary)
        
        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Save CSV for easy analysis
        if summary_data:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
                writer.writeheader()
                writer.writerows(summary_data)
        
        logger.info(f"Comprehensive results saved:")
        logger.info(f"  Detailed: {detailed_file}")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  CSV: {csv_file}")
        
        return detailed_file, summary_file, csv_file

def run_focused_comprehensive_evaluation():
    """Run focused comprehensive evaluation (faster but still thorough)."""
    
    print("=== FOCUSED COMPREHENSIVE SYSTEMATIC EVALUATION ===")
    print("This tests hundreds of configurations systematically")
    print("Expected runtime: 6-12 hours")
    print()
    
    evaluator = ComprehensiveSystematicEvaluator()
    configs = evaluator.generate_focused_configurations()
    
    print(f"Generated {len(configs)} focused configurations")
    print(f"Testing with {len(evaluator.all_test_queries)} questions per config")
    print(f"Total evaluations: {len(configs) * len(evaluator.all_test_queries)}")
    
    # Auto-confirm if running in non-interactive mode (background/nohup)
    import sys
    if not sys.stdin.isatty():
        print("Running in non-interactive mode - auto-confirming...")
        confirm = 'y'
    else:
        # Confirm before proceeding
        confirm = input("Continue with focused evaluation? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Evaluation cancelled.")
            return
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(configs)
    
    # Save results
    if results:
        evaluator.save_comprehensive_results(results, "focused_comprehensive")
        print(f"\n=== FOCUSED EVALUATION COMPLETE ===")
        print(f"Tested {len(results)} configurations")
        print(f"Results saved in test_logs/")

def run_full_comprehensive_evaluation():
    """Run FULL comprehensive evaluation (extremely thorough)."""
    
    print("=== FULL COMPREHENSIVE SYSTEMATIC EVALUATION ===")
    print("⚠️  WARNING: This will test THOUSANDS of configurations!")
    print("Expected runtime: 24-48 hours")
    print()
    
    evaluator = ComprehensiveSystematicEvaluator()
    configs = evaluator.generate_comprehensive_configurations()
    
    print(f"Generated {len(configs)} comprehensive configurations")
    print(f"Testing with {len(evaluator.all_test_queries)} questions per config")
    print(f"Total evaluations: {len(configs) * len(evaluator.all_test_queries)}")
    
    # Confirm before proceeding
    print("\n⚠️  This will take 1-2 DAYS to complete!")
    confirm = input("Are you SURE you want to continue? (yes/NO): ").strip().lower()
    if confirm != 'yes':
        print("Evaluation cancelled.")
        return
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(configs)
    
    # Save results
    if results:
        evaluator.save_comprehensive_results(results, "full_comprehensive")
        print(f"\n=== FULL EVALUATION COMPLETE ===")
        print(f"Tested {len(results)} configurations")
        print(f"Results saved in test_logs/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Systematic Engine Evaluation")
    parser.add_argument("--mode", choices=["focused", "full"], default="focused",
                       help="Evaluation mode: focused (hundreds of configs) or full (thousands)")
    
    args = parser.parse_args()
    
    # Create test logs directory
    Path("test_logs").mkdir(exist_ok=True)
    
    if args.mode == "focused":
        run_focused_comprehensive_evaluation()
    else:
        run_full_comprehensive_evaluation()
