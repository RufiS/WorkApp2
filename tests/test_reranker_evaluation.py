"""
Comprehensive Reranker Model Evaluation Suite for WorkApp2
Tests 10+ reranker models for dispatch domain Q&A performance
Works with optimized e5-large-v2 embedding model
"""

import os
import sys
import json
import time
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback
from typing import List, Dict, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class RerankerEvaluator:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "embedding_model": "intfloat/e5-large-v2",
            "rerankers_tested": [],
            "evaluation_summary": {},
            "recommendations": {}
        }
        
        # Dispatch-specific queries for reranker evaluation
        self.dispatch_queries = [
            {
                "query": "A client is calling about computer repair pricing and wants to know what we can fix",
                "category": "pricing_services",
                "expected_topics": ["pricing", "computer repair", "services", "troubleshooting", "diagnosis"]
            },
            {
                "query": "How do I handle a client who wants to cancel their appointment today?",
                "category": "appointment_management",
                "expected_topics": ["cancellation", "appointment", "scheduling", "same day", "policy"]
            },
            {
                "query": "What's the complete process when a Field Engineer calls out sick?",
                "category": "staff_management",
                "expected_topics": ["field engineer", "sick leave", "replacement", "scheduling", "coverage"]
            },
            {
                "query": "A client says their computer is still having the same problem after our Field Engineer visited",
                "category": "followup_issues",
                "expected_topics": ["revisit", "unresolved", "field engineer", "follow-up", "troubleshooting"]
            },
            {
                "query": "How do I help a client who submitted an appointment request online?",
                "category": "online_requests",
                "expected_topics": ["online booking", "appointment", "scheduling", "confirmation", "process"]
            },
            {
                "query": "Client needs emergency computer repair for their business",
                "category": "emergency_service",
                "expected_topics": ["emergency", "business", "priority", "urgent", "escalation"]
            },
            {
                "query": "How do I respond to a text message from a Field Engineer?",
                "category": "communication",
                "expected_topics": ["text message", "SMS", "field engineer", "communication", "response"]
            },
            {
                "query": "What should I tell a client about RingCentral phone system setup?",
                "category": "technical_support",
                "expected_topics": ["RingCentral", "phone system", "setup", "installation", "configuration"]
            }
        ]
        
        # Reranker models to test
        self.rerankers_to_test = [
            # Tier 1: High Priority
            {
                "name": "bge-reranker-large",
                "model_path": "BAAI/bge-reranker-large",
                "category": "BGE Family",
                "expected_size_mb": 560,
                "priority": "high",
                "description": "Best potential performance, designed for modern embeddings"
            },
            {
                "name": "bge-reranker-base",
                "model_path": "BAAI/bge-reranker-base",
                "category": "BGE Family",
                "expected_size_mb": 280,
                "priority": "high",
                "description": "Performance/speed balance, proven compatibility"
            },
            {
                "name": "ms-marco-MiniLM-L-12-v2",
                "model_path": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                "category": "MS Marco MiniLM",
                "expected_size_mb": 120,
                "priority": "high",
                "description": "Upgrade from current L-6, more layers for better understanding"
            },
            {
                "name": "ms-marco-electra-base",
                "model_path": "cross-encoder/ms-marco-electra-base",
                "category": "ELECTRA Architecture",
                "expected_size_mb": 420,
                "priority": "high",
                "description": "ELECTRA architecture, excellent for understanding"
            },
            {
                "name": "stsb-roberta-large",
                "model_path": "cross-encoder/stsb-roberta-large",
                "category": "RoBERTa Large",
                "expected_size_mb": 1200,
                "priority": "high",
                "description": "RoBERTa-large for semantic similarity"
            },
            
            # Tier 2: Extended Testing
            {
                "name": "qnli-electra-base",
                "model_path": "cross-encoder/qnli-electra-base",
                "category": "Q&A Specialized",
                "expected_size_mb": 420,
                "priority": "medium",
                "description": "Question-answer focused reranking"
            },
            {
                "name": "nli-deberta-v3-base",
                "model_path": "cross-encoder/nli-deberta-v3-base",
                "category": "DeBERTa v3",
                "expected_size_mb": 420,
                "priority": "medium",
                "description": "Natural language inference specialization"
            },
            {
                "name": "ms-marco-TinyBERT-L-2-v2",
                "model_path": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                "category": "Lightweight",
                "expected_size_mb": 60,
                "priority": "medium",
                "description": "Ultra-fast lightweight option"
            },
            
            # Current baseline for comparison
            {
                "name": "ms-marco-MiniLM-L-6-v2",
                "model_path": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "category": "Current Baseline",
                "expected_size_mb": 80,
                "priority": "baseline",
                "description": "Current production reranker model"
            }
        ]

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_reranker_model(self, reranker_config):
        """Test a specific reranker model with dispatch queries"""
        print(f"\n{'='*70}")
        print(f"TESTING RERANKER: {reranker_config['name']}")
        print(f"Category: {reranker_config['category']}")
        print(f"Expected size: {reranker_config['expected_size_mb']}MB")
        print(f"Description: {reranker_config['description']}")
        print(f"{'='*70}")
        
        reranker_result = {
            "name": reranker_config["name"],
            "model_path": reranker_config["model_path"],
            "category": reranker_config["category"],
            "priority": reranker_config["priority"],
            "loading_success": False,
            "loading_time_seconds": 0,
            "memory_usage_mb": 0,
            "avg_reranking_time_ms": 0,
            "query_results": [],
            "ranking_quality_metrics": {},
            "overall_score": 0,
            "q_and_a_performance": "UNKNOWN",
            "error_details": None
        }
        
        memory_before = self.get_memory_usage()
        start_time = time.time()
        
        try:
            print(f"⏳ Loading reranker {reranker_config['name']}...")
            
            # Temporarily update config to test this reranker
            original_config = self._update_reranker_config(reranker_config["model_path"])
            
            # Test if we can import and use the reranker
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder(reranker_config["model_path"])
            
            loading_time = time.time() - start_time
            memory_after = self.get_memory_usage()
            
            reranker_result["loading_success"] = True
            reranker_result["loading_time_seconds"] = round(loading_time, 2)
            reranker_result["memory_usage_mb"] = round(memory_after - memory_before, 1)
            
            print(f"✅ Reranker loaded in {loading_time:.2f}s")
            print(f"📊 Memory usage: {reranker_result['memory_usage_mb']}MB")
            
            # Test reranking performance with dispatch queries
            print(f"\n🎯 Testing Reranking Performance on Dispatch Queries:")
            reranking_times = []
            
            for i, query_test in enumerate(self.dispatch_queries, 1):
                query = query_test["query"]
                category = query_test["category"]
                expected_topics = query_test["expected_topics"]
                
                print(f"\n   Query {i}/{len(self.dispatch_queries)}: {category}")
                print(f"   '{query[:60]}{'...' if len(query) > 60 else ''}'")
                
                try:
                    # Measure end-to-end Q&A performance with this reranker
                    query_result = self._test_query_performance(query, expected_topics, cross_encoder)
                    query_result["category"] = category
                    query_result["query"] = query
                    
                    reranker_result["query_results"].append(query_result)
                    
                    if "reranking_time_ms" in query_result:
                        reranking_times.append(query_result["reranking_time_ms"])
                    
                    print(f"   📈 Topic coverage: {query_result['topic_coverage']:.1%} - {query_result['assessment']}")
                    print(f"   ⚡ Reranking time: {query_result.get('reranking_time_ms', 0):.1f}ms")
                    
                except Exception as e:
                    print(f"   ❌ Query failed: {e}")
                    reranker_result["query_results"].append({
                        "category": category,
                        "query": query,
                        "error": str(e),
                        "assessment": "ERROR"
                    })
            
            # Calculate overall metrics
            reranker_result["avg_reranking_time_ms"] = round(np.mean(reranking_times), 1) if reranking_times else 0
            reranker_result["ranking_quality_metrics"] = self._calculate_ranking_metrics(reranker_result["query_results"])
            reranker_result["overall_score"] = self._calculate_overall_score(reranker_result)
            reranker_result["q_and_a_performance"] = self._assess_qa_performance(reranker_result)
            
            print(f"\n📈 Overall Reranker Score: {reranker_result['overall_score']:.3f}")
            print(f"🎯 Q&A Performance: {reranker_result['q_and_a_performance']}")
            
            # Cleanup
            del cross_encoder
            self._restore_reranker_config(original_config)
            
        except Exception as e:
            loading_time = time.time() - start_time
            reranker_result["loading_time_seconds"] = round(loading_time, 2)
            reranker_result["error_details"] = str(e)
            print(f"❌ Reranker loading failed: {e}")
            print(f"🔍 Error details: {traceback.format_exc()}")
            
            # Restore config even on error
            if 'original_config' in locals():
                self._restore_reranker_config(original_config)
        
        return reranker_result

    def _update_reranker_config(self, reranker_model_path):
        """Temporarily update config with new reranker model"""
        try:
            config_path = project_root / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            original_reranker = config["retrieval"]["reranker_model"]
            config["retrieval"]["reranker_model"] = reranker_model_path
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return original_reranker
        except Exception as e:
            print(f"Warning: Could not update config: {e}")
            return None

    def _restore_reranker_config(self, original_reranker):
        """Restore original reranker config"""
        if original_reranker is None:
            return
        
        try:
            config_path = project_root / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config["retrieval"]["reranker_model"] = original_reranker
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not restore config: {e}")

    def _test_query_performance(self, query, expected_topics, cross_encoder):
        """Test Q&A performance for a specific query with current reranker"""
        try:
            # Import the retrieval system to test with current reranker
            from retrieval.retrieval_system import UnifiedRetrievalSystem
            from core.document_processor import DocumentProcessor
            
            # Initialize document processor and retrieval system
            doc_processor = DocumentProcessor()
            
            # Check if we have a document loaded (use KTI guide for testing)
            if not hasattr(doc_processor, 'index') or doc_processor.index is None:
                # Load the KTI dispatch guide for testing
                kti_path = project_root / "data" / "KTI_Dispatch_Guide.pdf"
                if kti_path.exists():
                    doc_processor.upload_file(str(kti_path))
                else:
                    # Try to find any PDF in data directory
                    data_dir = project_root / "data"
                    if data_dir.exists():
                        pdf_files = list(data_dir.glob("*.pdf"))
                        if pdf_files:
                            doc_processor.upload_file(str(pdf_files[0]))
            
            # Test reranking performance
            rerank_start = time.time()
            
            # Use reranking engine directly to measure performance
            from retrieval.engines.reranking_engine import RerankingEngine
            reranking_engine = RerankingEngine(doc_processor)
            
            # Perform search with reranking
            context, search_time, num_chunks, retrieval_scores = reranking_engine.search(query, top_k=15)
            
            rerank_time = (time.time() - rerank_start) * 1000  # Convert to ms
            
            # Analyze topic coverage
            topic_coverage = self._analyze_topic_coverage(context, expected_topics)
            
            assessment = "GOOD" if topic_coverage >= 0.6 else "PARTIAL" if topic_coverage >= 0.4 else "POOR"
            
            return {
                "topic_coverage": topic_coverage,
                "assessment": assessment,
                "num_chunks": num_chunks,
                "context_length": len(context),
                "search_time_ms": search_time * 1000,
                "reranking_time_ms": rerank_time,
                "retrieval_scores": retrieval_scores
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "assessment": "ERROR",
                "topic_coverage": 0.0
            }

    def _analyze_topic_coverage(self, context, expected_topics):
        """Analyze how well the context covers expected topics"""
        if not context or not expected_topics:
            return 0.0
        
        context_lower = context.lower()
        covered_topics = 0
        
        for topic in expected_topics:
            # Check for topic mentions with some flexibility
            topic_variations = [
                topic.lower(),
                topic.lower().replace(" ", ""),
                topic.lower().replace("-", " "),
                topic.lower().replace("_", " ")
            ]
            
            if any(variation in context_lower for variation in topic_variations):
                covered_topics += 1
        
        return covered_topics / len(expected_topics)

    def _calculate_ranking_metrics(self, query_results):
        """Calculate ranking quality metrics from query results"""
        if not query_results:
            return {}
        
        successful_queries = [q for q in query_results if "topic_coverage" in q]
        
        if not successful_queries:
            return {"success_rate": 0.0}
        
        # Calculate various metrics
        topic_coverages = [q["topic_coverage"] for q in successful_queries]
        reranking_times = [q.get("reranking_time_ms", 0) for q in successful_queries]
        
        good_queries = len([q for q in successful_queries if q.get("assessment") == "GOOD"])
        partial_queries = len([q for q in successful_queries if q.get("assessment") == "PARTIAL"])
        
        return {
            "success_rate": len(successful_queries) / len(query_results),
            "avg_topic_coverage": np.mean(topic_coverages),
            "good_coverage_rate": good_queries / len(successful_queries),
            "partial_or_better_rate": (good_queries + partial_queries) / len(successful_queries),
            "avg_reranking_time_ms": np.mean(reranking_times),
            "coverage_std": np.std(topic_coverages)
        }

    def _calculate_overall_score(self, reranker_result):
        """Calculate weighted overall performance score"""
        if not reranker_result["loading_success"]:
            return 0.0
        
        metrics = reranker_result.get("ranking_quality_metrics", {})
        
        # Quality metrics (weight: 80%)
        avg_coverage = metrics.get("avg_topic_coverage", 0)
        good_rate = metrics.get("good_coverage_rate", 0)
        success_rate = metrics.get("success_rate", 0)
        
        quality_score = (avg_coverage * 0.4) + (good_rate * 0.3) + (success_rate * 0.3)
        
        # Performance metrics (weight: 20%)
        loading_time = reranker_result["loading_time_seconds"]
        memory_usage = reranker_result["memory_usage_mb"]
        reranking_time = reranker_result["avg_reranking_time_ms"]
        
        # Normalize performance factors
        loading_score = max(0, (60 - min(loading_time, 60)) / 60)
        memory_score = max(0, (2000 - min(memory_usage, 2000)) / 2000)
        speed_score = max(0, (1000 - min(reranking_time, 1000)) / 1000)
        
        performance_score = (loading_score + memory_score + speed_score) / 3
        
        # Combined score
        overall_score = (quality_score * 0.8) + (performance_score * 0.2)
        
        return overall_score

    def _assess_qa_performance(self, reranker_result):
        """Assess Q&A performance level"""
        if not reranker_result["loading_success"]:
            return "FAILED"
        
        metrics = reranker_result.get("ranking_quality_metrics", {})
        
        good_rate = metrics.get("good_coverage_rate", 0)
        partial_rate = metrics.get("partial_or_better_rate", 0)
        avg_coverage = metrics.get("avg_topic_coverage", 0)
        
        if good_rate >= 0.6 and avg_coverage >= 0.7:
            return "EXCELLENT"
        elif good_rate >= 0.4 and partial_rate >= 0.7:
            return "GOOD"
        elif partial_rate >= 0.5:
            return "PARTIAL"
        else:
            return "POOR"

    def run_evaluation(self, priority_filter=None):
        """Run evaluation on selected reranker models"""
        print("WORKAPP2 COMPREHENSIVE RERANKER EVALUATION")
        print("=" * 80)
        print(f"Testing {len(self.rerankers_to_test)} reranker models with e5-large-v2")
        print(f"Dispatch queries: {len(self.dispatch_queries)}")
        print("=" * 80)
        
        models_to_run = self.rerankers_to_test
        if priority_filter:
            models_to_run = [m for m in self.rerankers_to_test if m["priority"] in priority_filter]
            print(f"🎯 Running priority filter: {priority_filter}")
            print(f"Models to test: {len(models_to_run)}")
        
        for i, reranker_config in enumerate(models_to_run, 1):
            print(f"\n📊 Progress: {i}/{len(models_to_run)} reranker models")
            
            reranker_result = self.test_reranker_model(reranker_config)
            self.results["rerankers_tested"].append(reranker_result)
            
            # Memory cleanup
            import gc
            gc.collect()
        
        # Generate evaluation summary and recommendations
        self.generate_summary()
        self.generate_recommendations()
        
        return self.results

    def generate_summary(self):
        """Generate evaluation summary statistics"""
        successful_rerankers = [r for r in self.results["rerankers_tested"] if r["loading_success"]]
        failed_rerankers = [r for r in self.results["rerankers_tested"] if not r["loading_success"]]
        
        if successful_rerankers:
            # Top performers by overall score
            top_performers = sorted(successful_rerankers, key=lambda x: x["overall_score"], reverse=True)[:5]
            
            # Best Q&A performance
            excellent_rerankers = [r for r in successful_rerankers if r["q_and_a_performance"] == "EXCELLENT"]
            good_rerankers = [r for r in successful_rerankers if r["q_and_a_performance"] == "GOOD"]
            
            self.results["evaluation_summary"] = {
                "total_rerankers_tested": len(self.results["rerankers_tested"]),
                "successful_rerankers": len(successful_rerankers),
                "failed_rerankers": len(failed_rerankers),
                "top_5_performers": [
                    {
                        "name": r["name"],
                        "overall_score": r["overall_score"],
                        "q_and_a_performance": r["q_and_a_performance"],
                        "category": r["category"]
                    } for r in top_performers
                ],
                "excellent_qa_performance": len(excellent_rerankers),
                "good_qa_performance": len(good_rerankers),
                "performance_stats": {
                    "avg_loading_time": np.mean([r["loading_time_seconds"] for r in successful_rerankers]),
                    "avg_memory_usage": np.mean([r["memory_usage_mb"] for r in successful_rerankers]),
                    "avg_reranking_time": np.mean([r["avg_reranking_time_ms"] for r in successful_rerankers])
                }
            }

    def generate_recommendations(self):
        """Generate deployment recommendations"""
        successful_rerankers = [r for r in self.results["rerankers_tested"] if r["loading_success"]]
        
        if not successful_rerankers:
            self.results["recommendations"] = {
                "status": "NO_SUCCESSFUL_RERANKERS",
                "message": "No rerankers loaded successfully - check environment and dependencies"
            }
            return
        
        # Sort by overall score
        ranked_rerankers = sorted(successful_rerankers, key=lambda x: x["overall_score"], reverse=True)
        
        # Find best rerankers by category
        best_overall = ranked_rerankers[0]
        best_performance = min(successful_rerankers, key=lambda x: x["avg_reranking_time_ms"] + x["loading_time_seconds"])
        best_qa = max(successful_rerankers, key=lambda x: x["overall_score"] if x["q_and_a_performance"] in ["EXCELLENT", "GOOD"] else 0)
        
        # Current baseline performance
        baseline_reranker = next((r for r in successful_rerankers if r["name"] == "ms-marco-MiniLM-L-6-v2"), None)
        
        self.results["recommendations"] = {
            "recommended_for_production": {
                "model_name": best_overall["name"],
                "model_path": best_overall["model_path"],
                "overall_score": best_overall["overall_score"],
                "q_and_a_performance": best_overall["q_and_a_performance"],
                "justification": f"Highest overall score ({best_overall['overall_score']:.3f}) with {best_overall['q_and_a_performance']} Q&A performance"
            },
            "best_performance": {
                "model_name": best_performance["name"],
                "avg_reranking_time_ms": best_performance["avg_reranking_time_ms"],
                "loading_time_seconds": best_performance["loading_time_seconds"],
                "justification": "Fastest reranking performance for production deployment"
            },
            "best_qa_performance": {
                "model_name": best_qa["name"],
                "q_and_a_performance": best_qa["q_and_a_performance"],
                "overall_score": best_qa["overall_score"],
                "justification": "Best Q&A performance for dispatch domain queries"
            }
        }
        
        if baseline_reranker:
            improvement = best_overall["overall_score"] - baseline_reranker["overall_score"]
            self.results["recommendations"]["improvement_vs_baseline"] = {
                "current_baseline_score": baseline_reranker["overall_score"],
                "recommended_reranker_score": best_overall["overall_score"],
                "improvement": improvement,
                "percentage_improvement": (improvement / baseline_reranker["overall_score"]) * 100 if baseline_reranker["overall_score"] > 0 else 0
            }

    def save_results(self, filename=None):
        """Save evaluation results to JSON file"""
        if filename is None:
            filename = f"reranker_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_file = project_root / filename
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        return results_file

    def print_summary(self):
        """Print evaluation summary to console"""
        print("\n" + "=" * 80)
        print("RERANKER EVALUATION SUMMARY")
        print("=" * 80)
        
        summary = self.results.get("evaluation_summary", {})
        recommendations = self.results.get("recommendations", {})
        
        print(f"📊 Total rerankers tested: {summary.get('total_rerankers_tested', 0)}")
        print(f"✅ Successful rerankers: {summary.get('successful_rerankers', 0)}")
        print(f"❌ Failed rerankers: {summary.get('failed_rerankers', 0)}")
        
        if summary.get('top_5_performers'):
            print(f"\n🏆 TOP 5 PERFORMERS:")
            for i, reranker in enumerate(summary['top_5_performers'], 1):
                print(f"   {i}. {reranker['name']} - Score: {reranker['overall_score']:.3f} - Q&A: {reranker['q_and_a_performance']}")
        
        if recommendations.get('recommended_for_production'):
            rec = recommendations['recommended_for_production']
            print(f"\n🎯 PRODUCTION RECOMMENDATION:")
            print(f"   Reranker: {rec['model_name']}")
            print(f"   Path: {rec['model_path']}")
            print(f"   Score: {rec['overall_score']:.3f}")
            print(f"   Q&A Performance: {rec['q_and_a_performance']}")
            print(f"   Reason: {rec['justification']}")
        
        if recommendations.get('improvement_vs_baseline'):
            imp = recommendations['improvement_vs_baseline']
            print(f"\n📈 IMPROVEMENT VS BASELINE:")
            print(f"   Baseline (ms-marco-MiniLM-L-6-v2): {imp['current_baseline_score']:.3f}")
            print(f"   Recommended reranker: {imp['recommended_reranker_score']:.3f}")
            print(f"   Improvement: +{imp['improvement']:.3f} ({imp['percentage_improvement']:.1f}%)")


def main():
    """Main evaluation function"""
    evaluator = RerankerEvaluator()
    
    # Run evaluation (start with high priority rerankers)
    print("🚀 Starting with high priority rerankers...")
    results = evaluator.run_evaluation(priority_filter=["high", "baseline"])
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    results_file = evaluator.save_results()
    
    return results, results_file


if __name__ == "__main__":
    results, results_file = main()
