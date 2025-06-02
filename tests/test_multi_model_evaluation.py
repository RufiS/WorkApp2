"""
Multi-Model Embedding Evaluation Suite for WorkApp2
Tests 13+ embedding models for dispatch domain understanding
Provides automated ranking and deployment recommendations
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class MultiModelEvaluator:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "models_tested": [],
            "evaluation_summary": {},
            "recommendations": {}
        }
        
        # Core dispatch terminology pairs for testing
        self.core_dispatch_pairs = [
            ("text message", "SMS", "dispatch_communication"),
            ("Field Engineer", "FE", "dispatch_roles"),
            ("RingCentral", "phone system", "dispatch_tools"),
            ("dispatch", "send technician", "dispatch_actions"),
            ("emergency call", "urgent ticket", "dispatch_priority")
        ]
        
        # Extended dispatch vocabulary pairs
        self.extended_dispatch_pairs = [
            ("troubleshooting", "problem solving", "technical_process"),
            ("on-site", "field visit", "service_location"),
            ("remote access", "VPN connection", "technical_tools"),
            ("escalation", "supervisor call", "process_escalation"),
            ("follow-up", "callback", "customer_service"),
            ("work order", "service ticket", "documentation"),
            ("service request", "customer call", "customer_interaction"),
            ("appointment", "scheduled visit", "time_management"),
            ("billing", "invoice", "financial_process"),
            ("mobile app", "smartphone application", "technical_tools"),
            ("technician", "service engineer", "dispatch_roles"),
            ("urgent", "high priority", "priority_levels"),
            ("client", "customer", "stakeholder_roles"),
            ("installation", "setup", "service_types"),
            ("maintenance", "regular service", "service_types")
        ]
        
        # Model configurations to test
        self.models_to_test = [
            # Category A: Enhanced General Purpose
            {
                "name": "all-mpnet-base-v2",
                "model_path": "sentence-transformers/all-mpnet-base-v2",
                "category": "Enhanced General Purpose",
                "expected_size_mb": 420,
                "priority": "high"
            },
            {
                "name": "all-MiniLM-L12-v2", 
                "model_path": "sentence-transformers/all-MiniLM-L12-v2",
                "category": "Enhanced General Purpose",
                "expected_size_mb": 120,
                "priority": "high"
            },
            {
                "name": "all-roberta-large-v1",
                "model_path": "sentence-transformers/all-roberta-large-v1", 
                "category": "Enhanced General Purpose",
                "expected_size_mb": 1200,
                "priority": "medium"
            },
            
            # Category B: Technical Domain
            {
                "name": "paraphrase-distilroberta-base-v2",
                "model_path": "sentence-transformers/paraphrase-distilroberta-base-v2",
                "category": "Technical Domain",
                "expected_size_mb": 290,
                "priority": "high"
            },
            
            # Category C: Q&A Specialized  
            {
                "name": "multi-qa-mpnet-base-dot-v1",
                "model_path": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
                "category": "Q&A Specialized",
                "expected_size_mb": 420,
                "priority": "high"
            },
            {
                "name": "nli-mpnet-base-v2",
                "model_path": "sentence-transformers/nli-mpnet-base-v2",
                "category": "Q&A Specialized", 
                "expected_size_mb": 420,
                "priority": "medium"
            },
            
            # Category D: State-of-the-Art Large Models
            {
                "name": "e5-large-v2",
                "model_path": "intfloat/e5-large-v2",
                "category": "State-of-the-Art",
                "expected_size_mb": 1200,
                "priority": "high"
            },
            {
                "name": "bge-large-en-v1.5",
                "model_path": "BAAI/bge-large-en-v1.5",
                "category": "State-of-the-Art",
                "expected_size_mb": 1300,
                "priority": "medium"
            },
            {
                "name": "gtr-t5-large",
                "model_path": "sentence-transformers/gtr-t5-large", 
                "category": "State-of-the-Art",
                "expected_size_mb": 670,
                "priority": "medium"
            },
            
            # Category E: Retrieval Specialized
            {
                "name": "msmarco-distilbert-base-tas-b",
                "model_path": "sentence-transformers/msmarco-distilbert-base-tas-b",
                "category": "Retrieval Specialized",
                "expected_size_mb": 250,
                "priority": "medium"
            },
            
            # Current baseline for comparison
            {
                "name": "all-MiniLM-L6-v2",
                "model_path": "sentence-transformers/all-MiniLM-L6-v2", 
                "category": "Current Baseline",
                "expected_size_mb": 80,
                "priority": "baseline"
            }
        ]

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_model_loading(self, model_config):
        """Test if model can be loaded and basic functionality works"""
        print(f"\n{'='*60}")
        print(f"TESTING MODEL: {model_config['name']}")
        print(f"Category: {model_config['category']}")
        print(f"Expected size: {model_config['expected_size_mb']}MB")
        print(f"{'='*60}")
        
        model_result = {
            "name": model_config["name"],
            "model_path": model_config["model_path"],
            "category": model_config["category"],
            "priority": model_config["priority"],
            "loading_success": False,
            "loading_time_seconds": 0,
            "memory_usage_mb": 0,
            "inference_speed_ms": 0,
            "core_similarity_scores": [],
            "extended_similarity_scores": [],
            "overall_score": 0,
            "dispatch_understanding": "UNKNOWN",
            "error_details": None
        }
        
        memory_before = self.get_memory_usage()
        start_time = time.time()
        
        try:
            print(f"⏳ Loading {model_config['name']}...")
            
            # Import sentence transformers
            from sentence_transformers import SentenceTransformer
            
            # Load the model
            model = SentenceTransformer(model_config["model_path"])
            loading_time = time.time() - start_time
            memory_after = self.get_memory_usage()
            
            model_result["loading_success"] = True
            model_result["loading_time_seconds"] = round(loading_time, 2)
            model_result["memory_usage_mb"] = round(memory_after - memory_before, 1)
            
            print(f"✅ Model loaded in {loading_time:.2f}s")
            print(f"📊 Memory usage: {model_result['memory_usage_mb']}MB")
            
            # Test inference speed
            speed_start = time.time()
            test_embedding = model.encode(["test inference speed"])
            inference_time = (time.time() - speed_start) * 1000  # Convert to ms
            model_result["inference_speed_ms"] = round(inference_time, 1)
            print(f"⚡ Inference speed: {inference_time:.1f}ms")
            
            # Test core dispatch pairs
            print(f"\n🎯 Testing Core Dispatch Terminology:")
            model_result["core_similarity_scores"] = self.test_similarity_pairs(
                model, self.core_dispatch_pairs, "core"
            )
            
            # Test extended dispatch pairs  
            print(f"\n🔍 Testing Extended Dispatch Vocabulary:")
            model_result["extended_similarity_scores"] = self.test_similarity_pairs(
                model, self.extended_dispatch_pairs, "extended"
            )
            
            # Calculate overall performance
            model_result["overall_score"] = self.calculate_overall_score(model_result)
            model_result["dispatch_understanding"] = self.assess_dispatch_understanding(model_result)
            
            print(f"\n📈 Overall Score: {model_result['overall_score']:.3f}")
            print(f"🧠 Dispatch Understanding: {model_result['dispatch_understanding']}")
            
            # Cleanup to save memory
            del model
            
        except Exception as e:
            loading_time = time.time() - start_time
            model_result["loading_time_seconds"] = round(loading_time, 2)
            model_result["error_details"] = str(e)
            print(f"❌ Model loading failed: {e}")
            print(f"🔍 Error details: {traceback.format_exc()}")
        
        return model_result

    def test_similarity_pairs(self, model, pairs, test_type):
        """Test similarity scores for pairs of terms"""
        similarity_results = []
        
        for term1, term2, domain in pairs:
            try:
                # Get embeddings
                embedding1 = model.encode([term1])
                embedding2 = model.encode([term2])
                
                # Calculate cosine similarity
                similarity = np.dot(embedding1[0], embedding2[0]) / (
                    np.linalg.norm(embedding1[0]) * np.linalg.norm(embedding2[0])
                )
                
                assessment = "HIGH" if similarity > 0.7 else "MEDIUM" if similarity > 0.5 else "LOW"
                
                similarity_result = {
                    "term1": term1,
                    "term2": term2, 
                    "domain": domain,
                    "similarity": float(similarity),
                    "assessment": assessment
                }
                
                similarity_results.append(similarity_result)
                print(f"   {term1} ↔ {term2}: {similarity:.3f} ({assessment})")
                
            except Exception as e:
                similarity_result = {
                    "term1": term1,
                    "term2": term2,
                    "domain": domain,
                    "error": str(e),
                    "assessment": "ERROR"
                }
                similarity_results.append(similarity_result)
                print(f"   {term1} ↔ {term2}: ERROR - {e}")
        
        return similarity_results

    def calculate_overall_score(self, model_result):
        """Calculate weighted overall performance score"""
        if not model_result["loading_success"]:
            return 0.0
        
        # Core similarity scores (weight: 70%)
        core_scores = [s["similarity"] for s in model_result["core_similarity_scores"] 
                      if "similarity" in s]
        core_avg = np.mean(core_scores) if core_scores else 0
        
        # Extended similarity scores (weight: 20%)
        extended_scores = [s["similarity"] for s in model_result["extended_similarity_scores"]
                          if "similarity" in s]
        extended_avg = np.mean(extended_scores) if extended_scores else 0
        
        # Performance factors (weight: 10%)
        # Normalize loading time (faster = better, cap at 60s)
        loading_score = max(0, (60 - min(model_result["loading_time_seconds"], 60)) / 60)
        
        # Normalize memory usage (lower = better, reasonable cap at 1000MB)
        memory_score = max(0, (1000 - min(model_result["memory_usage_mb"], 1000)) / 1000)
        
        # Normalize inference speed (faster = better, cap at 1000ms)
        speed_score = max(0, (1000 - min(model_result["inference_speed_ms"], 1000)) / 1000)
        
        performance_avg = (loading_score + memory_score + speed_score) / 3
        
        # Weighted overall score
        overall_score = (core_avg * 0.7) + (extended_avg * 0.2) + (performance_avg * 0.1)
        
        return overall_score

    def assess_dispatch_understanding(self, model_result):
        """Assess model's dispatch domain understanding capability"""
        if not model_result["loading_success"]:
            return "FAILED"
        
        # Count high/medium scores in core pairs
        core_scores = model_result["core_similarity_scores"]
        high_core = len([s for s in core_scores if s.get("assessment") == "HIGH"])
        medium_core = len([s for s in core_scores if s.get("assessment") == "MEDIUM"])
        
        # Count high/medium scores in extended pairs  
        extended_scores = model_result["extended_similarity_scores"]
        high_extended = len([s for s in extended_scores if s.get("assessment") == "HIGH"])
        medium_extended = len([s for s in extended_scores if s.get("assessment") == "MEDIUM"])
        
        # Assessment criteria
        if high_core >= 3 and (high_core + medium_core) >= 4:
            if high_extended >= 8 and (high_extended + medium_extended) >= 12:
                return "EXCELLENT"
            else:
                return "GOOD"
        elif high_core >= 2 and (high_core + medium_core) >= 3:
            return "PARTIAL"
        else:
            return "POOR"

    def run_evaluation(self, priority_filter=None):
        """Run evaluation on all models"""
        print("WORKAPP2 MULTI-MODEL EMBEDDING EVALUATION")
        print("=" * 80)
        print(f"Testing {len(self.models_to_test)} embedding models for dispatch domain understanding")
        print(f"Core terminology pairs: {len(self.core_dispatch_pairs)}")
        print(f"Extended vocabulary pairs: {len(self.extended_dispatch_pairs)}")
        print("=" * 80)
        
        models_to_run = self.models_to_test
        if priority_filter:
            models_to_run = [m for m in self.models_to_test if m["priority"] in priority_filter]
            print(f"🎯 Running priority filter: {priority_filter}")
            print(f"Models to test: {len(models_to_run)}")
        
        for i, model_config in enumerate(models_to_run, 1):
            print(f"\n📊 Progress: {i}/{len(models_to_run)} models")
            
            model_result = self.test_model_loading(model_config)
            self.results["models_tested"].append(model_result)
            
            # Quick memory cleanup
            import gc
            gc.collect()
        
        # Generate evaluation summary
        self.generate_summary()
        self.generate_recommendations()
        
        return self.results

    def generate_summary(self):
        """Generate evaluation summary statistics"""
        successful_models = [m for m in self.results["models_tested"] if m["loading_success"]]
        failed_models = [m for m in self.results["models_tested"] if not m["loading_success"]]
        
        if successful_models:
            # Top performers by overall score
            top_performers = sorted(successful_models, key=lambda x: x["overall_score"], reverse=True)[:5]
            
            # Best dispatch understanding
            excellent_models = [m for m in successful_models if m["dispatch_understanding"] == "EXCELLENT"]
            good_models = [m for m in successful_models if m["dispatch_understanding"] == "GOOD"]
            
            self.results["evaluation_summary"] = {
                "total_models_tested": len(self.results["models_tested"]),
                "successful_models": len(successful_models),
                "failed_models": len(failed_models),
                "top_5_performers": [
                    {
                        "name": m["name"],
                        "overall_score": m["overall_score"],
                        "dispatch_understanding": m["dispatch_understanding"],
                        "category": m["category"]
                    } for m in top_performers
                ],
                "excellent_dispatch_understanding": len(excellent_models),
                "good_dispatch_understanding": len(good_models),
                "performance_stats": {
                    "avg_loading_time": np.mean([m["loading_time_seconds"] for m in successful_models]),
                    "avg_memory_usage": np.mean([m["memory_usage_mb"] for m in successful_models]),
                    "avg_inference_speed": np.mean([m["inference_speed_ms"] for m in successful_models])
                }
            }

    def generate_recommendations(self):
        """Generate deployment recommendations"""
        successful_models = [m for m in self.results["models_tested"] if m["loading_success"]]
        
        if not successful_models:
            self.results["recommendations"] = {
                "status": "NO_SUCCESSFUL_MODELS",
                "message": "No models loaded successfully - check environment and dependencies"
            }
            return
        
        # Sort by overall score
        ranked_models = sorted(successful_models, key=lambda x: x["overall_score"], reverse=True)
        
        # Find best models by category
        best_overall = ranked_models[0]
        best_performance = min(successful_models, key=lambda x: x["memory_usage_mb"] + x["loading_time_seconds"])
        best_dispatch = max(successful_models, key=lambda x: x["overall_score"] if x["dispatch_understanding"] in ["EXCELLENT", "GOOD"] else 0)
        
        # Current baseline performance
        baseline_model = next((m for m in successful_models if m["name"] == "all-MiniLM-L6-v2"), None)
        
        self.results["recommendations"] = {
            "recommended_for_production": {
                "model_name": best_overall["name"],
                "model_path": best_overall["model_path"],
                "overall_score": best_overall["overall_score"],
                "dispatch_understanding": best_overall["dispatch_understanding"],
                "justification": f"Highest overall score ({best_overall['overall_score']:.3f}) with {best_overall['dispatch_understanding']} dispatch understanding"
            },
            "best_performance": {
                "model_name": best_performance["name"],
                "memory_usage_mb": best_performance["memory_usage_mb"],
                "loading_time_seconds": best_performance["loading_time_seconds"],
                "justification": "Lowest resource usage for production deployment"
            },
            "best_dispatch_understanding": {
                "model_name": best_dispatch["name"],
                "dispatch_understanding": best_dispatch["dispatch_understanding"],
                "overall_score": best_dispatch["overall_score"],
                "justification": "Best semantic understanding of dispatch domain terminology"
            }
        }
        
        if baseline_model:
            improvement = best_overall["overall_score"] - baseline_model["overall_score"]
            self.results["recommendations"]["improvement_vs_baseline"] = {
                "current_baseline_score": baseline_model["overall_score"],
                "recommended_model_score": best_overall["overall_score"],
                "improvement": improvement,
                "percentage_improvement": (improvement / baseline_model["overall_score"]) * 100 if baseline_model["overall_score"] > 0 else 0
            }

    def save_results(self, filename=None):
        """Save evaluation results to JSON file"""
        if filename is None:
            filename = f"multi_model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_file = project_root / filename
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        return results_file

    def print_summary(self):
        """Print evaluation summary to console"""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        summary = self.results.get("evaluation_summary", {})
        recommendations = self.results.get("recommendations", {})
        
        print(f"📊 Total models tested: {summary.get('total_models_tested', 0)}")
        print(f"✅ Successful models: {summary.get('successful_models', 0)}")
        print(f"❌ Failed models: {summary.get('failed_models', 0)}")
        
        if summary.get('top_5_performers'):
            print(f"\n🏆 TOP 5 PERFORMERS:")
            for i, model in enumerate(summary['top_5_performers'], 1):
                print(f"   {i}. {model['name']} - Score: {model['overall_score']:.3f} - Understanding: {model['dispatch_understanding']}")
        
        if recommendations.get('recommended_for_production'):
            rec = recommendations['recommended_for_production']
            print(f"\n🎯 PRODUCTION RECOMMENDATION:")
            print(f"   Model: {rec['model_name']}")
            print(f"   Path: {rec['model_path']}")
            print(f"   Score: {rec['overall_score']:.3f}")
            print(f"   Understanding: {rec['dispatch_understanding']}")
            print(f"   Reason: {rec['justification']}")
        
        if recommendations.get('improvement_vs_baseline'):
            imp = recommendations['improvement_vs_baseline']
            print(f"\n📈 IMPROVEMENT VS BASELINE:")
            print(f"   Baseline (all-MiniLM-L6-v2): {imp['current_baseline_score']:.3f}")
            print(f"   Recommended model: {imp['recommended_model_score']:.3f}")
            print(f"   Improvement: +{imp['improvement']:.3f} ({imp['percentage_improvement']:.1f}%)")


def main():
    """Main evaluation function"""
    evaluator = MultiModelEvaluator()
    
    # Run evaluation (start with high priority models)
    print("🚀 Starting with high priority models...")
    results = evaluator.run_evaluation(priority_filter=["high", "baseline"])
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    results_file = evaluator.save_results()
    
    return results, results_file


if __name__ == "__main__":
    results, results_file = main()
