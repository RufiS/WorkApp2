"""
Comprehensive Embedding Model + Retrieval Performance Test
Tests multiple embedding models with actual failed queries from investigation
Rebuilds index with KTI_Dispatch_Guide.pdf for each model test
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
import traceback
from typing import List, Dict, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.document_processor import DocumentProcessor
from retrieval.engines.vector_engine import VectorEngine
from core.config import retrieval_config


class ComprehensiveEmbeddingRetrievalTester:
    def __init__(self):
        """Initialize comprehensive embedding and retrieval tester"""
        self.project_root = project_root
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "comprehensive_embedding_retrieval",
            "models_tested": [],
            "query_results": {},
            "model_rankings": {},
            "best_performers": {}
        }
        
        # Test document
        self.test_document = "./KTI_Dispatch_Guide.pdf"
        
        # Failed queries from investigation roadmap
        self.critical_failed_queries = [
            "What is our main phone number?",
            "What's the complete text messaging workflow?", 
            "How do I create a customer concern?",
            "Are we licensed and insured?"
        ]
        
        # Natural language test queries (NO hardcoded variations - test actual user questions)
        self.natural_test_queries = [
            "What is our main phone number?",
            "How do I handle text messages?", 
            "How do I create a customer concern?",
            "Are we licensed and insured?"
        ]
        
        # Comprehensive local embedding models to test (NO OpenAI models)
        self.models_to_test = [
            # Current baseline
            {"name": "e5-large-v2", "model_path": "intfloat/e5-large-v2", "category": "Current"},
            
            # E5 family (Microsoft)
            {"name": "e5-base-v2", "model_path": "intfloat/e5-base-v2", "category": "E5 Family"},
            {"name": "e5-small-v2", "model_path": "intfloat/e5-small-v2", "category": "E5 Family"},
            {"name": "multilingual-e5-large", "model_path": "intfloat/multilingual-e5-large", "category": "E5 Family"},
            
            # BGE family (BAAI)
            {"name": "bge-large-en-v1.5", "model_path": "BAAI/bge-large-en-v1.5", "category": "BGE Family"},
            {"name": "bge-base-en-v1.5", "model_path": "BAAI/bge-base-en-v1.5", "category": "BGE Family"},
            {"name": "bge-small-en-v1.5", "model_path": "BAAI/bge-small-en-v1.5", "category": "BGE Family"},
            
            # Sentence Transformers family
            {"name": "all-mpnet-base-v2", "model_path": "sentence-transformers/all-mpnet-base-v2", "category": "SentenceTransformers"},
            {"name": "all-MiniLM-L12-v2", "model_path": "sentence-transformers/all-MiniLM-L12-v2", "category": "SentenceTransformers"},
            {"name": "all-MiniLM-L6-v2", "model_path": "sentence-transformers/all-MiniLM-L6-v2", "category": "SentenceTransformers"},
            {"name": "all-roberta-large-v1", "model_path": "sentence-transformers/all-roberta-large-v1", "category": "SentenceTransformers"},
            
            # Instruction-following models
            {"name": "instructor-large", "model_path": "hkunlp/instructor-large", "category": "Instruction-Following"},
            {"name": "instructor-base", "model_path": "hkunlp/instructor-base", "category": "Instruction-Following"},
            
            # Retrieval-specialized models
            {"name": "gtr-t5-large", "model_path": "sentence-transformers/gtr-t5-large", "category": "Retrieval-Specialized"},
            {"name": "msmarco-distilbert-base-v4", "model_path": "sentence-transformers/msmarco-distilbert-base-v4", "category": "Retrieval-Specialized"},
            {"name": "multi-qa-mpnet-base-dot-v1", "model_path": "sentence-transformers/multi-qa-mpnet-base-dot-v1", "category": "Retrieval-Specialized"},
            
            # Alternative architectures
            {"name": "paraphrase-mpnet-base-v2", "model_path": "sentence-transformers/paraphrase-mpnet-base-v2", "category": "Alternative"},
            {"name": "paraphrase-distilroberta-base-v2", "model_path": "sentence-transformers/paraphrase-distilroberta-base-v2", "category": "Alternative"}
        ]
        
        # Content expectations for validation
        self.content_expectations = {
            "phone_numbers": {
                "expected_content": ["480-999-3046", "main company number", "metro phone"],
                "success_criteria": "Should find main company number 480-999-3046"
            },
            "text_messaging": {
                "expected_content": ["text message", "SMS", "freshdesk", "ticket", "workflow"],
                "success_criteria": "Should find text messaging procedures and workflow steps"
            },
            "customer_concerns": {
                "expected_content": ["customer concern", "complaint", "freshdesk", "ticket", "helpdesk"],
                "success_criteria": "Should find customer concern creation process"
            }
        }
        
    def clear_and_rebuild_index(self, model_name: str) -> bool:
        """Clear existing index and rebuild with test document using specified model"""
        try:
            print(f"🗑️ Clearing existing index...")
            
            # Clear data/index directory
            index_dir = self.project_root / "data" / "index"
            if index_dir.exists():
                shutil.rmtree(index_dir)
                print(f"   Removed {index_dir}")
            
            # Clear current_index directory
            current_index_dir = self.project_root / "current_index"
            if current_index_dir.exists():
                shutil.rmtree(current_index_dir)
                print(f"   Removed {current_index_dir}")
            
            print(f"🏗️ Rebuilding index with model: {model_name}")
            
            # Update config to use new model
            print(f"   Updating config to use {model_name}")
            retrieval_config.embedding_model = model_name
            
            # Initialize document processor and rebuild index
            print(f"   Processing document: {self.test_document}")
            doc_processor = DocumentProcessor()
            
            # Check if test document exists
            test_doc_path = self.project_root / self.test_document
            if not test_doc_path.exists():
                print(f"❌ Test document not found: {test_doc_path}")
                return False
            
            # Process the document
            index, chunks = doc_processor.process_documents([str(test_doc_path)])
            success = index is not None and len(chunks) > 0
            
            if success:
                # Save the index to disk
                doc_processor.save_index()
                print(f"✅ Index rebuilt successfully with {model_name}")
                return True
            else:
                print(f"❌ Failed to rebuild index with {model_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error rebuilding index: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False
    
    def test_pure_vector_retrieval(self, queries: List[str], model_name: str, doc_processor: DocumentProcessor, top_k: int = 50) -> Dict[str, Any]:
        """Test pure vector retrieval (no reranking) for given queries"""
        try:
            print(f"🔍 Testing vector retrieval with top_k={top_k}")
            
            # Use the provided document processor with loaded index
            vector_engine = VectorEngine(doc_processor)
            
            query_results = {}
            
            for query in queries:
                try:
                    print(f"   Testing: '{query}'")
                    
                    # Perform pure vector search
                    start_time = time.time()
                    context, search_time, num_chunks, retrieval_scores = vector_engine.search(query, top_k=top_k)
                    
                    # Parse results - we need to get the raw chunks, not formatted context
                    raw_results = doc_processor.search(query, top_k=top_k) if doc_processor.has_index() else []
                    
                    result_chunks = []
                    if raw_results:
                        for i, result in enumerate(raw_results):
                            chunk_text = result.get("text", "")
                            score = result.get("score", 0.0)
                            result_chunks.append({
                                "rank": i + 1,
                                "content_preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text,
                                "similarity_score": float(score),
                                "content_length": len(chunk_text)
                            })
                    
                    query_results[query] = {
                        "chunks_found": len(result_chunks),
                        "search_time_ms": round(search_time * 1000, 2),
                        "top_chunks": result_chunks[:10],  # Top 10 for analysis
                        "success": len(result_chunks) > 0,
                        "top_score": float(retrieval_scores[0]) if retrieval_scores else 0.0
                    }
                    
                    status = f"✅ {len(result_chunks)} chunks" if result_chunks else "❌ 0 chunks"
                    top_score = f"(top: {retrieval_scores[0]:.3f})" if retrieval_scores else ""
                    print(f"      {status} {top_score} in {search_time*1000:.1f}ms")
                    
                except Exception as e:
                    print(f"      ❌ Query failed: {e}")
                    query_results[query] = {
                        "chunks_found": 0,
                        "error": str(e),
                        "success": False
                    }
            
            return query_results
            
        except Exception as e:
            print(f"❌ Vector retrieval test failed: {e}")
            return {}
    
    def analyze_content_relevance(self, query_results: Dict[str, Any], query_category: str) -> Dict[str, Any]:
        """Analyze if retrieved content contains expected information"""
        if query_category not in self.content_expectations:
            return {"analysis": "No expectations defined for this category"}
        
        expectations = self.content_expectations[query_category]
        expected_content = expectations["expected_content"]
        
        relevance_analysis = {
            "category": query_category,
            "expected_content": expected_content,
            "success_criteria": expectations["success_criteria"],
            "query_analysis": {},
            "overall_success_rate": 0.0,
            "content_found_rate": 0.0
        }
        
        successful_queries = 0
        content_found_queries = 0
        
        for query, results in query_results.items():
            if not results.get("success", False):
                relevance_analysis["query_analysis"][query] = {
                    "success": False,
                    "content_found": [],
                    "assessment": "NO_RETRIEVAL"
                }
                continue
            
            # Check if any expected content appears in retrieved chunks
            content_found = []
            for chunk in results.get("top_chunks", []):
                chunk_content = chunk.get("content_preview", "").lower()
                for expected in expected_content:
                    if expected.lower() in chunk_content:
                        content_found.append(expected)
            
            content_found = list(set(content_found))  # Remove duplicates
            
            if content_found:
                content_found_queries += 1
                if len(content_found) >= 2:  # Good relevance if 2+ expected terms found
                    successful_queries += 1
                    assessment = "GOOD_RELEVANCE"
                else:
                    assessment = "PARTIAL_RELEVANCE"
            else:
                assessment = "POOR_RELEVANCE"
            
            relevance_analysis["query_analysis"][query] = {
                "success": len(content_found) > 0,
                "content_found": content_found,
                "assessment": assessment,
                "chunks_retrieved": results.get("chunks_found", 0),
                "top_score": results.get("top_score", 0.0)
            }
        
        total_queries = len(query_results)
        relevance_analysis["overall_success_rate"] = successful_queries / total_queries if total_queries > 0 else 0.0
        relevance_analysis["content_found_rate"] = content_found_queries / total_queries if total_queries > 0 else 0.0
        
        return relevance_analysis
    
    def analyze_natural_language_performance(self, query_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze natural language query performance without hardcoded expectations"""
        total_queries = len(query_results)
        successful_queries = 0
        
        performance_analysis = {
            "total_queries": total_queries,
            "query_analysis": {},
            "overall_success_rate": 0.0
        }
        
        for query, results in query_results.items():
            success = results.get("success", False) and results.get("chunks_found", 0) > 0
            if success:
                successful_queries += 1
            
            performance_analysis["query_analysis"][query] = {
                "success": success,
                "chunks_found": results.get("chunks_found", 0),
                "top_score": results.get("top_score", 0.0),
                "assessment": "SUCCESS" if success else "FAILED"
            }
        
        performance_analysis["overall_success_rate"] = successful_queries / total_queries if total_queries > 0 else 0.0
        
        return performance_analysis
    
    def test_single_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single embedding model with comprehensive retrieval evaluation"""
        model_name = model_config["name"]
        model_path = model_config["model_path"]
        
        print(f"\n{'='*80}")
        print(f"TESTING MODEL: {model_name}")
        print(f"Path: {model_path}")
        print(f"Category: {model_config['category']}")
        print(f"{'='*80}")
        
        model_result = {
            "name": model_name,
            "model_path": model_path,
            "category": model_config["category"],
            "index_rebuild_success": False,
            "index_rebuild_time_seconds": 0,
            "retrieval_tests": {},
            "relevance_analysis": {},
            "overall_performance": {},
            "error_details": None
        }
        
        try:
            # Step 1: Rebuild index with this model
            rebuild_start = time.time()
            rebuild_success = self.clear_and_rebuild_index(model_path)
            rebuild_time = time.time() - rebuild_start
            
            model_result["index_rebuild_success"] = rebuild_success
            model_result["index_rebuild_time_seconds"] = round(rebuild_time, 2)
            
            if not rebuild_success:
                model_result["error_details"] = "Failed to rebuild index"
                return model_result
            
            print(f"✅ Index rebuilt in {rebuild_time:.1f}s")
            
            # Step 2: Test retrieval on all query categories
            # Create DocumentProcessor instance with the newly built index
            doc_processor = DocumentProcessor(model_path)
            # Load the index that was just built
            doc_processor.load_index()
            
            print(f"\n🧠 Testing natural language comprehension...")
            natural_results = self.test_pure_vector_retrieval(self.natural_test_queries, model_name, doc_processor)
            model_result["retrieval_tests"]["natural_language"] = natural_results
            
            # Step 3: Analyze content relevance
            print(f"\n📊 Analyzing content relevance...")
            
            model_result["relevance_analysis"]["natural_language"] = self.analyze_natural_language_performance(
                natural_results
            )
            
            # Step 4: Calculate overall performance
            model_result["overall_performance"] = self.calculate_model_performance(model_result)
            
            print(f"\n📈 Overall Performance Score: {model_result['overall_performance'].get('overall_score', 0):.3f}")
            
        except Exception as e:
            model_result["error_details"] = str(e)
            print(f"❌ Model test failed: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
        
        return model_result
    
    def calculate_model_performance(self, model_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics for a model"""
        if not model_result.get("index_rebuild_success", False):
            return {"overall_score": 0.0, "assessment": "FAILED"}
        
        relevance_data = model_result.get("relevance_analysis", {})
        
        # Natural language comprehension score
        natural_performance = relevance_data.get("natural_language", {})
        overall_score = natural_performance.get("overall_success_rate", 0.0)
        
        # Performance assessment based on natural language understanding
        if overall_score >= 0.8:
            assessment = "EXCELLENT"
        elif overall_score >= 0.6:
            assessment = "GOOD"
        elif overall_score >= 0.4:
            assessment = "PARTIAL"
        else:
            assessment = "POOR"
        
        return {
            "overall_score": overall_score,
            "assessment": assessment,
            "natural_language_comprehension": overall_score,
            "can_handle_natural_queries": overall_score >= 0.75
        }
    
    def run_comprehensive_test(self, priority_filter: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive embedding model and retrieval test"""
        print("🚀 COMPREHENSIVE EMBEDDING MODEL + RETRIEVAL TEST")
        print("=" * 80)
        print(f"Document: {self.test_document}")
        print(f"Total queries: {len(self.natural_test_queries)}")
        print(f"Test approach: Pure vector search with top_k=50")
        print("=" * 80)
        
        # Filter models if specified
        models_to_test = self.models_to_test
        if priority_filter:
            models_to_test = [m for m in self.models_to_test if m["priority"] in priority_filter]
            print(f"🎯 Testing priority levels: {priority_filter}")
            print(f"Models to test: {len(models_to_test)}")
        
        # Test each model
        for i, model_config in enumerate(models_to_test, 1):
            print(f"\n🔄 Progress: {i}/{len(models_to_test)} models")
            
            model_result = self.test_single_model(model_config)
            self.results["models_tested"].append(model_result)
            
            # Brief summary for this model
            performance = model_result.get("overall_performance", {})
            score = performance.get("overall_score", 0)
            assessment = performance.get("assessment", "UNKNOWN")
            print(f"📊 {model_config['name']}: {score:.3f} ({assessment})")
        
        # Generate final analysis
        self.generate_final_analysis()
        
        return self.results
    
    def generate_final_analysis(self):
        """Generate final analysis and recommendations"""
        successful_models = [m for m in self.results["models_tested"] if m.get("index_rebuild_success", False)]
        
        if not successful_models:
            self.results["final_analysis"] = {
                "status": "NO_SUCCESSFUL_MODELS",
                "message": "No models completed testing successfully"
            }
            return
        
        # Rank models by overall performance
        ranked_models = sorted(
            successful_models, 
            key=lambda x: x.get("overall_performance", {}).get("overall_score", 0), 
            reverse=True
        )
        
        # Find best performers
        best_overall = ranked_models[0] if ranked_models else None
        best_phone = max(
            successful_models,
            key=lambda x: x.get("relevance_analysis", {}).get("phone_numbers", {}).get("content_found_rate", 0)
        )
        best_text = max(
            successful_models,
            key=lambda x: x.get("relevance_analysis", {}).get("text_messaging", {}).get("content_found_rate", 0)
        )
        
        # Find current baseline performance
        current_baseline = next(
            (m for m in successful_models if m["name"] == "e5-large-v2"), 
            None
        )
        
        self.results["model_rankings"] = [
            {
                "rank": i + 1,
                "name": model["name"],
                "overall_score": model.get("overall_performance", {}).get("overall_score", 0),
                "assessment": model.get("overall_performance", {}).get("assessment", "UNKNOWN"),
                "category": model["category"],
                "can_handle_natural_queries": model.get("overall_performance", {}).get("can_handle_natural_queries", False)
            }
            for i, model in enumerate(ranked_models)
        ]
        
        self.results["best_performers"] = {
            "best_overall": {
                "name": best_overall["name"] if best_overall else "None",
                "score": best_overall.get("overall_performance", {}).get("overall_score", 0) if best_overall else 0,
                "assessment": best_overall.get("overall_performance", {}).get("assessment", "UNKNOWN") if best_overall else "UNKNOWN"
            },
            "best_phone_retrieval": {
                "name": best_phone["name"],
                "success_rate": best_phone.get("relevance_analysis", {}).get("phone_numbers", {}).get("content_found_rate", 0)
            },
            "best_text_retrieval": {
                "name": best_text["name"],
                "success_rate": best_text.get("relevance_analysis", {}).get("text_messaging", {}).get("content_found_rate", 0)
            }
        }
        
        if current_baseline and best_overall:
            baseline_score = current_baseline.get("overall_performance", {}).get("overall_score", 0)
            best_score = best_overall.get("overall_performance", {}).get("overall_score", 0)
            improvement = best_score - baseline_score
            
            self.results["improvement_analysis"] = {
                "current_model": "e5-large-v2",
                "current_score": baseline_score,
                "best_model": best_overall["name"],
                "best_score": best_score,
                "improvement": improvement,
                "percentage_improvement": (improvement / baseline_score * 100) if baseline_score > 0 else 0
            }
    
    def save_results(self, filename: str = None) -> Path:
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_embedding_retrieval_test_{timestamp}.json"
        
        results_file = self.project_root / filename
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        return results_file
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        
        print(f"📊 Models tested: {len(self.results['models_tested'])}")
        print(f"📝 Total queries per model: {len(self.natural_test_queries)}")
        print(f"📄 Test document: {self.test_document}")
        
        if self.results.get("model_rankings"):
            print(f"\n🏆 TOP PERFORMERS:")
            for rank_info in self.results["model_rankings"][:5]:
                critical_status = "✅ HANDLES NATURAL LANGUAGE" if rank_info["can_handle_natural_queries"] else "❌ Natural language struggles"
                print(f"   {rank_info['rank']}. {rank_info['name']} - {rank_info['overall_score']:.3f} ({rank_info['assessment']}) - {critical_status}")
        
        if self.results.get("best_performers"):
            performers = self.results["best_performers"]
            print(f"\n🎯 NATURAL LANGUAGE COMPREHENSION:")
            best_nl = performers['best_overall']
            print(f"   🧠 Best overall: {best_nl['name']} ({best_nl['score']:.1%} success rate)")
        
        if self.results.get("improvement_analysis"):
            imp = self.results["improvement_analysis"]
            print(f"\n📈 IMPROVEMENT VS CURRENT MODEL:")
            print(f"   Current ({imp['current_model']}): {imp['current_score']:.3f}")
            print(f"   Best ({imp['best_model']}): {imp['best_score']:.3f}")
            print(f"   Improvement: +{imp['improvement']:.3f} ({imp['percentage_improvement']:.1f}%)")


def main():
    """Main test execution"""
    tester = ComprehensiveEmbeddingRetrievalTester()
    
    # Test ALL local embedding models
    print("🚀 Testing ALL local embedding models...")
    results = tester.run_comprehensive_test()
    
    # Print summary
    tester.print_summary()
    
    # Save results
    results_file = tester.save_results()
    
    return results, results_file


if __name__ == "__main__":
    results, results_file = main()
