"""
Semantic Relevance Evaluation for Embedding Models
Tests actual content understanding and domain knowledge bridging
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
from core.config import retrieval_config


class SemanticRelevanceEvaluator:
    def __init__(self):
        """Initialize semantic relevance evaluator"""
        self.project_root = project_root
        self.test_document = "./KTI_Dispatch_Guide.pdf"
        
        # Ground truth content mapping - what should be retrieved for each query
        self.ground_truth = {
            "phone_number": {
                "query": "What is our main phone number?",
                "required_content": ["480-999-3046"],
                "relevant_keywords": ["phone", "number", "contact", "main", "company"],
                "context_keywords": ["metro", "dispatch", "office"],
                "expected_answer": "480-999-3046",
                "semantic_variations": [
                    "company contact number",
                    "main office phone",
                    "dispatch center number"
                ]
            },
            "text_messaging": {
                "query": "How do I handle text messages?",
                "required_content": ["SMS", "text message", "Freshdesk"],
                "relevant_keywords": ["text", "message", "SMS", "handle", "process", "workflow"],
                "context_keywords": ["ticket", "response", "customer", "procedure"],
                "expected_answer": "text message handling workflow with Freshdesk",
                "semantic_variations": [
                    "SMS support process",
                    "text message workflow",
                    "handling customer texts"
                ]
            },
            "customer_concern": {
                "query": "How do I create a customer concern?",
                "required_content": ["customer concern", "Freshdesk", "ticket"],
                "relevant_keywords": ["customer", "concern", "create", "complaint", "issue"],
                "context_keywords": ["ticket", "Freshdesk", "escalate", "helpdesk"],
                "expected_answer": "customer concern creation process in Freshdesk",
                "semantic_variations": [
                    "complaint handling process",
                    "customer issue escalation",
                    "concern ticket creation"
                ]
            },
            "licensing": {
                "query": "Are we licensed and insured?",
                "required_content": ["licensed", "insured", "license", "insurance"],
                "relevant_keywords": ["licensed", "insured", "license", "insurance", "certification"],
                "context_keywords": ["company", "business", "legal", "coverage"],
                "expected_answer": "licensing and insurance status information",
                "semantic_variations": [
                    "company licensing status",
                    "business insurance coverage",
                    "legal compliance information"
                ]
            }
        }
        
        # Domain understanding test queries
        self.domain_tests = {
            "semantic_bridging": {
                "query": "emergency contact information",
                "should_find": "phone_number",  # Should bridge to phone number content
                "description": "Tests if model bridges 'emergency contact' to phone numbers"
            },
            "process_understanding": {
                "query": "complaint handling procedure",
                "should_find": "customer_concern",  # Should find customer concern process
                "description": "Tests if model understands complaint = customer concern"
            },
            "terminology_mapping": {
                "query": "SMS support workflow",
                "should_find": "text_messaging",  # Should find text message procedures
                "description": "Tests if model maps SMS to text messaging"
            }
        }
        
        # Comprehensive local embedding models to test
        self.models_to_test = [
            # E5 family (Microsoft)
            {"name": "e5-large-v2", "model_path": "intfloat/e5-large-v2", "category": "E5 Family"},
            {"name": "e5-base-v2", "model_path": "intfloat/e5-base-v2", "category": "E5 Family"},
            {"name": "e5-small-v2", "model_path": "intfloat/e5-small-v2", "category": "E5 Family"},
            
            # BGE family (BAAI)
            {"name": "bge-large-en-v1.5", "model_path": "BAAI/bge-large-en-v1.5", "category": "BGE Family"},
            {"name": "bge-base-en-v1.5", "model_path": "BAAI/bge-base-en-v1.5", "category": "BGE Family"},
            {"name": "bge-small-en-v1.5", "model_path": "BAAI/bge-small-en-v1.5", "category": "BGE Family"},
            
            # Sentence Transformers family
            {"name": "all-mpnet-base-v2", "model_path": "sentence-transformers/all-mpnet-base-v2", "category": "SentenceTransformers"},
            {"name": "all-MiniLM-L12-v2", "model_path": "sentence-transformers/all-MiniLM-L12-v2", "category": "SentenceTransformers"},
            {"name": "all-MiniLM-L6-v2", "model_path": "sentence-transformers/all-MiniLM-L6-v2", "category": "SentenceTransformers"},
            
            # Instruction-following models
            {"name": "instructor-large", "model_path": "hkunlp/instructor-large", "category": "Instruction-Following"},
            {"name": "instructor-base", "model_path": "hkunlp/instructor-base", "category": "Instruction-Following"},
            
            # Retrieval-specialized models
            {"name": "gtr-t5-large", "model_path": "sentence-transformers/gtr-t5-large", "category": "Retrieval-Specialized"},
            {"name": "msmarco-distilbert-base-v4", "model_path": "sentence-transformers/msmarco-distilbert-base-v4", "category": "Retrieval-Specialized"},
            {"name": "multi-qa-mpnet-base-dot-v1", "model_path": "sentence-transformers/multi-qa-mpnet-base-dot-v1", "category": "Retrieval-Specialized"}
        ]
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "semantic_relevance_evaluation",
            "models_tested": [],
            "semantic_scores": {},
            "domain_understanding": {},
            "final_rankings": {}
        }

    def clear_and_rebuild_index(self, model_path: str) -> bool:
        """Clear existing index and rebuild with specified model"""
        try:
            print(f"🗑️ Clearing existing index...")
            
            # Clear data/index directory
            index_dir = self.project_root / "data" / "index"
            if index_dir.exists():
                shutil.rmtree(index_dir)
                
            # Clear current_index directory
            current_index_dir = self.project_root / "current_index"
            if current_index_dir.exists():
                shutil.rmtree(current_index_dir)
            
            print(f"🏗️ Rebuilding index with model: {model_path}")
            
            # Update config to use new model
            retrieval_config.embedding_model = model_path
            
            # Initialize document processor and rebuild index
            doc_processor = DocumentProcessor(model_path)
            
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
                print(f"✅ Index rebuilt successfully with {model_path}")
                return True
            else:
                print(f"❌ Failed to rebuild index with {model_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error rebuilding index: {e}")
            return False

    def analyze_content_relevance(self, chunks: List[Dict[str, Any]], query_key: str) -> Dict[str, Any]:
        """Analyze semantic relevance of retrieved chunks"""
        ground_truth = self.ground_truth[query_key]
        required_content = ground_truth["required_content"]
        relevant_keywords = ground_truth["relevant_keywords"]
        context_keywords = ground_truth["context_keywords"]
        
        analysis = {
            "query": ground_truth["query"],
            "expected_answer": ground_truth["expected_answer"],
            "chunks_analyzed": len(chunks),
            "required_content_found": [],
            "relevant_keyword_matches": 0,
            "context_keyword_matches": 0,
            "semantic_relevance_score": 0.0,
            "content_accuracy_score": 0.0,
            "ranking_quality_score": 0.0,
            "top_chunks_analysis": []
        }
        
        total_relevant_keywords = len(relevant_keywords)
        total_context_keywords = len(context_keywords)
        
        for i, chunk in enumerate(chunks[:10]):  # Analyze top 10 chunks
            chunk_text = chunk.get("text", "").lower()
            chunk_score = chunk.get("score", 0.0)
            
            # Check for required content
            required_found = []
            for required in required_content:
                if required.lower() in chunk_text:
                    required_found.append(required)
                    if required not in analysis["required_content_found"]:
                        analysis["required_content_found"].append(required)
            
            # Count keyword matches
            relevant_matches = sum(1 for keyword in relevant_keywords if keyword.lower() in chunk_text)
            context_matches = sum(1 for keyword in context_keywords if keyword.lower() in chunk_text)
            
            analysis["relevant_keyword_matches"] += relevant_matches
            analysis["context_keyword_matches"] += context_matches
            
            # Calculate chunk relevance score
            chunk_relevance = 0.0
            if required_found:
                chunk_relevance += 0.5  # High value for required content
            chunk_relevance += (relevant_matches / total_relevant_keywords) * 0.3
            chunk_relevance += (context_matches / total_context_keywords) * 0.2
            
            # Ranking quality - higher relevance should appear earlier
            position_weight = 1.0 / (i + 1)  # Earlier positions get higher weight
            
            analysis["top_chunks_analysis"].append({
                "rank": i + 1,
                "similarity_score": float(chunk_score),
                "required_content": required_found,
                "relevant_keywords": relevant_matches,
                "context_keywords": context_matches,
                "chunk_relevance": chunk_relevance,
                "position_weight": position_weight,
                "content_preview": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            })
        
        # Calculate overall scores
        total_required = len(required_content)
        found_required = len(analysis["required_content_found"])
        
        # Content accuracy: did we find the required content?
        analysis["content_accuracy_score"] = found_required / total_required if total_required > 0 else 0.0
        
        # Semantic relevance: keyword matching
        max_relevant_matches = total_relevant_keywords * len(chunks[:10])
        max_context_matches = total_context_keywords * len(chunks[:10])
        
        relevance_score = 0.0
        if max_relevant_matches > 0:
            relevance_score += (analysis["relevant_keyword_matches"] / max_relevant_matches) * 0.7
        if max_context_matches > 0:
            relevance_score += (analysis["context_keyword_matches"] / max_context_matches) * 0.3
        
        analysis["semantic_relevance_score"] = min(relevance_score, 1.0)
        
        # Ranking quality: relevant content should appear early
        weighted_relevance = sum(
            chunk["chunk_relevance"] * chunk["position_weight"]
            for chunk in analysis["top_chunks_analysis"]
        )
        max_possible_weighted = sum(1.0 / (i + 1) for i in range(len(chunks[:10])))
        analysis["ranking_quality_score"] = weighted_relevance / max_possible_weighted if max_possible_weighted > 0 else 0.0
        
        # Overall semantic score (weighted combination)
        analysis["overall_semantic_score"] = (
            analysis["content_accuracy_score"] * 0.5 +
            analysis["semantic_relevance_score"] * 0.3 +
            analysis["ranking_quality_score"] * 0.2
        )
        
        return analysis

    def test_domain_understanding(self, doc_processor: DocumentProcessor, model_name: str) -> Dict[str, Any]:
        """Test domain knowledge bridging capabilities"""
        domain_results = {}
        
        for test_key, test_config in self.domain_tests.items():
            query = test_config["query"]
            should_find = test_config["should_find"]
            description = test_config["description"]
            
            print(f"   Testing domain understanding: '{query}'")
            
            try:
                # Perform search
                results = doc_processor.search(query, top_k=10)
                
                if not results:
                    domain_results[test_key] = {
                        "query": query,
                        "description": description,
                        "should_find": should_find,
                        "success": False,
                        "reason": "No results returned",
                        "semantic_bridging_score": 0.0
                    }
                    continue
                
                # Analyze if it found content related to what it should find
                ground_truth = self.ground_truth[should_find]
                required_content = ground_truth["required_content"]
                relevant_keywords = ground_truth["relevant_keywords"]
                
                found_required = 0
                found_relevant = 0
                
                for chunk in results[:5]:  # Check top 5 results
                    chunk_text = chunk.get("text", "").lower()
                    
                    # Check for required content
                    for required in required_content:
                        if required.lower() in chunk_text:
                            found_required += 1
                    
                    # Check for relevant keywords
                    for keyword in relevant_keywords:
                        if keyword.lower() in chunk_text:
                            found_relevant += 1
                
                # Calculate semantic bridging score
                max_required = len(required_content) * 5  # Top 5 chunks
                max_relevant = len(relevant_keywords) * 5
                
                bridging_score = 0.0
                if max_required > 0:
                    bridging_score += (found_required / max_required) * 0.7
                if max_relevant > 0:
                    bridging_score += (found_relevant / max_relevant) * 0.3
                
                domain_results[test_key] = {
                    "query": query,
                    "description": description,
                    "should_find": should_find,
                    "success": bridging_score > 0.3,  # Threshold for success
                    "semantic_bridging_score": bridging_score,
                    "required_content_matches": found_required,
                    "relevant_keyword_matches": found_relevant
                }
                
                status = "✅" if bridging_score > 0.3 else "❌"
                print(f"      {status} Bridging score: {bridging_score:.3f}")
                
            except Exception as e:
                domain_results[test_key] = {
                    "query": query,
                    "description": description,
                    "should_find": should_find,
                    "success": False,
                    "reason": str(e),
                    "semantic_bridging_score": 0.0
                }
                print(f"      ❌ Error: {e}")
        
        return domain_results

    def test_single_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Test semantic relevance for a single embedding model"""
        model_name = model_config["name"]
        model_path = model_config["model_path"]
        
        print(f"\n{'='*80}")
        print(f"SEMANTIC EVALUATION: {model_name}")
        print(f"Path: {model_path}")
        print(f"Category: {model_config['category']}")
        print(f"{'='*80}")
        
        model_result = {
            "name": model_name,
            "model_path": model_path,
            "category": model_config["category"],
            "index_rebuild_success": False,
            "semantic_analysis": {},
            "domain_understanding": {},
            "overall_scores": {},
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
            
            # Step 2: Create DocumentProcessor and load index
            doc_processor = DocumentProcessor(model_path)
            doc_processor.load_index()
            
            # Step 3: Test semantic understanding for each query
            print(f"\n🧠 Testing semantic understanding...")
            
            for query_key, query_config in self.ground_truth.items():
                query = query_config["query"]
                print(f"   Testing: '{query}'")
                
                # Perform search
                results = doc_processor.search(query, top_k=20)
                
                if results:
                    # Analyze semantic relevance
                    semantic_analysis = self.analyze_content_relevance(results, query_key)
                    model_result["semantic_analysis"][query_key] = semantic_analysis
                    
                    score = semantic_analysis["overall_semantic_score"]
                    accuracy = semantic_analysis["content_accuracy_score"]
                    print(f"      Semantic Score: {score:.3f}, Content Accuracy: {accuracy:.3f}")
                else:
                    model_result["semantic_analysis"][query_key] = {
                        "query": query,
                        "overall_semantic_score": 0.0,
                        "content_accuracy_score": 0.0,
                        "error": "No results returned"
                    }
                    print(f"      ❌ No results returned")
            
            # Step 4: Test domain understanding
            print(f"\n🌉 Testing domain knowledge bridging...")
            domain_results = self.test_domain_understanding(doc_processor, model_name)
            model_result["domain_understanding"] = domain_results
            
            # Step 5: Calculate overall scores
            model_result["overall_scores"] = self.calculate_overall_scores(model_result)
            
            overall_score = model_result["overall_scores"]["combined_score"]
            print(f"\n📈 Overall Semantic Performance: {overall_score:.3f}")
            
        except Exception as e:
            model_result["error_details"] = str(e)
            print(f"❌ Model test failed: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
        
        return model_result

    def calculate_overall_scores(self, model_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance scores"""
        semantic_analysis = model_result.get("semantic_analysis", {})
        domain_understanding = model_result.get("domain_understanding", {})
        
        # Calculate average semantic scores
        semantic_scores = []
        accuracy_scores = []
        
        for query_analysis in semantic_analysis.values():
            if "overall_semantic_score" in query_analysis:
                semantic_scores.append(query_analysis["overall_semantic_score"])
            if "content_accuracy_score" in query_analysis:
                accuracy_scores.append(query_analysis["content_accuracy_score"])
        
        avg_semantic = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
        
        # Calculate domain understanding score
        domain_scores = [test.get("semantic_bridging_score", 0.0) for test in domain_understanding.values()]
        avg_domain = sum(domain_scores) / len(domain_scores) if domain_scores else 0.0
        
        # Combined score (weighted)
        combined_score = (avg_semantic * 0.5) + (avg_accuracy * 0.3) + (avg_domain * 0.2)
        
        # Performance assessment
        if combined_score >= 0.8:
            assessment = "EXCELLENT"
        elif combined_score >= 0.6:
            assessment = "GOOD"
        elif combined_score >= 0.4:
            assessment = "FAIR"
        elif combined_score >= 0.2:
            assessment = "POOR"
        else:
            assessment = "FAILED"
        
        return {
            "avg_semantic_score": avg_semantic,
            "avg_accuracy_score": avg_accuracy,
            "avg_domain_score": avg_domain,
            "combined_score": combined_score,
            "assessment": assessment,
            "strong_domain_understanding": avg_domain >= 0.6,
            "accurate_content_retrieval": avg_accuracy >= 0.7,
            "semantic_relevance_quality": avg_semantic >= 0.6
        }

    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete semantic relevance evaluation"""
        print("🚀 SEMANTIC RELEVANCE EVALUATION FOR EMBEDDING MODELS")
        print("=" * 80)
        print(f"Document: {self.test_document}")
        print(f"Ground truth queries: {len(self.ground_truth)}")
        print(f"Domain understanding tests: {len(self.domain_tests)}")
        print(f"Models to evaluate: {len(self.models_to_test)}")
        print("=" * 80)
        
        # Test each model
        for i, model_config in enumerate(self.models_to_test, 1):
            print(f"\n🔄 Progress: {i}/{len(self.models_to_test)} models")
            
            model_result = self.test_single_model(model_config)
            self.results["models_tested"].append(model_result)
            
            # Brief summary
            if model_result.get("overall_scores"):
                scores = model_result["overall_scores"]
                combined = scores.get("combined_score", 0)
                assessment = scores.get("assessment", "UNKNOWN")
                print(f"📊 {model_config['name']}: {combined:.3f} ({assessment})")
        
        # Generate final rankings
        self.generate_final_rankings()
        
        return self.results

    def generate_final_rankings(self):
        """Generate final performance rankings"""
        successful_models = [
            m for m in self.results["models_tested"] 
            if m.get("index_rebuild_success", False) and m.get("overall_scores")
        ]
        
        if not successful_models:
            self.results["final_rankings"] = {"error": "No successful models"}
            return
        
        # Sort by combined score
        ranked_models = sorted(
            successful_models,
            key=lambda x: x.get("overall_scores", {}).get("combined_score", 0),
            reverse=True
        )
        
        self.results["final_rankings"] = {
            "by_overall_score": [
                {
                    "rank": i + 1,
                    "name": model["name"],
                    "category": model["category"],
                    "combined_score": model["overall_scores"]["combined_score"],
                    "assessment": model["overall_scores"]["assessment"],
                    "semantic_score": model["overall_scores"]["avg_semantic_score"],
                    "accuracy_score": model["overall_scores"]["avg_accuracy_score"],
                    "domain_score": model["overall_scores"]["avg_domain_score"],
                    "strong_domain_understanding": model["overall_scores"]["strong_domain_understanding"],
                    "accurate_content_retrieval": model["overall_scores"]["accurate_content_retrieval"]
                }
                for i, model in enumerate(ranked_models)
            ]
        }
        
        # Find best performers by category
        best_overall = ranked_models[0] if ranked_models else None
        best_accuracy = max(successful_models, key=lambda x: x.get("overall_scores", {}).get("avg_accuracy_score", 0))
        best_domain = max(successful_models, key=lambda x: x.get("overall_scores", {}).get("avg_domain_score", 0))
        
        self.results["best_performers"] = {
            "overall_best": {
                "name": best_overall["name"] if best_overall else "None",
                "score": best_overall["overall_scores"]["combined_score"] if best_overall else 0,
                "category": best_overall["category"] if best_overall else "None"
            },
            "best_accuracy": {
                "name": best_accuracy["name"],
                "score": best_accuracy["overall_scores"]["avg_accuracy_score"]
            },
            "best_domain_understanding": {
                "name": best_domain["name"],
                "score": best_domain["overall_scores"]["avg_domain_score"]
            }
        }

    def save_results(self, filename: str = None) -> Path:
        """Save evaluation results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"semantic_relevance_evaluation_{timestamp}.json"
        
        results_file = self.project_root / filename
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        return results_file

    def print_summary(self):
        """Print comprehensive evaluation summary"""
        print("\n" + "=" * 80)
        print("SEMANTIC RELEVANCE EVALUATION SUMMARY")
        print("=" * 80)
        
        rankings = self.results.get("final_rankings", {}).get("by_overall_score", [])
        
        if rankings:
            print(f"\n🏆 TOP SEMANTIC PERFORMERS:")
            for rank_info in rankings[:10]:  # Top 10
                domain_status = "🌉 Strong Domain Understanding" if rank_info["strong_domain_understanding"] else "❌ Weak Domain Understanding"
                accuracy_status = "🎯 Accurate Retrieval" if rank_info["accurate_content_retrieval"] else "❌ Poor Accuracy"
                
                print(f"   {rank_info['rank']}. {rank_info['name']} - {rank_info['combined_score']:.3f} ({rank_info['assessment']})")
                print(f"      Category: {rank_info['category']}")
                print(f"      Semantic: {rank_info['semantic_score']:.3f} | Accuracy: {rank_info['accuracy_score']:.3f} | Domain: {rank_info['domain_score']:.3f}")
                print(f"      {domain_status} | {accuracy_status}")
                print()
        
        best_performers = self.results.get("best_performers", {})
        if best_performers:
            print(f"🎯 SPECIALIZED BEST PERFORMERS:")
            print(f"   🏆 Overall Best: {best_performers['overall_best']['name']} ({best_performers['overall_best']['score']:.3f})")
            print(f"   🎯 Best Accuracy: {best_performers['best_accuracy']['name']} ({best_performers['best_accuracy']['score']:.3f})")
            print(f"   🌉 Best Domain Understanding: {best_performers['best_domain_understanding']['name']} ({best_performers['best_domain_understanding']['score']:.3f})")


def main():
    """Main evaluation execution"""
    evaluator = SemanticRelevanceEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    results_file = evaluator.save_results()
    
    return results, results_file


if __name__ == "__main__":
    results, results_file = main()
