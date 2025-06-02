"""
End-to-End Pipeline Validation Test
Tests complete system: Retrieval → LLM Extraction → Formatted Answer
Validates production readiness for LLM-assisted extraction
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
from llm.pipeline.answer_pipeline import AnswerPipeline
from llm.services.llm_service import LLMService
from llm.prompt_generator import PromptGenerator
from core.config import retrieval_config, app_config


class EndToEndPipelineValidator:
    def __init__(self):
        """Initialize end-to-end pipeline validator"""
        self.project_root = project_root
        self.test_document = "./KTI_Dispatch_Guide.pdf"
        
        # Test queries with expected answer validation
        self.test_queries = {
            "phone_number": {
                "query": "What is our main phone number?",
                "expected_content": ["480-999-3046"],
                "validation_criteria": [
                    "Contains the phone number 480-999-3046",
                    "Clearly identifies it as the main/company number",
                    "Response is formatted clearly"
                ],
                "critical": True  # Must work for production
            },
            "text_messaging_natural": {
                "query": "How do I handle text messages from customers?",
                "expected_content": ["SMS", "text message", "Freshdesk", "ticket"],
                "validation_criteria": [
                    "Explains the text message workflow",
                    "Mentions Freshdesk integration",
                    "Provides step-by-step process",
                    "Mentions ticket creation"
                ],
                "critical": True
            },
            "customer_concern_natural": {
                "query": "How do I create a customer concern?",
                "expected_content": ["customer concern", "Freshdesk", "ticket"],
                "validation_criteria": [
                    "Explains customer concern creation process",
                    "Mentions Freshdesk system",
                    "Provides clear steps",
                    "Explains when to use customer concerns"
                ],
                "critical": True
            },
            "licensing_natural": {
                "query": "Are we bonded and insured?",
                "expected_content": ["bonded", "insured"],
                "validation_criteria": [
                    "Provides bonding information",
                    "Provides insurance information",
                    "Clear yes/no answer if possible"
                ],
                "critical": False  # Nice to have
            },
            "emergency_contact": {
                "query": "What's our Riverview phone number?",
                "expected_content": ["813-400-2865", "phone", "contact"],
                "validation_criteria": [
                    "Provides phone number information",
                    "Recognizes 'Riverview phone number' relates to phone numbers",
                    "Clear and actionable response"
                ],
                "critical": True  # Test semantic bridging
            },
            "complaint_handling": {
                "query": "How do I handle customer complaints?",
                "expected_content": ["customer concern", "complaint", "Freshdesk"],
                "validation_criteria": [
                    "Connects complaints to customer concern process",
                    "Provides actionable steps",
                    "Mentions appropriate system (Freshdesk)"
                ],
                "critical": True  # Test semantic bridging
            }
        }
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "end_to_end_pipeline_validation",
            "system_config": {
                "embedding_model": retrieval_config.embedding_model,
                "enhanced_mode": retrieval_config.enhanced_mode,
                "top_k": retrieval_config.top_k,
                "extraction_model": "gpt-4-turbo"
            },
            "query_results": {},
            "overall_performance": {},
            "production_readiness": {}
        }

    def setup_optimized_system(self) -> bool:
        """Setup system with optimized configuration"""
        try:
            print(f"🔧 Setting up optimized system...")
            print(f"   Embedding Model: {retrieval_config.embedding_model}")
            print(f"   Enhanced Mode: {retrieval_config.enhanced_mode}")
            print(f"   Top K: {retrieval_config.top_k}")
            
            # Clear existing index
            index_dir = self.project_root / "data" / "index"
            if index_dir.exists():
                shutil.rmtree(index_dir)
                print(f"   Cleared existing index")
                
            current_index_dir = self.project_root / "current_index"
            if current_index_dir.exists():
                shutil.rmtree(current_index_dir)
                print(f"   Cleared current index")
            
            # Initialize document processor with optimized config
            doc_processor = DocumentProcessor()
            
            # Process document
            test_doc_path = self.project_root / self.test_document
            if not test_doc_path.exists():
                print(f"❌ Test document not found: {test_doc_path}")
                return False
            
            print(f"📄 Processing document: {self.test_document}")
            start_time = time.time()
            index, chunks = doc_processor.process_documents([str(test_doc_path)])
            setup_time = time.time() - start_time
            
            if index is None or len(chunks) == 0:
                print(f"❌ Failed to process document")
                return False
            
            # Save index
            doc_processor.save_index()
            
            print(f"✅ System setup complete in {setup_time:.1f}s")
            print(f"   Chunks created: {len(chunks)}")
            print(f"   Index size: {index.ntotal if hasattr(index, 'ntotal') else 'Unknown'}")
            
            self.results["setup_metrics"] = {
                "setup_time_seconds": round(setup_time, 2),
                "chunks_created": len(chunks),
                "index_size": index.ntotal if hasattr(index, 'ntotal') else 0
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    def validate_answer_quality(self, query_key: str, answer: str, retrieval_time: float, llm_time: float) -> Dict[str, Any]:
        """Validate the quality and correctness of the final answer"""
        query_config = self.test_queries[query_key]
        expected_content = query_config["expected_content"]
        validation_criteria = query_config["validation_criteria"]
        is_critical = query_config["critical"]
        
        validation_result = {
            "query": query_config["query"],
            "answer": answer,
            "retrieval_time_ms": round(retrieval_time * 1000, 2),
            "llm_time_ms": round(llm_time * 1000, 2),
            "total_time_ms": round((retrieval_time + llm_time) * 1000, 2),
            "expected_content_found": [],
            "criteria_met": [],
            "criteria_failed": [],
            "content_score": 0.0,
            "criteria_score": 0.0,
            "overall_score": 0.0,
            "is_critical": is_critical,
            "production_ready": False
        }
        
        if not answer or answer.strip() == "":
            validation_result["overall_score"] = 0.0
            validation_result["criteria_failed"] = ["No answer generated"]
            return validation_result
        
        answer_lower = answer.lower()
        
        # Check for expected content
        for content in expected_content:
            if content.lower() in answer_lower:
                validation_result["expected_content_found"].append(content)
        
        content_score = len(validation_result["expected_content_found"]) / len(expected_content)
        validation_result["content_score"] = content_score
        
        # Manual criteria validation (simplified - in real system would use LLM judge)
        criteria_met = 0
        for criteria in validation_criteria:
            # Simple heuristic validation - could be enhanced with LLM evaluation
            criteria_keywords = criteria.lower().split()
            relevant_keywords = ["explain", "step", "process", "clear", "mention", "provide"]
            
            # Check if answer seems to address the criteria based on content and length
            if (len(answer) > 50 and  # Has substantial content
                any(content.lower() in answer_lower for content in expected_content) and  # Has expected content
                len(answer.split()) > 10):  # Has reasonable detail
                validation_result["criteria_met"].append(criteria)
                criteria_met += 1
            else:
                validation_result["criteria_failed"].append(criteria)
        
        criteria_score = criteria_met / len(validation_criteria) if validation_criteria else 0.0
        validation_result["criteria_score"] = criteria_score
        
        # Overall score (weighted)
        overall_score = (content_score * 0.6) + (criteria_score * 0.4)
        validation_result["overall_score"] = overall_score
        
        # Production readiness threshold
        validation_result["production_ready"] = (
            overall_score >= 0.7 and  # Good overall score
            content_score >= 0.5 and  # Has most expected content
            validation_result["total_time_ms"] < 10000  # Under 10 seconds
        )
        
        return validation_result

    def test_single_query(self, query_key: str) -> Dict[str, Any]:
        """Test a single query through the complete pipeline"""
        query_config = self.test_queries[query_key]
        query = query_config["query"]
        
        print(f"\n🔍 Testing: '{query}'")
        
        try:
            # Initialize pipeline components
            doc_processor = DocumentProcessor()
            doc_processor.load_index()
            
            # Initialize LLM service and prompt generator
            api_key = app_config.api_keys.get("openai")
            if not api_key:
                return {
                    "query": query,
                    "success": False,
                    "error": "OpenAI API key not configured",
                    "overall_score": 0.0,
                    "production_ready": False
                }
            
            llm_service = LLMService(api_key)
            prompt_generator = PromptGenerator()
            answer_pipeline = AnswerPipeline(llm_service, prompt_generator)
            
            # Step 1: Vector Retrieval
            retrieval_start = time.time()
            chunks = doc_processor.search(query, top_k=retrieval_config.top_k)
            retrieval_time = time.time() - retrieval_start
            
            print(f"   📊 Retrieved {len(chunks)} chunks in {retrieval_time*1000:.1f}ms")
            
            if not chunks:
                return {
                    "query": query,
                    "success": False,
                    "error": "No chunks retrieved",
                    "retrieval_time_ms": round(retrieval_time * 1000, 2)
                }
            
            # Step 2: LLM Extraction and Formatting
            llm_start = time.time()
            
            # Convert chunks to context string
            context = "\n\n".join([chunk.get("text", "") for chunk in chunks])
            
            # Generate answer using the pipeline
            answer_result = answer_pipeline.generate_answer(query, context)
            answer = answer_result.get("content", "")
            llm_time = time.time() - llm_start
            
            print(f"   🧠 LLM generated answer in {llm_time*1000:.1f}ms")
            print(f"   📝 Answer length: {len(answer)} characters")
            
            # Step 3: Validate Answer Quality
            validation_result = self.validate_answer_quality(query_key, answer, retrieval_time, llm_time)
            
            print(f"   ✅ Content Score: {validation_result['content_score']:.2f}")
            print(f"   ✅ Criteria Score: {validation_result['criteria_score']:.2f}")
            print(f"   ✅ Overall Score: {validation_result['overall_score']:.2f}")
            print(f"   🚀 Production Ready: {validation_result['production_ready']}")
            
            validation_result["success"] = True
            return validation_result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print(f"   🔍 Traceback: {traceback.format_exc()}")
            return {
                "query": query,
                "success": False,
                "error": str(e),
                "overall_score": 0.0,
                "production_ready": False
            }

    def run_validation(self) -> Dict[str, Any]:
        """Run complete end-to-end validation"""
        print("🚀 END-TO-END PIPELINE VALIDATION")
        print("=" * 80)
        print(f"Document: {self.test_document}")
        print(f"Embedding Model: {retrieval_config.embedding_model}")
        print(f"Test Queries: {len(self.test_queries)}")
        print("=" * 80)
        
        # Step 1: Setup optimized system
        setup_success = self.setup_optimized_system()
        if not setup_success:
            self.results["overall_performance"]["setup_failed"] = True
            return self.results
        
        # Step 2: Test each query
        for query_key, query_config in self.test_queries.items():
            print(f"\n🔄 Progress: {query_key}")
            
            result = self.test_single_query(query_key)
            self.results["query_results"][query_key] = result
        
        # Step 3: Calculate overall performance
        self.calculate_overall_performance()
        
        return self.results

    def calculate_overall_performance(self):
        """Calculate overall system performance and production readiness"""
        successful_queries = [r for r in self.results["query_results"].values() if r.get("success", False)]
        total_queries = len(self.test_queries)
        
        if not successful_queries:
            self.results["overall_performance"] = {
                "success_rate": 0.0,
                "production_ready": False,
                "critical_failures": total_queries
            }
            return
        
        # Success metrics
        success_rate = len(successful_queries) / total_queries
        
        # Score metrics
        overall_scores = [r.get("overall_score", 0.0) for r in successful_queries]
        avg_overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        
        # Performance metrics
        total_times = [r.get("total_time_ms", 0) for r in successful_queries]
        avg_response_time = sum(total_times) / len(total_times) if total_times else 0.0
        
        # Production readiness
        production_ready_queries = [r for r in successful_queries if r.get("production_ready", False)]
        production_ready_rate = len(production_ready_queries) / total_queries
        
        # Critical query analysis
        critical_queries = [k for k, v in self.test_queries.items() if v["critical"]]
        critical_results = [r for k, r in self.results["query_results"].items() if k in critical_queries and r.get("success", False)]
        critical_production_ready = [r for r in critical_results if r.get("production_ready", False)]
        
        critical_success_rate = len(critical_results) / len(critical_queries) if critical_queries else 1.0
        critical_production_rate = len(critical_production_ready) / len(critical_queries) if critical_queries else 1.0
        
        # Overall assessment
        production_ready = (
            critical_production_rate >= 0.8 and  # 80% of critical queries work
            production_ready_rate >= 0.67 and   # 67% of all queries work
            avg_response_time < 8000 and        # Under 8 seconds average
            avg_overall_score >= 0.6            # Good answer quality
        )
        
        self.results["overall_performance"] = {
            "success_rate": round(success_rate, 3),
            "avg_overall_score": round(avg_overall_score, 3),
            "avg_response_time_ms": round(avg_response_time, 2),
            "production_ready_rate": round(production_ready_rate, 3),
            "critical_success_rate": round(critical_success_rate, 3),
            "critical_production_rate": round(critical_production_rate, 3),
            "production_ready": production_ready
        }
        
        # Detailed production readiness assessment
        self.results["production_readiness"] = {
            "overall_assessment": "READY" if production_ready else "NOT_READY",
            "critical_queries_working": f"{len(critical_production_ready)}/{len(critical_queries)}",
            "response_time_acceptable": avg_response_time < 8000,
            "answer_quality_good": avg_overall_score >= 0.6,
            "recommendations": self.generate_recommendations()
        }

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        performance = self.results["overall_performance"]
        
        if performance["critical_production_rate"] < 0.8:
            recommendations.append("Critical queries failing - investigate retrieval quality for core dispatch tasks")
        
        if performance["avg_response_time_ms"] > 8000:
            recommendations.append("Response time too slow - consider optimizing LLM calls or retrieval")
        
        if performance["avg_overall_score"] < 0.6:
            recommendations.append("Answer quality below threshold - review LLM prompts and context formatting")
        
        if performance["production_ready_rate"] < 0.67:
            recommendations.append("Too many queries failing - need broader system improvements")
        
        if not recommendations:
            recommendations.append("System appears production-ready for LLM-assisted extraction")
        
        return recommendations

    def save_results(self, filename: str = None) -> Path:
        """Save validation results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"end_to_end_validation_{timestamp}.json"
        
        results_file = self.project_root / filename
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        return results_file

    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 80)
        print("END-TO-END PIPELINE VALIDATION SUMMARY")
        print("=" * 80)
        
        performance = self.results.get("overall_performance", {})
        readiness = self.results.get("production_readiness", {})
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Success Rate: {performance.get('success_rate', 0):.1%}")
        print(f"   Average Answer Quality: {performance.get('avg_overall_score', 0):.2f}/1.0")
        print(f"   Average Response Time: {performance.get('avg_response_time_ms', 0):.0f}ms")
        print(f"   Production Ready Rate: {performance.get('production_ready_rate', 0):.1%}")
        
        print(f"\n🎯 CRITICAL QUERIES:")
        print(f"   Success Rate: {performance.get('critical_success_rate', 0):.1%}")
        print(f"   Production Ready: {performance.get('critical_production_rate', 0):.1%}")
        print(f"   Working: {readiness.get('critical_queries_working', 'Unknown')}")
        
        assessment = readiness.get('overall_assessment', 'UNKNOWN')
        status_emoji = "✅" if assessment == "READY" else "❌"
        print(f"\n🚀 PRODUCTION READINESS: {status_emoji} {assessment}")
        
        if readiness.get("recommendations"):
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in readiness["recommendations"]:
                print(f"   • {rec}")
        
        # Query details
        print(f"\n📝 QUERY RESULTS:")
        for query_key, result in self.results.get("query_results", {}).items():
            if result.get("success", False):
                status = "✅" if result.get("production_ready", False) else "⚠️"
                score = result.get("overall_score", 0)
                time_ms = result.get("total_time_ms", 0)
                print(f"   {status} {query_key}: {score:.2f} score, {time_ms:.0f}ms")
            else:
                print(f"   ❌ {query_key}: FAILED - {result.get('error', 'Unknown error')}")


def main():
    """Main validation execution"""
    validator = EndToEndPipelineValidator()
    
    # Run validation
    results = validator.run_validation()
    
    # Print summary
    validator.print_summary()
    
    # Save results
    results_file = validator.save_results()
    
    return results, results_file


if __name__ == "__main__":
    results, results_file = main()
