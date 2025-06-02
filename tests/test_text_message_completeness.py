"""
Text Message Retrieval Completeness Test
Uses text_message_retrieval_reference.json to validate complete workflow retrieval
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.document_processor import DocumentProcessor
from retrieval.retrieval_system import UnifiedRetrievalSystem


class TextMessageCompletenessTest:
    def __init__(self):
        self.reference_file = project_root / "text_message_retrieval_reference.json"
        self.load_reference()
        
    def load_reference(self):
        """Load the reference file with expected complete context"""
        with open(self.reference_file, 'r') as f:
            self.reference = json.load(f)
            
    def test_retrieval_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a specific retrieval configuration"""
        print(f"\n🔍 Testing configuration: {config['name']}")
        print(f"   Threshold: {config['similarity_threshold']}, Top-K: {config['top_k']}")
        
        # Initialize document processor with current index
        doc_processor = DocumentProcessor()
        
        # Check if we have an index loaded
        if not hasattr(doc_processor, 'index') or doc_processor.index is None:
            # Load the index
            doc_processor.load_index()
            
        if not hasattr(doc_processor, 'index') or doc_processor.index is None:
            return {"error": "No index available"}
            
        # Get the query
        query = self.reference["test_case"]["query"]
        
        # Perform retrieval with specified parameters
        start_time = time.time()
        
        # Use retrieval system with custom parameters
        from retrieval.engines.reranking_engine import RerankingEngine
        reranking_engine = RerankingEngine(doc_processor)
        
        # Temporarily modify config for this test
        original_config = self._update_config(config)
        
        try:
            context, search_time, num_chunks, retrieval_scores = reranking_engine.search(
                query, top_k=config['top_k']
            )
            
            retrieval_time = time.time() - start_time
            
            # Analyze coverage
            coverage_analysis = self.analyze_coverage(context)
            
            result = {
                "configuration": config,
                "retrieval_performance": {
                    "retrieval_time": retrieval_time,
                    "search_time": search_time,
                    "num_chunks": num_chunks,
                    "context_length": len(context),
                    "retrieval_scores": retrieval_scores[:5] if retrieval_scores else []
                },
                "coverage_analysis": coverage_analysis,
                "retrieved_context": context[:2000] + "..." if len(context) > 2000 else context
            }
            
        except Exception as e:
            result = {
                "configuration": config,
                "error": str(e),
                "coverage_analysis": {"error": True}
            }
            
        finally:
            # Restore original config
            self._restore_config(original_config)
            
        return result
        
    def _update_config(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Temporarily update retrieval config for testing"""
        try:
            config_path = project_root / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Store original values
            original = {
                "similarity_threshold": config["retrieval"]["similarity_threshold"],
                "top_k": config["retrieval"]["top_k"]
            }
            
            # Update with test values
            config["retrieval"]["similarity_threshold"] = test_config["similarity_threshold"]
            config["retrieval"]["top_k"] = test_config["top_k"]
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            return original
            
        except Exception as e:
            print(f"Warning: Could not update config: {e}")
            return {}
            
    def _restore_config(self, original: Dict[str, Any]):
        """Restore original config values"""
        if not original:
            return
            
        try:
            config_path = project_root / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            config["retrieval"]["similarity_threshold"] = original["similarity_threshold"]
            config["retrieval"]["top_k"] = original["top_k"]
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not restore config: {e}")
    
    def analyze_coverage(self, retrieved_context: str) -> Dict[str, Any]:
        """Analyze how well the retrieved context covers expected sections"""
        expected_sections = self.reference["expected_complete_context"]
        coverage_results = {}
        
        # Check each expected section
        for section_id, section_data in expected_sections.items():
            section_content = section_data["content"].lower()
            retrieved_lower = retrieved_context.lower()
            
            # Simple keyword matching for key phrases
            key_phrases = self._extract_key_phrases(section_content)
            found_phrases = sum(1 for phrase in key_phrases if phrase in retrieved_lower)
            
            coverage_results[section_id] = {
                "title": section_data["title"],
                "importance": section_data["importance"],
                "key_phrases_total": len(key_phrases),
                "key_phrases_found": found_phrases,
                "coverage_percentage": (found_phrases / len(key_phrases)) * 100 if key_phrases else 0,
                "is_covered": found_phrases >= len(key_phrases) * 0.7  # 70% threshold
            }
            
        # Calculate overall coverage
        total_sections = len(expected_sections)
        covered_sections = sum(1 for result in coverage_results.values() if result["is_covered"])
        
        # Check critical procedures
        critical_procedures = self.reference["critical_procedures_checklist"]
        procedures_found = []
        for procedure in critical_procedures:
            procedure_keywords = procedure.lower().split()
            if any(keyword in retrieved_context.lower() for keyword in procedure_keywords[-3:]):  # Check last 3 words
                procedures_found.append(procedure)
        
        return {
            "section_coverage": coverage_results,
            "overall_coverage": {
                "sections_covered": covered_sections,
                "total_sections": total_sections,
                "coverage_percentage": (covered_sections / total_sections) * 100,
                "meets_success_criteria": covered_sections >= total_sections * 0.9  # 90% threshold
            },
            "procedure_coverage": {
                "procedures_found": procedures_found,
                "total_procedures": len(critical_procedures),
                "coverage_percentage": (len(procedures_found) / len(critical_procedures)) * 100
            }
        }
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases for matching"""
        # Simple extraction - look for quoted text and key terms
        phrases = []
        
        # Extract quoted text
        import re
        quoted = re.findall(r'"([^"]*)"', content)
        phrases.extend([q.lower() for q in quoted if len(q) > 10])
        
        # Key dispatch terms
        key_terms = [
            "ringcentral", "sms", "text message", "freshdesk", "field engineer",
            "appointment confirmation", "vm/sms", "ticket", "3 touches",
            "30 min", "kti channel", "dispatch queue"
        ]
        
        for term in key_terms:
            if term in content.lower():
                phrases.append(term)
                
        return list(set(phrases))  # Remove duplicates
    
    def run_completeness_test(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        print("TEXT MESSAGE RETRIEVAL COMPLETENESS TEST")
        print("=" * 60)
        print(f"Query: {self.reference['test_case']['query']}")
        print(f"Expected sections: {len(self.reference['expected_complete_context'])}")
        print(f"Critical procedures: {len(self.reference['critical_procedures_checklist'])}")
        print("=" * 60)
        
        # Get test configurations
        test_configs = self.reference["retrieval_testing_parameters"]["test_configurations"]
        
        results = []
        
        for config in test_configs:
            result = self.test_retrieval_configuration(config)
            results.append(result)
            
            # Print summary
            if "error" not in result:
                coverage = result["coverage_analysis"]["overall_coverage"]
                print(f"   📊 Overall coverage: {coverage['coverage_percentage']:.1f}%")
                print(f"   ⏱️ Retrieval time: {result['retrieval_performance']['retrieval_time']:.3f}s")
                print(f"   📄 Chunks retrieved: {result['retrieval_performance']['num_chunks']}")
                
                if coverage["meets_success_criteria"]:
                    print("   ✅ MEETS SUCCESS CRITERIA")
                else:
                    print("   ❌ Does not meet success criteria")
            else:
                print(f"   ❌ Error: {result['error']}")
        
        # Generate summary and recommendations
        summary = self.generate_summary(results)
        
        # Save results
        results_file = project_root / f"text_message_completeness_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_summary": summary,
                "detailed_results": results,
                "reference_used": self.reference
            }, f, indent=2)
            
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return {
            "summary": summary,
            "results": results,
            "results_file": str(results_file)
        }
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary and recommendations"""
        successful_results = [r for r in results if "error" not in r]
        
        if not successful_results:
            return {"error": "No successful configurations"}
            
        # Find best configuration
        best_config = max(
            successful_results,
            key=lambda x: x["coverage_analysis"]["overall_coverage"]["coverage_percentage"]
        )
        
        # Calculate averages
        avg_coverage = sum(
            r["coverage_analysis"]["overall_coverage"]["coverage_percentage"] 
            for r in successful_results
        ) / len(successful_results)
        
        # Check if any meet criteria
        meeting_criteria = [
            r for r in successful_results 
            if r["coverage_analysis"]["overall_coverage"]["meets_success_criteria"]
        ]
        
        recommendations = []
        
        if not meeting_criteria:
            recommendations.append("❌ No configurations meet 90% coverage criteria")
            recommendations.append("🔧 Consider lower similarity threshold (0.15-0.25)")
            recommendations.append("📈 Consider higher top_k (25-50)")
            recommendations.append("🔍 May need hybrid search improvements")
        else:
            best_meeting = meeting_criteria[0]
            recommendations.append(f"✅ Found {len(meeting_criteria)} configurations meeting criteria")
            recommendations.append(f"🎯 Best: {best_meeting['configuration']['name']}")
            
        if avg_coverage < 70:
            recommendations.append("⚠️ Low average coverage - fundamental retrieval issue")
        elif avg_coverage < 90:
            recommendations.append("📊 Moderate coverage - parameter tuning needed")
        
        return {
            "total_configurations_tested": len(results),
            "successful_configurations": len(successful_results),
            "configurations_meeting_criteria": len(meeting_criteria),
            "average_coverage_percentage": round(avg_coverage, 1),
            "best_configuration": {
                "name": best_config["configuration"]["name"],
                "coverage_percentage": round(best_config["coverage_analysis"]["overall_coverage"]["coverage_percentage"], 1),
                "retrieval_time": round(best_config["retrieval_performance"]["retrieval_time"], 3)
            },
            "recommendations": recommendations
        }


def main():
    """Run the text message completeness test"""
    tester = TextMessageCompletenessTest()
    results = tester.run_completeness_test()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    summary = results["summary"]
    print(f"📊 Configurations tested: {summary['total_configurations_tested']}")
    print(f"✅ Successful tests: {summary['successful_configurations']}")
    print(f"🎯 Meeting criteria: {summary['configurations_meeting_criteria']}")
    print(f"📈 Average coverage: {summary['average_coverage_percentage']}%")
    
    print(f"\n🏆 Best configuration: {summary['best_configuration']['name']}")
    print(f"   Coverage: {summary['best_configuration']['coverage_percentage']}%")
    print(f"   Time: {summary['best_configuration']['retrieval_time']}s")
    
    print("\n📋 Recommendations:")
    for rec in summary["recommendations"]:
        print(f"   {rec}")
    
    return results


if __name__ == "__main__":
    main()
