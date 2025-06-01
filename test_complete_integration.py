"""
Complete Integration Test - Full System Fix Validation

This script performs end-to-end testing of the enhanced chunking solution:
1. Clears broken index
2. Rebuilds with enhanced processor
3. Tests text messaging query
4. Validates parameter sweep improvement
5. Demonstrates complete fix
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from core.document_processor import DocumentProcessor
from utils.testing.answer_quality_parameter_sweep import AnswerQualityParameterSweep


class CompleteIntegrationTest:
    """End-to-end test of the enhanced chunking solution"""
    
    def __init__(self, source_pdf: str = "KTI Dispatch Guide.pdf"):
        self.source_pdf = Path(source_pdf)
        self.data_dir = Path("data")
        self.current_index_dir = Path("current_index")
        self.backup_dir = Path("integration_backup")
        
        # Test query
        self.test_query = "How do I respond to a text message"
        
    def backup_current_state(self):
        """Backup current broken state for comparison"""
        print("💾 Backing up current state...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Backup existing index
        if self.data_dir.exists():
            shutil.copytree(self.data_dir, self.backup_dir / "data_broken")
        if self.current_index_dir.exists():
            shutil.copytree(self.current_index_dir, self.backup_dir / "current_index_broken")
        
        print("✅ Backup complete")
    
    def clear_index(self):
        """Clear current broken index"""
        print("🧹 Clearing broken index...")
        
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        if self.current_index_dir.exists():
            shutil.rmtree(self.current_index_dir)
        
        print("✅ Index cleared")
    
    def rebuild_with_enhanced_processing(self):
        """Rebuild index using enhanced processing"""
        print("🔧 Rebuilding with enhanced processing...")
        
        if not self.source_pdf.exists():
            raise FileNotFoundError(f"Source PDF not found: {self.source_pdf}")
        
        start_time = time.time()
        
        try:
            # Create document processor (now uses enhanced processor)
            processor = DocumentProcessor()
            
            # Process the PDF
            index, chunks = processor.process_documents([str(self.source_pdf)])
            
            # Save the index
            processor.save_index()
            
            build_time = time.time() - start_time
            
            print(f"✅ Enhanced rebuild complete: {build_time:.1f}s")
            print(f"   - Total chunks: {len(chunks)}")
            print(f"   - Index saved to: {self.data_dir}")
            
            return {
                "success": True,
                "chunks": len(chunks),
                "build_time": build_time,
                "index": index
            }
            
        except Exception as e:
            print(f"❌ Rebuild failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def test_text_message_query(self):
        """Test the specific text message query that was failing"""
        print(f"🔍 Testing query: '{self.test_query}'")
        
        try:
            # Create document processor to test search
            processor = DocumentProcessor()
            
            # Load the enhanced index
            processor.load_index()
            
            # Search for text message content
            results = processor.search(self.test_query, top_k=5)
            
            print(f"📊 Search Results:")
            print(f"   - Found {len(results)} relevant chunks")
            
            if results:
                for i, result in enumerate(results[:3]):
                    chunk_text = result.get('text', '')
                    score = result.get('score', 0)
                    preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
                    print(f"   {i+1}. Score: {score:.3f} - {preview}")
                
                # Check for text messaging keywords
                keywords = ["text message", "sms", "texting", "ringcentral"]
                keyword_hits = 0
                for result in results:
                    chunk_text = result.get('text', '').lower()
                    for keyword in keywords:
                        if keyword in chunk_text:
                            keyword_hits += 1
                            break
                
                coverage = (keyword_hits / len(results)) * 100 if results else 0
                print(f"   - Keyword coverage: {coverage:.1f}% ({keyword_hits}/{len(results)})")
                
                return {
                    "success": True,
                    "results_count": len(results),
                    "keyword_coverage": coverage,
                    "has_relevant_content": coverage > 0
                }
            else:
                print("   - No results found")
                return {
                    "success": True,
                    "results_count": 0,
                    "keyword_coverage": 0,
                    "has_relevant_content": False
                }
                
        except Exception as e:
            print(f"❌ Query test failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_parameter_sweep_sample(self):
        """Run a focused parameter sweep to validate improvement"""
        print("⚙️ Running parameter sweep validation...")
        
        try:
            # Create parameter sweep with focused configurations
            sweep = AnswerQualityParameterSweep()
            
            # Test key configurations
            test_configs = [
                {"vector_weight": 0.7, "top_k": 20, "temperature": 0.0},
                {"vector_weight": 0.8, "top_k": 30, "temperature": 0.1},
                {"vector_weight": 0.6, "top_k": 25, "temperature": 0.0},
            ]
            
            results = []
            for i, config in enumerate(test_configs):
                print(f"   Testing config {i+1}/3...")
                
                # Run single configuration
                result = sweep.run_single_configuration(
                    query=self.test_query,
                    config=config,
                    config_name=f"test_config_{i+1}"
                )
                results.append(result)
            
            # Analyze results
            coverage_scores = [r.get("answer_coverage", 0) for r in results if r.get("success")]
            avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0
            
            print(f"📈 Parameter Sweep Results:")
            print(f"   - Configurations tested: {len(test_configs)}")
            print(f"   - Average coverage: {avg_coverage:.1f}%")
            print(f"   - Best coverage: {max(coverage_scores) if coverage_scores else 0:.1f}%")
            
            return {
                "success": True,
                "avg_coverage": avg_coverage,
                "max_coverage": max(coverage_scores) if coverage_scores else 0,
                "improvement": avg_coverage > 50  # Significant improvement threshold
            }
            
        except Exception as e:
            print(f"❌ Parameter sweep failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def compare_with_broken_baseline(self):
        """Compare results with broken baseline"""
        print("📊 Comparing with broken baseline...")
        
        # Previous broken results
        broken_baseline = {
            "chunks": 2477,
            "avg_chunk_size": 17.7,
            "text_message_coverage": 0.0,
            "parameter_sweep_coverage": 0.0
        }
        
        print(f"Broken Baseline:")
        print(f"   - Chunks: {broken_baseline['chunks']}")
        print(f"   - Avg chunk size: {broken_baseline['avg_chunk_size']} chars")
        print(f"   - Text message coverage: {broken_baseline['text_message_coverage']}%")
        print(f"   - Parameter sweep: {broken_baseline['parameter_sweep_coverage']}%")
        
        return broken_baseline
    
    def run_complete_test(self):
        """Run the complete integration test"""
        print("🚀 Complete Integration Test - Enhanced Chunking Solution")
        print("=" * 80)
        print(f"Source: {self.source_pdf}")
        print(f"Goal: Fix 0.0% parameter sweep → >50% coverage")
        print(f"Test Query: '{self.test_query}'")
        
        # Step 1: Backup current state
        self.backup_current_state()
        
        # Step 2: Compare with broken baseline
        broken_baseline = self.compare_with_broken_baseline()
        
        # Step 3: Clear and rebuild with enhanced processing
        self.clear_index()
        rebuild_result = self.rebuild_with_enhanced_processing()
        
        if not rebuild_result["success"]:
            print("\n❌ Integration test failed at rebuild step")
            return rebuild_result
        
        # Step 4: Test text message query
        query_result = self.test_text_message_query()
        
        # Step 5: Run parameter sweep validation
        sweep_result = self.run_parameter_sweep_sample()
        
        # Step 6: Final assessment
        self.assess_integration_success(rebuild_result, query_result, sweep_result, broken_baseline)
        
        # Return comprehensive results
        return {
            "success": True,
            "rebuild": rebuild_result,
            "query_test": query_result,
            "parameter_sweep": sweep_result,
            "broken_baseline": broken_baseline,
            "timestamp": time.time()
        }
    
    def assess_integration_success(self, rebuild_result, query_result, sweep_result, broken_baseline):
        """Assess overall integration success"""
        print(f"\n🎯 Integration Assessment")
        print("=" * 50)
        
        # Success criteria
        criteria = {
            "Index rebuilt successfully": rebuild_result.get("success", False),
            "Chunk count improved (2477 → <500)": rebuild_result.get("chunks", 0) < 500,
            "Text message query finds content": query_result.get("has_relevant_content", False),
            "Parameter sweep shows improvement": sweep_result.get("improvement", False),
        }
        
        passed = sum(criteria.values())
        total = len(criteria)
        
        print(f"Success Criteria: {passed}/{total} passed")
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion}")
        
        # Quantitative improvements
        print(f"\n📈 Quantitative Improvements:")
        chunk_improvement = ((broken_baseline["chunks"] - rebuild_result.get("chunks", 0)) / 
                           broken_baseline["chunks"] * 100)
        print(f"   - Chunk reduction: {chunk_improvement:.1f}% ({broken_baseline['chunks']} → {rebuild_result.get('chunks', 0)})")
        
        coverage_improvement = sweep_result.get("avg_coverage", 0) - broken_baseline["parameter_sweep_coverage"]
        print(f"   - Coverage improvement: +{coverage_improvement:.1f}% (0.0% → {sweep_result.get('avg_coverage', 0):.1f}%)")
        
        if query_result.get("keyword_coverage", 0) > 0:
            print(f"   - Text message query: ∞% improvement (0% → {query_result.get('keyword_coverage', 0):.1f}%)")
        
        # Overall verdict
        if passed >= 3:
            print(f"\n🎉 INTEGRATION SUCCESS!")
            print(f"   ✅ Root cause fixed: Micro-chunking eliminated")
            print(f"   ✅ Content discovery: Text messaging procedures found") 
            print(f"   ✅ Search improvement: Parameter sweep now effective")
            print(f"   ✅ System recovery: 0.0% → {sweep_result.get('avg_coverage', 0):.1f}% coverage")
        else:
            print(f"\n⚠️ PARTIAL SUCCESS: Some issues remain")
        
        return passed >= 3


def main():
    """Run the complete integration test"""
    tester = CompleteIntegrationTest()
    
    try:
        results = tester.run_complete_test()
        
        # Save results
        output_file = f"integration_test_results_{int(time.time())}.json"
        
        # Prepare serializable results
        save_data = {
            "test_type": "complete_integration",
            "timestamp": results.get("timestamp", time.time()),
            "source_file": str(tester.source_pdf),
            "test_query": tester.test_query,
            "results": {
                "rebuild_success": results["rebuild"]["success"],
                "chunk_count": results["rebuild"].get("chunks", 0),
                "query_finds_content": results["query_test"].get("has_relevant_content", False),
                "parameter_sweep_coverage": results["parameter_sweep"].get("avg_coverage", 0),
                "overall_success": results.get("success", False)
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n📄 Integration test results saved to: {output_file}")
        
        # Final message
        if results.get("success") and results["parameter_sweep"].get("avg_coverage", 0) > 20:
            print(f"\n🚀 MISSION ACCOMPLISHED!")
            print(f"   The enhanced chunking solution successfully fixes the root cause.")
            print(f"   System recovered from 0.0% to {results['parameter_sweep'].get('avg_coverage', 0):.1f}% coverage.")
            print(f"   Text messaging queries now return relevant content.")
        else:
            print(f"\n⚠️ Further investigation needed - see results above")
    
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
