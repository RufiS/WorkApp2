"""
Fresh Index Builder - Rebuild index from source PDF with proper chunking

Creates new index with corrected chunking parameters to compare against broken index.
Identifies optimal chunking settings for text message content retrieval.
"""

import os
import sys
import json
import logging
import warnings
import shutil
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io
from typing import Dict, List, Optional
import time

# Add current directory to Python path for imports
sys.path.insert(0, os.getcwd())

# Suppress verbose output
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
warnings.filterwarnings('ignore')
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

class FreshIndexBuilder:
    """Rebuild index with proper chunking parameters"""
    
    def __init__(self, source_pdf: str = "KTI Dispatch Guide.pdf"):
        self.source_pdf = Path(source_pdf)
        self.backup_dir = Path("index_backup")
        self.data_dir = Path("data")
        self.current_index_dir = Path("current_index")
        
        # Test different chunking configurations
        self.chunking_configs = [
            {
                "name": "default_fixed",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "description": "Standard config with proper sizes"
            },
            {
                "name": "large_chunks",
                "chunk_size": 1500,
                "chunk_overlap": 300,
                "description": "Larger chunks for more context"
            },
            {
                "name": "medium_chunks", 
                "chunk_size": 800,
                "chunk_overlap": 150,
                "description": "Medium chunks for balance"
            }
        ]
    
    def backup_current_index(self) -> bool:
        """Backup current broken index for comparison"""
        try:
            if self.backup_dir.exists():
                shutil.rmtree(self.backup_dir)
            
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup data/index
            if self.data_dir.exists():
                shutil.copytree(self.data_dir, self.backup_dir / "data")
            
            # Backup current_index
            if self.current_index_dir.exists():
                shutil.copytree(self.current_index_dir, self.backup_dir / "current_index")
            
            print(f"✅ Backed up broken index to: {self.backup_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def clear_current_index(self):
        """Remove current broken index"""
        try:
            if self.data_dir.exists():
                shutil.rmtree(self.data_dir)
            if self.current_index_dir.exists():
                shutil.rmtree(self.current_index_dir)
            print("✅ Cleared broken index")
        except Exception as e:
            print(f"⚠️ Clear warning: {e}")
    
    def build_fresh_index(self, config: Dict) -> Dict:
        """Build fresh index with specified chunking config"""
        print(f"\n🔧 Building fresh index: {config['name']}")
        print(f"   Chunk size: {config['chunk_size']}, Overlap: {config['chunk_overlap']}")
        
        results = {
            "config": config,
            "success": False,
            "error": None,
            "chunks_created": 0,
            "build_time": 0,
            "inspection_results": None
        }
        
        start_time = time.time()
        
        try:
            # Import here to avoid startup overhead
            import importlib
            sys.path.insert(0, os.getcwd())
            
            # Try multiple import strategies
            processor = None
            
            # Strategy 1: Direct import
            try:
                from core.document_processor import DocumentProcessor
                processor = DocumentProcessor()
            except ImportError:
                # Strategy 2: Use workapp3 interface
                try:
                    import workapp3
                    # We'll use a simpler approach - directly run the main app to process
                    # This is a fallback if direct import fails
                    results["error"] = "Direct import failed, need alternative processing method"
                    return results
                except ImportError:
                    results["error"] = "Cannot import document processing modules"
                    return results
            
            if processor:
                # Create processor with custom chunking
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    # Override chunking parameters if possible
                    if hasattr(processor, 'chunk_size'):
                        processor.chunk_size = config['chunk_size']
                    if hasattr(processor, 'chunk_overlap'):
                        processor.chunk_overlap = config['chunk_overlap']
                    
                    # Process the PDF
                    success = processor.process_documents([str(self.source_pdf)])
                
                if success:
                    results["success"] = True
                    results["build_time"] = time.time() - start_time
                    
                    # Quick inspection of results
                    chunks_file = self.current_index_dir / "chunks.txt"
                    if chunks_file.exists():
                        with open(chunks_file, 'r', encoding='utf-8') as f:
                            chunks = [line.strip() for line in f if line.strip()]
                            results["chunks_created"] = len(chunks)
                    
                    print(f"✅ Index built: {results['chunks_created']} chunks in {results['build_time']:.1f}s")
                    
                else:
                    results["error"] = "Document processing failed"
                    print(f"❌ Index build failed: {results['error']}")
            
        except Exception as e:
            results["error"] = str(e)
            print(f"❌ Index build error: {e}")
        
        return results
    
    def inspect_fresh_index(self) -> Optional[Dict]:
        """Run chunk inspector on fresh index"""
        try:
            from utils.testing.chunk_inspector import ChunkInspector
            
            print("🔍 Inspecting fresh index...")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                inspector = ChunkInspector()
                results = inspector.inspect_chunks()
            
            return results
            
        except Exception as e:
            print(f"❌ Inspection failed: {e}")
            return None
    
    def compare_indexes(self, fresh_results: Dict, broken_results: Dict):
        """Compare fresh vs broken index results"""
        print("\n📊 Index Comparison: Fresh vs Broken")
        print("=" * 50)
        
        # Chunk counts
        fresh_chunks = fresh_results.get("total_chunks", 0)
        broken_chunks = broken_results.get("total_chunks", 0)
        print(f"📈 Total Chunks: Fresh={fresh_chunks}, Broken={broken_chunks}")
        
        # Target chunk analysis
        fresh_found = len(fresh_results.get("target_chunks_found", {}))
        broken_found = len(broken_results.get("target_chunks_found", {}))
        print(f"🎯 Target Chunks Found: Fresh={fresh_found}/7, Broken={broken_found}/7")
        
        # Content quality comparison
        print(f"\n🔍 Content Quality Comparison:")
        
        for chunk_id in [10, 11, 12, 56, 58, 59, 60]:
            fresh_chunk = fresh_results.get("target_chunks_found", {}).get(str(chunk_id))
            broken_chunk = broken_results.get("target_chunks_found", {}).get(str(chunk_id))
            
            if fresh_chunk and broken_chunk:
                fresh_keywords = len(fresh_chunk.get("keywords_found", []))
                broken_keywords = len(broken_chunk.get("keywords_found", []))
                fresh_len = fresh_chunk.get("content_length", 0)
                broken_len = broken_chunk.get("content_length", 0)
                
                print(f"   Chunk {chunk_id}: Fresh={fresh_len} chars, {fresh_keywords} keywords | Broken={broken_len} chars, {broken_keywords} keywords")
                
                if fresh_len > broken_len * 5:  # Significantly larger
                    print(f"      ✅ Fresh chunk much larger - likely contains proper content")
                elif fresh_keywords > broken_keywords:
                    print(f"      ✅ Fresh chunk has better keyword coverage")
    
    def rebuild_with_optimal_config(self) -> Dict:
        """Test multiple configs and select best one"""
        if not self.source_pdf.exists():
            return {"error": f"Source PDF not found: {self.source_pdf}"}
        
        print(f"🚀 Starting Fresh Index Rebuild")
        print(f"📄 Source: {self.source_pdf}")
        
        # Step 1: Backup current broken index
        if not self.backup_current_index():
            return {"error": "Failed to backup current index"}
        
        # Load broken index results for comparison
        broken_results = None
        try:
            with open("chunk_inspection_results_1748727205.json", 'r') as f:
                broken_results = json.load(f)
        except:
            print("⚠️ Could not load broken index results for comparison")
        
        best_config = None
        best_results = None
        all_results = []
        
        # Step 2: Test each chunking configuration
        for config in self.chunking_configs:
            self.clear_current_index()
            
            # Build with this config
            build_results = self.build_fresh_index(config)
            
            if build_results["success"]:
                # Inspect the results
                inspection = self.inspect_fresh_index()
                if inspection:
                    build_results["inspection_results"] = inspection
                    
                    # Score this configuration
                    score = self._score_config(inspection)
                    build_results["quality_score"] = score
                    
                    if best_results is None or score > best_results.get("quality_score", 0):
                        best_config = config
                        best_results = build_results
                    
                    print(f"   Quality Score: {score:.2f}")
            
            all_results.append(build_results)
        
        # Step 3: Use best configuration
        if best_config and best_results:
            print(f"\n🏆 Best Configuration: {best_config['name']}")
            print(f"   Score: {best_results['quality_score']:.2f}")
            print(f"   Chunks: {best_results['chunks_created']}")
            
            # Rebuild with best config one final time
            self.clear_current_index()
            final_build = self.build_fresh_index(best_config)
            
            if final_build["success"]:
                final_inspection = self.inspect_fresh_index()
                if final_inspection and broken_results:
                    self.compare_indexes(final_inspection, broken_results)
                
                return {
                    "success": True,
                    "best_config": best_config,
                    "final_results": final_build,
                    "all_configs_tested": all_results,
                    "fresh_inspection": final_inspection,
                    "broken_comparison": broken_results
                }
        
        return {"error": "No successful configuration found"}
    
    def _score_config(self, inspection: Dict) -> float:
        """Score a configuration based on inspection results"""
        score = 0.0
        
        # Reward finding target chunks
        found_chunks = len(inspection.get("target_chunks_found", {}))
        score += found_chunks * 10
        
        # Reward keyword coverage
        verification = inspection.get("content_verification", {})
        keyword_coverage = verification.get("keyword_coverage", {})
        
        for chunk_id, coverage_info in keyword_coverage.items():
            coverage_percent = coverage_info.get("coverage_percent", 0)
            score += coverage_percent / 100 * 5  # Up to 5 points per chunk
        
        # Penalize too many or too few chunks
        total_chunks = inspection.get("total_chunks", 0)
        if 150 <= total_chunks <= 300:  # Sweet spot
            score += 20
        elif total_chunks > 1000:  # Too many (like broken index)
            score -= 50
        
        # Reward reasonable chunk sizes
        for chunk_info in inspection.get("target_chunks_found", {}).values():
            content_length = chunk_info.get("content_length", 0)
            if content_length > 100:  # Substantial content
                score += 5
            elif content_length < 50:  # Too small
                score -= 5
        
        return score


def main():
    """Run fresh index rebuild with optimal configuration"""
    builder = FreshIndexBuilder()
    results = builder.rebuild_with_optimal_config()
    
    # Save detailed results
    output_file = f"fresh_index_rebuild_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {output_file}")
    
    if results.get("success"):
        print("\n🎯 Next Steps:")
        print("   1. Run similarity analysis on fresh index")
        print("   2. Test retrieval with 'How do I respond to a text message'")
        print("   3. Compare parameter sweep results")
    else:
        print(f"\n❌ Rebuild failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
