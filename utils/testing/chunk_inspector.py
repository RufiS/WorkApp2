"""
Phase 1A: Chunk Content Audit - Context-Efficient Diagnostic Tool

Examines target chunks 10-12, 56, 58-60 to verify:
- Chunk existence in index
- Content verification and keyword presence  
- Index integrity and completeness
- Silent execution to preserve context window
"""

import os
import json
import logging
import warnings
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import io
from typing import Dict, List, Tuple, Optional

# Suppress all verbose output to preserve context window
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
warnings.filterwarnings('ignore')
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

class ChunkInspector:
    """Context-efficient chunk integrity analyzer"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.index_dir = self.data_dir / "index"
        self.target_chunks = [10, 11, 12, 56, 58, 59, 60]
        self.expected_keywords = {
            10: ["RingCentral", "Texting", "text"],
            11: ["SMS", "format", "message"],
            12: ["text message", "SMS", "response"],
            56: ["Text Response", "30 min", "contact"],
            58: ["Text Tickets", "Freshdesk", "number"],
            59: ["Field Engineer", "FE", "phone"],
            60: ["text message", "Field Engineer", "FE"]
        }
    
    def inspect_chunks(self) -> Dict:
        """
        Silent chunk inspection with summary output only
        Returns comprehensive diagnostic information
        """
        results = {
            "total_chunks": 0,
            "target_chunks_found": {},
            "missing_chunks": [],
            "content_verification": {},
            "index_health": {},
            "critical_issues": []
        }
        
        try:
            # Phase 1: Check index file integrity
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                index_status = self._check_index_files()
                results["index_health"] = index_status
            
            # Phase 2: Load and examine chunk content
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                chunk_data = self._load_chunk_content()
                results["total_chunks"] = len(chunk_data) if chunk_data else 0
            
            # Phase 3: Verify target chunks
            if chunk_data:
                for chunk_id in self.target_chunks:
                    chunk_info = self._verify_chunk(chunk_id, chunk_data)
                    if chunk_info["found"]:
                        results["target_chunks_found"][chunk_id] = chunk_info
                    else:
                        results["missing_chunks"].append(chunk_id)
                        results["critical_issues"].append(f"Chunk {chunk_id} missing from index")
            else:
                results["critical_issues"].append("No chunk data found - index may be corrupted")
            
            # Phase 4: Content verification
            results["content_verification"] = self._verify_content_quality(results["target_chunks_found"])
            
            return results
            
        except Exception as e:
            results["critical_issues"].append(f"Inspection failed: {str(e)}")
            return results
    
    def _check_index_files(self) -> Dict:
        """Check FAISS index and metadata file integrity"""
        status = {
            "faiss_index": False,
            "metadata": False,
            "texts": False,
            "chunks_txt": False,
            "index_lock": False
        }
        
        # Check for essential index files
        if (self.index_dir / "index.faiss").exists():
            status["faiss_index"] = True
        if (self.index_dir / "metadata.json").exists():
            status["metadata"] = True
        if (self.index_dir / "texts.npy").exists():
            status["texts"] = True
        if (Path("current_index") / "chunks.txt").exists():
            status["chunks_txt"] = True
        if (self.index_dir / "index.lock").exists():
            status["index_lock"] = True
            
        return status
    
    def _load_chunk_content(self) -> Optional[List[str]]:
        """Load chunk content from available sources"""
        chunks = []
        
        # Try chunks.txt first
        chunks_file = Path("current_index") / "chunks.txt"
        if chunks_file.exists():
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = [line.strip() for line in f if line.strip()]
                return chunks
            except Exception:
                pass
        
        # Try metadata.json
        metadata_file = self.index_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    if 'chunks' in metadata:
                        return metadata['chunks']
            except Exception:
                pass
        
        return None
    
    def _verify_chunk(self, chunk_id: int, chunk_data: List[str]) -> Dict:
        """Verify individual chunk existence and content"""
        if chunk_id >= len(chunk_data):
            return {
                "found": False,
                "reason": f"Index only has {len(chunk_data)} chunks, chunk {chunk_id} out of range"
            }
        
        chunk_content = chunk_data[chunk_id]
        if not chunk_content or len(chunk_content.strip()) == 0:
            return {
                "found": False,
                "reason": f"Chunk {chunk_id} exists but is empty"
            }
        
        return {
            "found": True,
            "content_length": len(chunk_content),
            "content_preview": chunk_content[:100] + "..." if len(chunk_content) > 100 else chunk_content,
            "keywords_found": self._check_keywords(chunk_id, chunk_content)
        }
    
    def _check_keywords(self, chunk_id: int, content: str) -> List[str]:
        """Check for expected keywords in chunk content"""
        found_keywords = []
        expected = self.expected_keywords.get(chunk_id, [])
        
        content_lower = content.lower()
        for keyword in expected:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _verify_content_quality(self, found_chunks: Dict) -> Dict:
        """Analyze content quality of found chunks"""
        quality_report = {
            "total_found": len(found_chunks),
            "keyword_coverage": {},
            "content_issues": []
        }
        
        for chunk_id, chunk_info in found_chunks.items():
            expected_keywords = self.expected_keywords.get(chunk_id, [])
            found_keywords = chunk_info.get("keywords_found", [])
            
            coverage = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
            quality_report["keyword_coverage"][chunk_id] = {
                "coverage_percent": round(coverage * 100, 1),
                "expected": expected_keywords,
                "found": found_keywords,
                "missing": [kw for kw in expected_keywords if kw not in found_keywords]
            }
            
            if coverage < 0.5:
                quality_report["content_issues"].append(
                    f"Chunk {chunk_id}: Poor keyword coverage ({coverage*100:.1f}%)"
                )
        
        return quality_report
    
    def print_diagnostic_report(self, results: Dict):
        """Print context-efficient diagnostic summary"""
        print("\n🔍 Phase 1A: Index Integrity Analysis")
        print("=" * 50)
        
        # Index Health
        health = results["index_health"]
        healthy_files = sum(health.values())
        total_files = len(health)
        print(f"📁 Index Files: {healthy_files}/{total_files} present")
        
        if not health["faiss_index"]:
            print("   ❌ FAISS index missing")
        if not health["chunks_txt"]:
            print("   ❌ chunks.txt missing")
        
        # Chunk Status
        total_chunks = results["total_chunks"]
        found_count = len(results["target_chunks_found"])
        missing_count = len(results["missing_chunks"])
        
        print(f"\n📊 Target Chunks: {found_count}/{len(self.target_chunks)} found ({found_count/len(self.target_chunks)*100:.1f}%)")
        print(f"📈 Total Index Size: {total_chunks} chunks")
        
        # Found chunks detail
        if results["target_chunks_found"]:
            print("\n✅ Found Chunks:")
            for chunk_id, info in results["target_chunks_found"].items():
                keywords = len(info.get("keywords_found", []))
                expected = len(self.expected_keywords.get(chunk_id, []))
                print(f"   Chunk {chunk_id}: {info['content_length']} chars, {keywords}/{expected} keywords")
        
        # Missing chunks
        if results["missing_chunks"]:
            print(f"\n❌ Missing Chunks: {results['missing_chunks']}")
        
        # Content Quality
        quality = results["content_verification"]
        if quality["content_issues"]:
            print(f"\n⚠️  Content Issues:")
            for issue in quality["content_issues"]:
                print(f"   {issue}")
        
        # Critical Issues
        if results["critical_issues"]:
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in results["critical_issues"]:
                print(f"   {issue}")
        
        # Next Steps
        print(f"\n🎯 Next Steps:")
        if missing_count > 0:
            print(f"   1. Investigate missing chunks: {results['missing_chunks']}")
        if total_chunks != 221:
            print(f"   2. Index size mismatch: expected 221, found {total_chunks}")
        if quality["content_issues"]:
            print(f"   3. Address content quality issues")
        
        if not results["critical_issues"] and missing_count == 0:
            print("   ✅ Proceed to Phase 1B: Similarity Analysis")


def main():
    """Run chunk inspection with context-efficient output"""
    print("🔍 Starting Phase 1A: Chunk Content Audit...")
    
    inspector = ChunkInspector()
    results = inspector.inspect_chunks()
    inspector.print_diagnostic_report(results)
    
    # Save detailed results for later analysis
    output_file = f"chunk_inspection_results_{int(__import__('time').time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
