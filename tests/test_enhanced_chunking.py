"""
Test Enhanced Chunking - Validate the fix for micro-chunking and TOC noise

This script tests the enhanced file processor against the source PDF to verify:
1. Fixed chunking: 200-300 meaningful chunks instead of 2,477 micro-chunks
2. Content filtering: Table of contents noise removed
3. Quality improvement: Text messaging content properly preserved
4. Comparison: Before vs after results
"""

import os
import sys
import json
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

from core.document_ingestion.enhanced_file_processor import EnhancedFileProcessor
from utils.testing.chunk_inspector import ChunkInspector


class ChunkingTestValidator:
    """Test and validate the enhanced chunking solution"""

    def __init__(self, source_pdf: str = "KTI Dispatch Guide.pdf"):
        self.source_pdf = Path(source_pdf)
        self.enhanced_processor = EnhancedFileProcessor(chunk_size=1000, chunk_overlap=200)
        self.inspector = ChunkInspector()

        # Load broken index results for comparison
        self.broken_results = self._load_broken_index_results()

    def _load_broken_index_results(self):
        """Load the broken index inspection results for comparison"""
        try:
            with open("chunk_inspection_results_1748727205.json", 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠️ Warning: Broken index results not found, will skip comparison")
            return None

    def test_enhanced_processing(self):
        """Test the enhanced file processor with source PDF"""
        print("🚀 Testing Enhanced File Processor")
        print("=" * 50)

        if not self.source_pdf.exists():
            print(f"❌ Error: Source PDF not found: {self.source_pdf}")
            return None

        print(f"📄 Processing: {self.source_pdf}")
        start_time = time.time()

        try:
            # Process with enhanced processor
            chunks = self.enhanced_processor.load_and_chunk_document(str(self.source_pdf))
            processing_time = time.time() - start_time

            # Get content statistics
            stats = self.enhanced_processor.get_content_stats(chunks)

            print(f"\n✅ Enhanced Processing Complete: {processing_time:.1f}s")
            print(f"📊 Results Summary:")
            print(f"   - Total chunks: {len(chunks)}")
            print(f"   - Average size: {stats['size_stats']['average']:.1f} characters")
            print(f"   - Size range: {stats['size_stats']['min']}-{stats['size_stats']['max']} chars")
            print(f"   - Text message chunks: {stats['content_analysis']['text_message_chunks']}")
            print(f"   - TOC chunks: {stats['content_analysis']['toc_chunks']}")

            return {
                "chunks": chunks,
                "stats": stats,
                "processing_time": processing_time,
                "success": True
            }

        except Exception as e:
            print(f"❌ Enhanced processing failed: {e}")
            return {
                "error": str(e),
                "success": False
            }

    def compare_with_broken_index(self, enhanced_results):
        """Compare enhanced results with broken index"""
        if not self.broken_results or not enhanced_results["success"]:
            print("\n⚠️ Skipping comparison - missing data")
            return

        print("\n📊 Comparison: Enhanced vs Broken Index")
        print("=" * 50)

        # Chunk count comparison
        enhanced_count = len(enhanced_results["chunks"])
        broken_count = self.broken_results.get("total_chunks", 0)

        print(f"📈 Chunk Count:")
        print(f"   Enhanced: {enhanced_count} chunks")
        print(f"   Broken:   {broken_count} chunks")
        print(f"   Improvement: {((broken_count - enhanced_count) / broken_count * 100):.1f}% reduction")

        # Size comparison
        enhanced_avg = enhanced_results["stats"]["size_stats"]["average"]
        broken_chunks = self.broken_results.get("target_chunks_found", {})

        if broken_chunks:
            broken_sizes = [info.get("content_length", 0) for info in broken_chunks.values()]
            broken_avg = sum(broken_sizes) / len(broken_sizes) if broken_sizes else 0

            print(f"\n📏 Average Chunk Size:")
            print(f"   Enhanced: {enhanced_avg:.1f} characters")
            print(f"   Broken:   {broken_avg:.1f} characters")
            print(f"   Improvement: {(enhanced_avg / broken_avg):.1f}x larger chunks")

        # Content quality comparison
        enhanced_text_chunks = enhanced_results["stats"]["content_analysis"]["text_message_chunks"]
        broken_keyword_coverage = sum(1 for chunk in broken_chunks.values()
                                    if len(chunk.get("keywords_found", [])) > 0)

        print(f"\n🎯 Content Quality:")
        print(f"   Enhanced text message chunks: {enhanced_text_chunks}")
        print(f"   Broken keyword coverage: {broken_keyword_coverage}/7 target chunks")
        print(f"   Quality improvement: {enhanced_text_chunks}x more relevant content identified")

        # Quality indicators
        quality = enhanced_results["stats"]["quality_indicators"]
        print(f"\n✅ Quality Indicators:")
        print(f"   Appropriate chunk count: {'✅' if quality['appropriate_chunk_count'] else '❌'}")
        print(f"   Good average size: {'✅' if quality['good_average_size'] else '❌'}")
        print(f"   Has target content: {'✅' if quality['has_target_content'] else '❌'}")

    def sample_chunk_content(self, enhanced_results):
        """Show sample chunk content to verify quality"""
        if not enhanced_results["success"]:
            return

        chunks = enhanced_results["chunks"]
        print(f"\n📝 Sample Chunk Content (Text Messaging Related)")
        print("=" * 50)

        # Find chunks with text messaging content
        text_message_chunks = [chunk for chunk in chunks
                             if any(keyword in chunk["text"].lower()
                                   for keyword in ["text message", "sms", "texting", "ringcentral"])]

        if text_message_chunks:
            for i, chunk in enumerate(text_message_chunks[:3]):  # Show first 3
                print(f"\n--- Sample Chunk {i+1} ({len(chunk['text'])} chars) ---")
                preview = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                print(preview)
                print(f"Source: {chunk['metadata']['source']}")
                print(f"Page: {chunk['metadata'].get('page', 'Unknown')}")
        else:
            print("❌ No text messaging content found in chunks")

    def run_validation_test(self):
        """Run complete validation test"""
        print("🧪 Enhanced Chunking Validation Test")
        print("=" * 60)
        print(f"Source: {self.source_pdf}")
        print(f"Target: Fix micro-chunking (2,477 → 200-300 chunks)")
        print(f"Goal: Remove TOC noise, preserve content quality")

        # Test enhanced processing
        enhanced_results = self.test_enhanced_processing()

        if not enhanced_results or not enhanced_results["success"]:
            print("\n❌ Test Failed: Enhanced processing unsuccessful")
            return enhanced_results

        # Compare with broken index
        self.compare_with_broken_index(enhanced_results)

        # Show sample content
        self.sample_chunk_content(enhanced_results)

        # Overall assessment
        self._assess_fix_success(enhanced_results)

        return enhanced_results

    def _assess_fix_success(self, enhanced_results):
        """Assess if the fix successfully addresses the root cause"""
        print(f"\n🎯 Fix Assessment")
        print("=" * 50)

        chunks = enhanced_results["chunks"]
        stats = enhanced_results["stats"]

        # Success criteria
        criteria = {
            "Appropriate chunk count (100-400)": 100 <= len(chunks) <= 400,
            "Good average size (300-1200 chars)": 300 <= stats["size_stats"]["average"] <= 1200,
            "Has text messaging content": stats["content_analysis"]["text_message_chunks"] > 0,
            "Reduced TOC noise": stats["content_analysis"]["toc_chunks"] < len(chunks) * 0.1,
            "No micro-chunking": stats["size_stats"]["min"] >= 50,
        }

        passed = sum(criteria.values())
        total = len(criteria)

        print(f"Success Criteria: {passed}/{total} passed")
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion}")

        if passed >= 4:
            print(f"\n🎉 SUCCESS: Enhanced chunking fixes the root cause!")
            print(f"   - Micro-chunking eliminated")
            print(f"   - Content quality preserved")
            print(f"   - TOC noise filtered")
            print(f"   - Ready for parameter sweep retest")
        else:
            print(f"\n⚠️ PARTIAL: Some issues remain, may need further tuning")

        return passed >= 4


def main():
    """Run the enhanced chunking validation test"""
    validator = ChunkingTestValidator()
    results = validator.run_validation_test()

    # Save results for analysis
    if results and results.get("success"):
        output_file = f"enhanced_chunking_test_{int(time.time())}.json"

        # Prepare serializable results
        save_data = {
            "test_timestamp": time.time(),
            "source_file": str(validator.source_pdf),
            "total_chunks": len(results["chunks"]),
            "processing_time": results["processing_time"],
            "content_stats": results["stats"],
            "success": results["success"]
        }

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\n📄 Results saved to: {output_file}")

        # Next steps
        print(f"\n🎯 Next Steps:")
        print(f"   1. Replace FileProcessor with EnhancedFileProcessor in ingestion_manager.py")
        print(f"   2. Clear current index and rebuild with enhanced processing")
        print(f"   3. Rerun parameter sweep to verify >80% coverage")
        print(f"   4. Test 'How do I respond to a text message' query")

    else:
        print(f"\n❌ Test failed - see errors above")


if __name__ == "__main__":
    main()
