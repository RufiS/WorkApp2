"""Test to verify metadata preservation through the document processing pipeline"""

import os
import sys
import tempfile
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document_ingestion.enhanced_file_processor import EnhancedFileProcessor
from core.document_processor import DocumentProcessor
from retrieval.retrieval_system import UnifiedRetrievalSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_metadata_preservation():
    """Test that metadata is preserved through the entire pipeline"""
    
    # Create a test file
    test_content = """
    # Text Messaging Guide
    
    When responding to a text message:
    1. Look up the phone number
    2. Find the client's info
    3. Send a message to the Field Engineer
    
    Example: @FieldEngineer Your client - Jane Doe 123-456-7890 - is texting you
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        test_file_path = f.name
    
    try:
        print("\n=== Testing Metadata Preservation ===\n")
        
        # Step 1: Process file with EnhancedFileProcessor
        print("1. Processing file with EnhancedFileProcessor...")
        processor = EnhancedFileProcessor(chunk_size=800, chunk_overlap=150)
        chunks = processor.load_and_chunk_document(test_file_path, original_filename="test_guide.txt")
        
        print(f"   - Generated {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"   - Chunk {i}: metadata = {chunk.get('metadata', {})}")
            assert 'metadata' in chunk, f"Chunk {i} missing metadata"
            assert 'source' in chunk['metadata'], f"Chunk {i} missing source in metadata"
            assert chunk['metadata']['source'] == 'test_guide.txt', f"Chunk {i} has wrong source: {chunk['metadata']['source']}"
        
        # Step 2: Process with DocumentProcessor
        print("\n2. Processing with DocumentProcessor...")
        doc_processor = DocumentProcessor()
        
        # Create index from chunks
        index, stored_chunks = doc_processor.create_index_from_chunks(chunks)
        print(f"   - Created index with {len(stored_chunks)} chunks")
        
        # Check metadata in stored chunks
        for i, chunk in enumerate(stored_chunks):
            print(f"   - Stored chunk {i}: metadata = {chunk.get('metadata', {})}")
            if 'metadata' not in chunk:
                print(f"   WARNING: Stored chunk {i} missing metadata!")
        
        # Step 3: Search and verify metadata
        print("\n3. Testing search retrieval...")
        results = doc_processor.search("text message", top_k=3)
        
        print(f"   - Retrieved {len(results)} results")
        for i, result in enumerate(results):
            print(f"   - Result {i}:")
            print(f"     - Score: {result.get('score', 'N/A')}")
            print(f"     - Metadata: {result.get('metadata', {})}")
            print(f"     - Source: {result.get('metadata', {}).get('source', 'NOT FOUND')}")
            print(f"     - Text preview: {result.get('text', '')[:50]}...")
        
        # Step 4: Test with UnifiedRetrievalSystem
        print("\n4. Testing with UnifiedRetrievalSystem...")
        # Use the same document processor that has the indexed data
        retrieval_system = UnifiedRetrievalSystem(document_processor=doc_processor)
        
        # Retrieve using the retrieval system
        context, retrieval_time, chunk_count, scores = retrieval_system.retrieve("text message", top_k=3)
        print(f"\n   - Retrieved context:\n{context[:300]}...")
        print(f"   - Retrieval time: {retrieval_time:.3f}s")
        print(f"   - Chunk count: {chunk_count}")
        
        # Check if "unknown" appears in context
        if "From unknown:" in context:
            print("\n   ERROR: Context contains 'From unknown:' - metadata was lost!")
            # Print the full context to debug
            print("\n   Full context:")
            print(context)
        else:
            print("\n   SUCCESS: Context does not contain 'From unknown:' - metadata preserved!")
        
    finally:
        # Cleanup
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)


if __name__ == "__main__":
    test_metadata_preservation()
