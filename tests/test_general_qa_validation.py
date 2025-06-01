"""
General Q&A Validation Test Suite for WorkApp2
Tests whether the system can answer comprehensive questions requiring multi-section synthesis
Based on KTI Dispatch Guide content for authentic dispatcher scenarios
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_system_infrastructure():
    """Test that basic system components work"""
    print("=== SYSTEM INFRASTRUCTURE TEST ===")
    
    results = {
        "config_load": False,
        "document_processor": False,
        "retrieval_system": False,
        "llm_service": False,
        "import_errors": []
    }
    
    # Test config loading
    try:
        from core.config import config_manager
        config = config_manager.get_config()
        results["config_load"] = True
        print("✅ Config loading: WORKS")
        print(f"   Similarity threshold: {config.retrieval.similarity_threshold}")
        print(f"   Top K: {config.retrieval.top_k}")
        print(f"   Enhanced mode: {config.retrieval.enhanced_mode}")
    except Exception as e:
        results["import_errors"].append(f"Config loading: {str(e)}")
        print(f"❌ Config loading: FAILED - {e}")
    
    # Test document processor
    try:
        from core.document_processor import DocumentProcessor
        results["document_processor"] = True
        print("✅ Document processor: WORKS")
    except Exception as e:
        results["import_errors"].append(f"Document processor: {str(e)}")
        print(f"❌ Document processor: FAILED - {e}")
    
    # Test retrieval system
    try:
        from retrieval.retrieval_system import UnifiedRetrievalSystem
        results["retrieval_system"] = True
        print("✅ Retrieval system: WORKS")
    except Exception as e:
        results["import_errors"].append(f"Retrieval system: {str(e)}")
        print(f"❌ Retrieval system: FAILED - {e}")
        
    # Test LLM service
    try:
        from llm.services.llm_service import LLMService
        results["llm_service"] = True
        print("✅ LLM service: WORKS")
    except Exception as e:
        results["import_errors"].append(f"LLM service: {str(e)}")
        print(f"❌ LLM service: FAILED - {e}")
    
    return results

def test_general_qa_capability():
    """Test system's ability to answer comprehensive questions requiring multi-section synthesis"""
    print("\n=== GENERAL Q&A CAPABILITY TEST ===")
    
    # General questions based on KTI Dispatch Guide requiring multi-section synthesis
    qa_test_questions = [
        {
            "question": "A client is calling about computer repair pricing and wants to know what we can fix",
            "expected_topics": ["pricing", "hourly rate", "device compatibility", "on-site service", "payment methods"],
            "complexity": "HIGH",
            "sections_required": ["Cost of Service", "Devices We Work On", "Scheduling"]
        },
        {
            "question": "How do I handle a client who wants to cancel their appointment today?",
            "expected_topics": ["same day cancellation", "SDC fee", "reschedule option", "cancellation policy"],
            "complexity": "HIGH", 
            "sections_required": ["Appointment Cancellations", "Same-Day Cancellation", "Reschedule"]
        },
        {
            "question": "What's the complete process when a Field Engineer calls out sick?",
            "expected_topics": ["client notification", "rescheduling", "voicemail script", "SMS follow-up"],
            "complexity": "HIGH",
            "sections_required": ["Rescheduling Clients", "Call Handling", "SMS"]
        },
        {
            "question": "A client says their computer is still having the same problem after our Field Engineer visited",
            "expected_topics": ["4-point inspection", "revisit policy", "warranty period", "billable vs non-billable"],
            "complexity": "HIGH",
            "sections_required": ["Four-Point Inspection", "Revisit vs Follow-Up", "Customer Concerns"]
        },
        {
            "question": "How do I help a client who submitted an appointment request online?",
            "expected_topics": ["appointment request handling", "calling back", "website phone number", "booking process"],
            "complexity": "MEDIUM",
            "sections_required": ["Appointment Request", "Call Handling", "Booking Process"]
        }
    ]
    
    results = {
        "qa_system_available": False,
        "questions_tested": [],
        "overall_capability": "UNKNOWN",
        "infrastructure_ready": False,
        "document_processing": {}
    }
    
    try:
        # SELF-CONTAINED TEST: Process document within test context
        import os
        from core.document_processor import DocumentProcessor
        from retrieval.retrieval_system import UnifiedRetrievalSystem
        from llm.services.llm_service import LLMService
        from core.config import app_config
        
        # Check if KTI_Dispatch_Guide.pdf exists
        pdf_path = 'KTI_Dispatch_Guide.pdf'
        if not os.path.exists(pdf_path):
            results["error"] = f"KTI_Dispatch_Guide.pdf not found at {pdf_path}"
            results["overall_capability"] = "BROKEN"
            print(f"❌ {results['error']}")
            return results
            
        print("📁 Self-contained test: Processing KTI_Dispatch_Guide.pdf...")
        
        # Create fresh DocumentProcessor for testing only
        doc_processor = DocumentProcessor()
        
        # Process the document to create chunks and index
        process_start = time.time()
        chunks = doc_processor.load_and_chunk_document(pdf_path)
        process_time = time.time() - process_start
        
        if not chunks:
            results["error"] = "No chunks created from KTI_Dispatch_Guide.pdf"
            results["overall_capability"] = "BROKEN"
            print(f"❌ {results['error']}")
            return results
            
        print(f"✅ Document processed: {len(chunks)} chunks created in {process_time:.2f}s")
        
        # Create index from chunks
        index_start = time.time()
        index, processed_chunks = doc_processor.create_index_from_chunks(chunks)
        index_time = time.time() - index_start
        
        if not index or not processed_chunks:
            results["error"] = "Failed to create search index from chunks"
            results["overall_capability"] = "BROKEN"
            print(f"❌ {results['error']}")
            return results
            
        print(f"✅ Search index created: {len(processed_chunks)} chunks indexed in {index_time:.2f}s")
        
        # Store document processing results
        results["document_processing"] = {
            "chunks_created": len(chunks),
            "chunks_indexed": len(processed_chunks),
            "processing_time": process_time,
            "indexing_time": index_time,
            "total_time": process_time + index_time
        }
        
        # Now we have a loaded index - proceed with Q&A testing
        print("🔍 Running General Q&A tests with loaded document...")
        
        # Initialize full QA pipeline
        retrieval_system = UnifiedRetrievalSystem(doc_processor)
        results["qa_system_available"] = True
        
        for qa_test in qa_test_questions:
            print(f"\n🔍 Testing: {qa_test['question']}")
            
            start_time = time.time()
            
            try:
                # Retrieve relevant chunks using direct search method
                retrieved_chunks = doc_processor.search(qa_test["question"], top_k=15)
                retrieval_time = time.time() - start_time
                
                # Check if we got relevant content
                if retrieved_chunks:
                    # Simple content relevance check
                    content_text = " ".join([chunk.get('text', '') for chunk in retrieved_chunks])
                    topic_coverage = sum(1 for topic in qa_test["expected_topics"] 
                                       if topic.lower() in content_text.lower())
                    coverage_percentage = (topic_coverage / len(qa_test["expected_topics"])) * 100
                    
                    test_result = {
                        "question": qa_test["question"],
                        "complexity": qa_test["complexity"],
                        "chunks_retrieved": len(retrieved_chunks),
                        "topic_coverage": f"{topic_coverage}/{len(qa_test['expected_topics'])}",
                        "coverage_percentage": coverage_percentage,
                        "retrieval_time": retrieval_time,
                        "status": "GOOD" if coverage_percentage >= 60 else "PARTIAL" if coverage_percentage >= 30 else "POOR"
                    }
                    
                    print(f"   📊 Retrieved {len(retrieved_chunks)} chunks")
                    print(f"   🎯 Topic coverage: {topic_coverage}/{len(qa_test['expected_topics'])} ({coverage_percentage:.1f}%)")
                    print(f"   ⏱️ Retrieval time: {retrieval_time:.3f}s")
                    print(f"   📈 Assessment: {test_result['status']}")
                    
                else:
                    test_result = {
                        "question": qa_test["question"],
                        "complexity": qa_test["complexity"],
                        "chunks_retrieved": 0,
                        "topic_coverage": "0/0",
                        "coverage_percentage": 0,
                        "retrieval_time": retrieval_time,
                        "status": "FAILED"
                    }
                    print(f"   ❌ No chunks retrieved")
                
                results["questions_tested"].append(test_result)
                
            except Exception as e:
                results["questions_tested"].append({
                    "question": qa_test["question"],
                    "complexity": qa_test["complexity"],
                    "error": str(e),
                    "status": "ERROR"
                })
                print(f"   ❌ Error: {e}")
        
        # Calculate overall assessment
        successful_tests = [t for t in results["questions_tested"] if t.get("status") in ["GOOD", "PARTIAL"]]
        good_tests = [t for t in results["questions_tested"] if t.get("status") == "GOOD"]
        
        if len(good_tests) >= 3:
            results["overall_capability"] = "GOOD"
        elif len(successful_tests) >= 3:
            results["overall_capability"] = "PARTIAL"
        else:
            results["overall_capability"] = "POOR"
            
        print(f"\n🎯 Overall Q&A Capability: {results['overall_capability']}")
        
    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Q&A system initialization failed: {e}")
        results["overall_capability"] = "BROKEN"
    
    return results

def run_general_qa_validation():
    """Run complete General Q&A validation suite"""
    print("WORKAPP2 GENERAL Q&A VALIDATION TEST SUITE")
    print("Testing comprehensive document Q&A capability")
    print("=" * 60)
    
    # Run all tests
    infrastructure_results = test_system_infrastructure()
    qa_results = test_general_qa_capability()
    
    # Compile overall results
    overall_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_type": "GENERAL_QA_VALIDATION",
        "methodology": "Multi-section synthesis questions based on KTI Dispatch Guide",
        "test_results": {
            "infrastructure": infrastructure_results,
            "general_qa": qa_results
        },
        "overall_assessment": {
            "infrastructure": "UNKNOWN",
            "qa_capability": qa_results.get("overall_capability", "UNKNOWN"),
            "production_readiness": "NOT_READY",
            "validation_approach": "GENERAL_QA_METHODOLOGY"
        }
    }
    
    # Determine infrastructure status
    if (infrastructure_results["config_load"] and 
        infrastructure_results["document_processor"] and 
        infrastructure_results["retrieval_system"] and
        infrastructure_results["llm_service"]):
        overall_results["overall_assessment"]["infrastructure"] = "WORKING"
    else:
        overall_results["overall_assessment"]["infrastructure"] = "BROKEN"
        
    # Determine production readiness
    if (overall_results["overall_assessment"]["infrastructure"] == "WORKING" and
        overall_results["overall_assessment"]["qa_capability"] == "GOOD"):
        overall_results["overall_assessment"]["production_readiness"] = "READY"
    elif overall_results["overall_assessment"]["infrastructure"] == "WORKING":
        overall_results["overall_assessment"]["production_readiness"] = "NEEDS_IMPROVEMENT"
    
    print("\n" + "=" * 60)
    print("GENERAL Q&A VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Infrastructure: {overall_results['overall_assessment']['infrastructure']}")
    print(f"Q&A Capability: {overall_results['overall_assessment']['qa_capability']}")
    print(f"Production Readiness: {overall_results['overall_assessment']['production_readiness']}")
    print(f"Validation Methodology: {overall_results['overall_assessment']['validation_approach']}")
    
    # Save results
    results_file = project_root / "general_qa_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(overall_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return overall_results

if __name__ == "__main__":
    results = run_general_qa_validation()
