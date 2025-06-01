"""
Semantic Validation Test Suite for WorkApp2
Tests whether the embedding model understands dispatch domain terminology
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_system_functionality():
    """Test basic system components work without crashing"""
    print("=== BASIC SYSTEM FUNCTIONALITY TEST ===")
    
    results = {
        "config_load": False,
        "embedding_service": False,
        "document_processor": False,
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
    
    # Test embedding service import
    try:
        from core.embeddings.embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
        results["embedding_service"] = True
        print("✅ Embedding service import: WORKS")
    except Exception as e:
        results["import_errors"].append(f"Embedding service: {str(e)}")
        print(f"❌ Embedding service import: FAILED - {e}")
    
    # Test document processor import
    try:
        from core.document_processor import DocumentProcessor
        results["document_processor"] = True
        print("✅ Document processor import: WORKS")
    except Exception as e:
        results["import_errors"].append(f"Document processor: {str(e)}")
        print(f"❌ Document processor import: FAILED - {e}")
    
    return results

def test_semantic_understanding():
    """Test if embedding model understands dispatch terminology"""
    print("\n=== SEMANTIC UNDERSTANDING TEST ===")
    
    # Test pairs: (query term, expected similar term, domain)
    test_pairs = [
        ("text message", "SMS", "dispatch_communication"),
        ("Field Engineer", "FE", "dispatch_roles"),
        ("RingCentral", "phone system", "dispatch_tools"),
        ("dispatch", "send technician", "dispatch_actions"),
        ("emergency call", "urgent ticket", "dispatch_priority")
    ]
    
    results = {
        "embedding_available": False,
        "semantic_tests": [],
        "domain_understanding": "UNKNOWN"
    }
    
    try:
        from core.embeddings.embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
        results["embedding_available"] = True
        print("✅ Embedding service available")
        
        for query_term, similar_term, domain in test_pairs:
            try:
                # Get embeddings for both terms using correct API
                query_embedding = embedding_service.embed_query(query_term)[0]  # embed_query returns array with one row
                similar_embedding = embedding_service.embed_query(similar_term)[0]
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, similar_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(similar_embedding)
                )
                
                test_result = {
                    "query_term": query_term,
                    "similar_term": similar_term,
                    "domain": domain,
                    "similarity": float(similarity),
                    "assessment": "HIGH" if similarity > 0.7 else "MEDIUM" if similarity > 0.5 else "LOW"
                }
                
                results["semantic_tests"].append(test_result)
                print(f"   {query_term} ↔ {similar_term}: {similarity:.3f} ({test_result['assessment']})")
                
            except Exception as e:
                results["semantic_tests"].append({
                    "query_term": query_term,
                    "similar_term": similar_term,
                    "domain": domain,
                    "error": str(e),
                    "assessment": "ERROR"
                })
                print(f"   {query_term} ↔ {similar_term}: ERROR - {e}")
        
        # Overall assessment
        high_scores = [t for t in results["semantic_tests"] if t.get("assessment") == "HIGH"]
        medium_scores = [t for t in results["semantic_tests"] if t.get("assessment") == "MEDIUM"]
        
        if len(high_scores) >= 3:
            results["domain_understanding"] = "GOOD"
        elif len(high_scores) + len(medium_scores) >= 3:
            results["domain_understanding"] = "PARTIAL"
        else:
            results["domain_understanding"] = "POOR"
            
        print(f"\n🔍 Domain Understanding Assessment: {results['domain_understanding']}")
        
    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Semantic testing failed: {e}")
    
    return results

def test_actual_retrieval():
    """Test if system can retrieve relevant content for dispatch queries"""
    print("\n=== ACTUAL RETRIEVAL TEST ===")
    
    test_queries = [
        "How do I respond to a text message?",
        "What should a Field Engineer do for emergency calls?",
        "RingCentral setup procedures",
        "SMS response protocols"
    ]
    
    results = {
        "retrieval_available": False,
        "queries_tested": [],
        "overall_assessment": "UNKNOWN"
    }
    
    try:
        # Check if we can import retrieval components
        from retrieval.retrieval_system import UnifiedRetrievalSystem
        results["retrieval_available"] = True
        print("✅ Retrieval system available")
        
        # Note: We can't actually test retrieval without a built index
        # This would require document upload first
        print("⚠️ Full retrieval testing requires uploaded documents and built index")
        print("   This test validates import availability only")
        
        for query in test_queries:
            results["queries_tested"].append({
                "query": query,
                "status": "READY_FOR_TESTING",
                "note": "Requires document index to test"
            })
            print(f"   Query ready: {query}")
        
        results["overall_assessment"] = "INFRASTRUCTURE_READY"
        
    except Exception as e:
        results["error"] = str(e)
        print(f"❌ Retrieval system import failed: {e}")
        results["overall_assessment"] = "BROKEN"
    
    return results

def run_validation_suite():
    """Run complete validation test suite"""
    print("WORKAPP2 SEMANTIC VALIDATION TEST SUITE")
    print("=" * 50)
    
    # Run all tests
    basic_results = test_basic_system_functionality()
    semantic_results = test_semantic_understanding()
    retrieval_results = test_actual_retrieval()
    
    # Compile overall results
    overall_results = {
        "timestamp": "2025-06-01 04:32:00",
        "test_results": {
            "basic_functionality": basic_results,
            "semantic_understanding": semantic_results,
            "retrieval_availability": retrieval_results
        },
        "overall_assessment": {
            "infrastructure": "UNKNOWN",
            "semantic_capability": semantic_results.get("domain_understanding", "UNKNOWN"),
            "production_readiness": "NOT_READY",
            "validation_status": "IN_PROGRESS"
        }
    }
    
    # Determine infrastructure status
    if (basic_results["config_load"] and 
        basic_results["embedding_service"] and 
        basic_results["document_processor"]):
        overall_results["overall_assessment"]["infrastructure"] = "WORKING"
    else:
        overall_results["overall_assessment"]["infrastructure"] = "BROKEN"
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Infrastructure: {overall_results['overall_assessment']['infrastructure']}")
    print(f"Semantic Capability: {overall_results['overall_assessment']['semantic_capability']}")
    print(f"Production Readiness: {overall_results['overall_assessment']['production_readiness']}")
    
    # Save results
    results_file = project_root / "validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(overall_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return overall_results

if __name__ == "__main__":
    results = run_validation_suite()
