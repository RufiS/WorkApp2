#!/usr/bin/env python3
"""
GPU Search Performance Diagnostic Test

Identifies why FAISS GPU searches are falling back to CPU performance
"""

import time
import logging
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.embeddings.embedding_service import get_embedding_service
from core.index_management.gpu_manager import gpu_manager
from core.config import performance_config
import torch
import faiss
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_gpu_search_diagnostic():
    """Run comprehensive GPU search performance diagnostic"""
    
    print("🔍 GPU SEARCH PERFORMANCE DIAGNOSTIC")
    print("=" * 50)
    
    # Phase 1: System Information
    print("\n📊 PHASE 1: SYSTEM STATUS")
    print("-" * 30)
    
    # Check GPU availability
    print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"🔧 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB total")
        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        reserved = torch.cuda.memory_reserved(0) / (1024**2)
        print(f"🔧 GPU Memory Usage: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")
    
    print(f"🔧 FAISS GPU Config: {performance_config.use_gpu_for_faiss}")
    
    # Phase 2: GPU Manager Test
    print("\n🚀 PHASE 2: GPU MANAGER TEST")
    print("-" * 30)
    
    # Test GPU manager initialization
    stats = gpu_manager.get_gpu_stats()
    print(f"📈 GPU Manager Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Phase 3: Embedding Service Test
    print("\n🧠 PHASE 3: EMBEDDING SERVICE TEST")
    print("-" * 30)
    
    embedding_service = get_embedding_service()
    model_info = embedding_service.get_model_info()
    
    print(f"📝 Model: {model_info['model_name']}")
    print(f"📝 Device: {model_info['device']}")
    print(f"📝 Dimension: {model_info['embedding_dim']}")
    
    # Test embedding performance
    test_query = "What is the hourly rate for on-site service?"
    
    print(f"\n⏱️  Testing Query Embedding Performance...")
    start_time = time.time()
    query_embedding = embedding_service.embed_query(test_query)
    embed_time = time.time() - start_time
    print(f"✅ Query embedding: {embed_time:.4f}s")
    print(f"📊 Embedding shape: {query_embedding.shape}")
    print(f"📊 Embedding dtype: {query_embedding.dtype}")
    
    # Phase 4: Index Creation and GPU Transfer Test
    print("\n📚 PHASE 4: INDEX CREATION & GPU TRANSFER TEST")
    print("-" * 30)
    
    # Create small test index
    dim = model_info['embedding_dim']
    test_size = 100
    
    print(f"🔨 Creating test index with {test_size} vectors of dimension {dim}")
    
    # Generate test embeddings
    start_time = time.time()
    test_texts = [f"Test document {i} with content about various topics." for i in range(test_size)]
    test_embeddings = embedding_service.embed_texts(test_texts)
    embed_batch_time = time.time() - start_time
    print(f"✅ Batch embedding: {embed_batch_time:.4f}s for {test_size} texts")
    
    # Create FAISS index
    print(f"🔨 Creating FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(test_embeddings.astype(np.float32))
    print(f"✅ Index created with {index.ntotal} vectors")
    
    # Test GPU transfer
    print(f"🚀 Testing GPU transfer...")
    start_time = time.time()
    gpu_index, gpu_success = gpu_manager.move_index_to_gpu(index)
    transfer_time = time.time() - start_time
    
    print(f"✅ GPU Transfer: {gpu_success} in {transfer_time:.4f}s")
    
    if gpu_success:
        print(f"📊 Index device: {gpu_index.getDevice() if hasattr(gpu_index, 'getDevice') else 'CPU'}")
    
    # Phase 5: Search Performance Test
    print("\n🔍 PHASE 5: SEARCH PERFORMANCE TEST")
    print("-" * 30)
    
    search_index = gpu_index if gpu_success else index
    test_queries = [
        "What is the hourly rate?",
        "Service information",
        "Technical documentation",
        "Billing procedures",
        "Contact information"
    ]
    
    print(f"🔍 Testing {len(test_queries)} search queries...")
    
    search_times = []
    for i, query in enumerate(test_queries):
        print(f"\n🔍 Query {i+1}: {query}")
        
        # Embed query
        start_time = time.time()
        q_embedding = embedding_service.embed_query(query)
        embed_time = time.time() - start_time
        
        # Perform search
        start_time = time.time()
        scores, indices = search_index.search(q_embedding.astype(np.float32), 5)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        print(f"   ⏱️  Embed: {embed_time:.4f}s, Search: {search_time:.4f}s")
        print(f"   📊 Results: {len(indices[0])} hits, top score: {scores[0][0]:.4f}")
        
        # Check for performance anomalies
        if search_time > 1.0:
            print(f"   ⚠️  SLOW SEARCH DETECTED! ({search_time:.4f}s)")
        elif search_time < 0.001:
            print(f"   ⚡ Very fast search: {search_time:.4f}s")
    
    # Phase 6: Performance Analysis
    print("\n📈 PHASE 6: PERFORMANCE ANALYSIS")
    print("-" * 30)
    
    avg_search_time = sum(search_times) / len(search_times)
    min_search_time = min(search_times)
    max_search_time = max(search_times)
    
    print(f"📊 Search Performance Summary:")
    print(f"   Average: {avg_search_time:.4f}s")
    print(f"   Minimum: {min_search_time:.4f}s")
    print(f"   Maximum: {max_search_time:.4f}s")
    
    # Performance classification
    if avg_search_time < 0.01:
        print(f"✅ EXCELLENT: GPU performance confirmed")
    elif avg_search_time < 0.1:
        print(f"✅ GOOD: Acceptable GPU performance")
    elif avg_search_time < 1.0:
        print(f"⚠️  MODERATE: Possible GPU inefficiency")
    else:
        print(f"❌ POOR: Likely CPU fallback detected")
    
    # Performance variance analysis
    variance = max_search_time - min_search_time
    if variance > 0.1:
        print(f"⚠️  HIGH VARIANCE: {variance:.4f}s difference between fastest/slowest")
        print(f"   This suggests inconsistent GPU performance or intermittent CPU fallback")
    
    # Memory analysis
    if torch.cuda.is_available():
        final_allocated = torch.cuda.memory_allocated(0) / (1024**2)
        final_reserved = torch.cuda.memory_reserved(0) / (1024**2)
        print(f"\n💾 Final GPU Memory: {final_allocated:.1f}MB allocated, {final_reserved:.1f}MB reserved")
    
    print("\n🏁 DIAGNOSTIC COMPLETE")
    print("=" * 50)
    
    return {
        "gpu_available": torch.cuda.is_available(),
        "gpu_transfer_success": gpu_success,
        "avg_search_time": avg_search_time,
        "search_variance": variance,
        "performance_classification": "excellent" if avg_search_time < 0.01 else 
                                   "good" if avg_search_time < 0.1 else
                                   "moderate" if avg_search_time < 1.0 else "poor"
    }


if __name__ == "__main__":
    try:
        results = run_gpu_search_diagnostic()
        print(f"\n📋 Results Summary: {results}")
    except Exception as e:
        print(f"❌ Diagnostic failed: {str(e)}")
        import traceback
        traceback.print_exc()
