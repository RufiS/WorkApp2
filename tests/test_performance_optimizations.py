"""
Performance Optimizations Validation Test
Tests all optimizations: Model Preloading, Pipeline Async, Connection Pooling, Streaming
Compares optimized vs original performance
"""

import os
import sys
import json
import time
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
import traceback
from typing import List, Dict, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.document_processor import DocumentProcessor
from core.config import retrieval_config, app_config

# Import original services
from llm.services.llm_service import LLMService
from llm.prompt_generator import PromptGenerator
from llm.pipeline.answer_pipeline import AnswerPipeline

# Import optimized services
from llm.services.optimized_llm_service import OptimizedLLMService
from llm.pipeline.optimized_answer_pipeline import OptimizedAnswerPipeline
from llm.services.model_preloader import ModelPreloader, PreloadConfig
from llm.services.streaming_service import StreamingService, StreamingAnswerPipeline


class PerformanceOptimizationValidator:
    """Validates all performance optimizations and measures improvements"""

    def __init__(self):
        """Initialize performance validator"""
        self.project_root = project_root
        self.test_document = "./KTI_Dispatch_Guide.pdf"
        
        # Test queries for performance testing
        self.test_queries = [
            "What is our main phone number?",
            "How do I handle text messages from customers?",
            "How do I create a customer concern?",
            "Are we bonded and insured?",
        ]
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_type": "performance_optimization_validation",
            "optimizations_tested": [
                "model_preloading",
                "pipeline_async_optimization", 
                "connection_optimization",
                "response_streaming"
            ],
            "baseline_performance": {},
            "optimized_performance": {},
            "performance_improvements": {},
            "optimization_breakdown": {}
        }

    def setup_test_environment(self) -> bool:
        """Setup the test environment with document processing"""
        try:
            print("🔧 Setting up test environment...")
            
            # Clear existing indices
            index_dir = self.project_root / "data" / "index"
            if index_dir.exists():
                shutil.rmtree(index_dir)
                
            current_index_dir = self.project_root / "current_index"
            if current_index_dir.exists():
                shutil.rmtree(current_index_dir)
            
            # Process document once for both tests
            doc_processor = DocumentProcessor()
            test_doc_path = self.project_root / self.test_document
            
            if not test_doc_path.exists():
                print(f"❌ Test document not found: {test_doc_path}")
                return False
            
            print(f"📄 Processing test document...")
            start_time = time.time()
            index, chunks = doc_processor.process_documents([str(test_doc_path)])
            setup_time = time.time() - start_time
            
            if index is None or len(chunks) == 0:
                print(f"❌ Failed to process document")
                return False
            
            doc_processor.save_index()
            
            print(f"✅ Test environment ready in {setup_time:.1f}s")
            print(f"   Chunks: {len(chunks)}, Index size: {index.ntotal if hasattr(index, 'ntotal') else 'Unknown'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False

    async def test_baseline_performance(self) -> Dict[str, Any]:
        """Test performance with original components"""
        print("\n📊 Testing BASELINE performance (original components)...")
        
        baseline_results = {
            "query_results": {},
            "total_time_ms": 0,
            "avg_response_time_ms": 0,
            "setup_time_ms": 0
        }
        
        try:
            # Setup original services
            setup_start = time.time()
            api_key = app_config.api_keys.get("openai")
            if not api_key:
                return {"error": "No API key configured"}
            
            llm_service = LLMService(api_key)
            prompt_generator = PromptGenerator()
            answer_pipeline = AnswerPipeline(llm_service, prompt_generator)
            
            doc_processor = DocumentProcessor()
            doc_processor.load_index()
            setup_time = (time.time() - setup_start) * 1000
            baseline_results["setup_time_ms"] = setup_time
            
            print(f"   ⚙️ Baseline setup: {setup_time:.1f}ms")
            
            # Test each query
            total_time = 0
            for i, query in enumerate(self.test_queries):
                print(f"   🔍 Query {i+1}/{len(self.test_queries)}: '{query[:50]}...'")
                
                # Retrieval
                retrieval_start = time.time()
                chunks = doc_processor.search(query, top_k=retrieval_config.top_k)
                retrieval_time = (time.time() - retrieval_start) * 1000
                
                if not chunks:
                    continue
                
                # LLM Processing
                llm_start = time.time()
                context = "\n\n".join([chunk.get("text", "") for chunk in chunks])
                answer_result = answer_pipeline.generate_answer(query, context)
                llm_time = (time.time() - llm_start) * 1000
                
                query_total_time = retrieval_time + llm_time
                total_time += query_total_time
                
                baseline_results["query_results"][f"query_{i+1}"] = {
                    "query": query,
                    "retrieval_time_ms": round(retrieval_time, 2),
                    "llm_time_ms": round(llm_time, 2),
                    "total_time_ms": round(query_total_time, 2),
                    "answer_length": len(answer_result.get("content", "")),
                    "success": "error" not in answer_result
                }
                
                print(f"      ⏱️ {query_total_time:.0f}ms (retrieval: {retrieval_time:.0f}ms, LLM: {llm_time:.0f}ms)")
            
            baseline_results["total_time_ms"] = round(total_time, 2)
            baseline_results["avg_response_time_ms"] = round(total_time / len(self.test_queries), 2)
            
            print(f"📈 Baseline Results:")
            print(f"   Average Response Time: {baseline_results['avg_response_time_ms']:.0f}ms")
            print(f"   Total Time: {baseline_results['total_time_ms']:.0f}ms")
            
            return baseline_results
            
        except Exception as e:
            print(f"❌ Baseline test failed: {e}")
            return {"error": str(e)}

    async def test_optimized_performance(self) -> Dict[str, Any]:
        """Test performance with all optimizations enabled"""
        print("\n🚀 Testing OPTIMIZED performance (all optimizations)...")
        
        optimized_results = {
            "query_results": {},
            "total_time_ms": 0,
            "avg_response_time_ms": 0,
            "setup_time_ms": 0,
            "preload_time_ms": 0,
            "optimizations_used": []
        }
        
        optimized_llm_service = None
        try:
            # Setup optimized services
            setup_start = time.time()
            api_key = app_config.api_keys.get("openai")
            if not api_key:
                return {"error": "No API key configured"}
            
            # Use optimized LLM service with connection pooling as async context manager
            optimized_llm_service = OptimizedLLMService(api_key)
            prompt_generator = PromptGenerator()
            optimized_pipeline = OptimizedAnswerPipeline(optimized_llm_service, prompt_generator)
            
            doc_processor = DocumentProcessor()
            doc_processor.load_index()
            setup_time = (time.time() - setup_start) * 1000
            optimized_results["setup_time_ms"] = setup_time
            optimized_results["optimizations_used"].append("connection_optimization")
            optimized_results["optimizations_used"].append("pipeline_async_optimization")
            
            print(f"   ⚙️ Optimized setup: {setup_time:.1f}ms")
            
            # Model preloading
            preload_start = time.time()
            preloader = ModelPreloader(optimized_llm_service, None, PreloadConfig())
            preload_results = await preloader.preload_all_models()
            preload_time = (time.time() - preload_start) * 1000
            optimized_results["preload_time_ms"] = preload_time
            optimized_results["optimizations_used"].append("model_preloading")
            
            print(f"   🔥 Model preloading: {preload_time:.1f}ms")
            
            # Cleanup preloader resources immediately
            await preloader.cleanup()
            
            # Test each query with optimized pipeline
            total_time = 0
            for i, query in enumerate(self.test_queries):
                print(f"   🔍 Query {i+1}/{len(self.test_queries)}: '{query[:50]}...'")
                
                # Retrieval (same as baseline)
                retrieval_start = time.time()
                chunks = doc_processor.search(query, top_k=retrieval_config.top_k)
                retrieval_time = (time.time() - retrieval_start) * 1000
                
                if not chunks:
                    continue
                
                # Optimized LLM Processing
                llm_start = time.time()
                context = "\n\n".join([chunk.get("text", "") for chunk in chunks])
                answer_result = await optimized_pipeline.generate_answer_async(query, context)
                llm_time = (time.time() - llm_start) * 1000
                
                query_total_time = retrieval_time + llm_time
                total_time += query_total_time
                
                optimized_results["query_results"][f"query_{i+1}"] = {
                    "query": query,
                    "retrieval_time_ms": round(retrieval_time, 2),
                    "llm_time_ms": round(llm_time, 2),
                    "total_time_ms": round(query_total_time, 2),
                    "answer_length": len(answer_result.get("content", "")),
                    "success": "error" not in answer_result,
                    "strategy_used": answer_result.get("strategy_used", "unknown")
                }
                
                print(f"      ⏱️ {query_total_time:.0f}ms (retrieval: {retrieval_time:.0f}ms, LLM: {llm_time:.0f}ms)")
            
            optimized_results["total_time_ms"] = round(total_time, 2)
            optimized_results["avg_response_time_ms"] = round(total_time / len(self.test_queries), 2)
            
            # Get performance stats
            performance_stats = optimized_llm_service.get_performance_stats()
            pipeline_stats = optimized_pipeline.get_pipeline_stats()
            
            optimized_results["performance_stats"] = performance_stats
            optimized_results["pipeline_stats"] = pipeline_stats
            
            print(f"🚀 Optimized Results:")
            print(f"   Average Response Time: {optimized_results['avg_response_time_ms']:.0f}ms")
            print(f"   Total Time: {optimized_results['total_time_ms']:.0f}ms")
            print(f"   Cache Hit Rate: {performance_stats.get('cache_hit_rate', 0):.1%}")
            
            return optimized_results
            
        except Exception as e:
            print(f"❌ Optimized test failed: {e}")
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
        finally:
            # Ensure proper cleanup of async resources
            if optimized_llm_service:
                try:
                    await optimized_llm_service.close_connections()
                except Exception as e:
                    print(f"Warning: Error closing optimized LLM service: {e}")

    async def test_streaming_performance(self) -> Dict[str, Any]:
        """Test streaming performance for perceived speed"""
        print("\n📡 Testing STREAMING performance...")
        
        streaming_results = {
            "streaming_tests": {},
            "avg_time_to_first_chunk_ms": 0,
            "avg_total_streaming_time_ms": 0
        }
        
        streaming_service = None
        try:
            # Setup streaming service
            api_key = app_config.api_keys.get("openai")
            if not api_key:
                return {"error": "No API key configured"}
            
            # Use streaming service as async context manager
            streaming_service = StreamingService(api_key)
            prompt_generator = PromptGenerator()
            streaming_pipeline = StreamingAnswerPipeline(None, prompt_generator, streaming_service)
            
            doc_processor = DocumentProcessor()
            doc_processor.load_index()
            
            # Test streaming for first query
            query = self.test_queries[0]
            chunks = doc_processor.search(query, top_k=retrieval_config.top_k)
            context = "\n\n".join([chunk.get("text", "") for chunk in chunks])
            
            print(f"   🔍 Streaming test: '{query[:50]}...'")
            
            # Collect streaming chunks
            chunks_received = []
            total_content = ""
            first_chunk_time = None
            start_time = time.time()
            
            async for chunk in streaming_pipeline.stream_answer_generation(query, context):
                chunk_time = time.time()
                if first_chunk_time is None and chunk.content:
                    first_chunk_time = chunk_time
                
                chunks_received.append({
                    "chunk_index": chunk.chunk_index,
                    "content_length": len(chunk.content),
                    "total_content_length": len(chunk.total_content),
                    "time_ms": round((chunk_time - start_time) * 1000, 2),
                    "is_complete": chunk.is_complete
                })
                
                total_content = chunk.total_content
                
                if chunk.is_complete:
                    break
            
            total_time = (time.time() - start_time) * 1000
            time_to_first_chunk = (first_chunk_time - start_time) * 1000 if first_chunk_time else 0
            
            streaming_results["streaming_tests"]["query_1"] = {
                "query": query,
                "chunks_received": len(chunks_received),
                "time_to_first_chunk_ms": round(time_to_first_chunk, 2),
                "total_streaming_time_ms": round(total_time, 2),
                "final_content_length": len(total_content),
                "chunks_data": chunks_received[:5]  # First 5 chunks for analysis
            }
            
            streaming_results["avg_time_to_first_chunk_ms"] = round(time_to_first_chunk, 2)
            streaming_results["avg_total_streaming_time_ms"] = round(total_time, 2)
            
            print(f"📡 Streaming Results:")
            print(f"   Time to First Chunk: {time_to_first_chunk:.0f}ms")
            print(f"   Total Streaming Time: {total_time:.0f}ms")
            print(f"   Chunks Received: {len(chunks_received)}")
            
            return streaming_results
            
        except Exception as e:
            print(f"❌ Streaming test failed: {e}")
            return {"error": str(e)}
        finally:
            # Properly close streaming service
            if streaming_service:
                try:
                    await streaming_service.close()
                except Exception as e:
                    print(f"Warning: Error closing streaming service: {e}")

    def calculate_performance_improvements(self) -> Dict[str, Any]:
        """Calculate performance improvements from optimizations"""
        baseline = self.results.get("baseline_performance", {})
        optimized = self.results.get("optimized_performance", {})
        streaming = self.results.get("streaming_performance", {})
        
        if not baseline or not optimized:
            return {"error": "Missing baseline or optimized results"}
        
        improvements = {}
        
        # Response time improvements
        baseline_avg = baseline.get("avg_response_time_ms", 0)
        optimized_avg = optimized.get("avg_response_time_ms", 0)
        
        if baseline_avg > 0:
            speed_improvement = ((baseline_avg - optimized_avg) / baseline_avg) * 100
            time_saved_ms = baseline_avg - optimized_avg
            
            improvements["response_time"] = {
                "baseline_avg_ms": round(baseline_avg, 2),
                "optimized_avg_ms": round(optimized_avg, 2),
                "improvement_percentage": round(speed_improvement, 1),
                "time_saved_ms": round(time_saved_ms, 2),
                "improvement_ratio": round(baseline_avg / optimized_avg, 2) if optimized_avg > 0 else 0
            }
        
        # Streaming benefits
        if streaming:
            first_chunk_time = streaming.get("avg_time_to_first_chunk_ms", 0)
            if first_chunk_time > 0 and optimized_avg > 0:
                perceived_improvement = ((optimized_avg - first_chunk_time) / optimized_avg) * 100
                improvements["streaming"] = {
                    "time_to_first_chunk_ms": round(first_chunk_time, 2),
                    "perceived_improvement_percentage": round(perceived_improvement, 1),
                    "user_perceived_speedup": round(optimized_avg / first_chunk_time, 2) if first_chunk_time > 0 else 0
                }
            elif first_chunk_time > 0:
                improvements["streaming"] = {
                    "time_to_first_chunk_ms": round(first_chunk_time, 2),
                    "perceived_improvement_percentage": 0,
                    "user_perceived_speedup": 0,
                    "note": "Optimized test failed, cannot calculate improvement"
                }
        
        # Setup time comparison
        baseline_setup = baseline.get("setup_time_ms", 0)
        optimized_setup = optimized.get("setup_time_ms", 0)
        preload_time = optimized.get("preload_time_ms", 0)
        
        improvements["setup"] = {
            "baseline_setup_ms": round(baseline_setup, 2),
            "optimized_setup_ms": round(optimized_setup, 2),
            "preload_time_ms": round(preload_time, 2),
            "total_optimized_setup_ms": round(optimized_setup + preload_time, 2)
        }
        
        # Performance statistics
        perf_stats = optimized.get("performance_stats", {})
        if perf_stats:
            improvements["optimization_benefits"] = {
                "cache_hit_rate": perf_stats.get("cache_hit_rate", 0),
                "connection_reuses": perf_stats.get("connection_reuses", 0),
                "fastest_response_ms": perf_stats.get("fastest_response_ms", 0),
                "avg_response_time_ms": perf_stats.get("avg_response_time_ms", 0)
            }
        
        return improvements

    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete performance optimization validation"""
        print("🚀 PERFORMANCE OPTIMIZATION VALIDATION")
        print("=" * 80)
        print("Testing: Model Preloading, Pipeline Async, Connection Optimization, Streaming")
        print("=" * 80)
        
        # Setup
        if not self.setup_test_environment():
            self.results["error"] = "Failed to setup test environment"
            return self.results
        
        # Test baseline performance
        baseline_results = await self.test_baseline_performance()
        self.results["baseline_performance"] = baseline_results
        
        # Test optimized performance
        optimized_results = await self.test_optimized_performance()
        self.results["optimized_performance"] = optimized_results
        
        # Test streaming performance
        streaming_results = await self.test_streaming_performance()
        self.results["streaming_performance"] = streaming_results
        
        # Calculate improvements
        improvements = self.calculate_performance_improvements()
        self.results["performance_improvements"] = improvements
        
        return self.results

    def print_summary(self):
        """Print performance optimization summary"""
        print("\n" + "=" * 80)
        print("PERFORMANCE OPTIMIZATION VALIDATION SUMMARY")
        print("=" * 80)
        
        improvements = self.results.get("performance_improvements", {})
        
        if "response_time" in improvements:
            rt = improvements["response_time"]
            print(f"\n⚡ RESPONSE TIME IMPROVEMENTS:")
            print(f"   Baseline Average: {rt['baseline_avg_ms']:.0f}ms")
            print(f"   Optimized Average: {rt['optimized_avg_ms']:.0f}ms")
            print(f"   Improvement: {rt['improvement_percentage']:.1f}% faster")
            print(f"   Time Saved: {rt['time_saved_ms']:.0f}ms per query")
            print(f"   Speed Multiplier: {rt['improvement_ratio']:.1f}x")
        
        if "streaming" in improvements:
            st = improvements["streaming"]
            print(f"\n📡 STREAMING BENEFITS:")
            print(f"   Time to First Chunk: {st['time_to_first_chunk_ms']:.0f}ms")
            print(f"   Perceived Improvement: {st['perceived_improvement_percentage']:.1f}%")
            print(f"   User Perceived Speedup: {st['user_perceived_speedup']:.1f}x")
        
        if "optimization_benefits" in improvements:
            ob = improvements["optimization_benefits"]
            print(f"\n🔧 OPTIMIZATION STATISTICS:")
            print(f"   Cache Hit Rate: {ob['cache_hit_rate']:.1%}")
            print(f"   Connection Reuses: {ob['connection_reuses']}")
            print(f"   Fastest Response: {ob['fastest_response_ms']:.0f}ms")
        
        # Production readiness assessment
        baseline_avg = improvements.get("response_time", {}).get("baseline_avg_ms", 0)
        optimized_avg = improvements.get("response_time", {}).get("optimized_avg_ms", 0)
        
        if optimized_avg < 8000:  # Under 8 seconds
            print(f"\n✅ PRODUCTION READINESS: READY")
            print(f"   Optimized response time ({optimized_avg:.0f}ms) meets production requirements")
        else:
            print(f"\n⚠️ PRODUCTION READINESS: NEEDS IMPROVEMENT")
            print(f"   Optimized response time ({optimized_avg:.0f}ms) still above 8 second target")

    def save_results(self, filename: str = None) -> Path:
        """Save validation results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_optimization_validation_{timestamp}.json"
        
        results_file = self.project_root / filename
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        return results_file


async def main():
    """Main validation execution"""
    validator = PerformanceOptimizationValidator()
    
    # Run validation
    results = await validator.run_full_validation()
    
    # Print summary
    validator.print_summary()
    
    # Save results
    results_file = validator.save_results()
    
    return results, results_file


if __name__ == "__main__":
    results, results_file = asyncio.run(main())
