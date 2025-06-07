#!/usr/bin/env python3
"""
GPU Resource Management Test
============================

Tests the new GPU resource management system that prevents
testing framework from displacing Ollama to CPU.
"""

import os
import sys
import time
import requests
from pathlib import Path

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def print_test(message: str):
    print(f"🔧 {message}")

def print_success(message: str):
    print(f"✅ {message}")

def print_warning(message: str):
    print(f"⚠️ {message}")

def print_error(message: str):
    print(f"❌ {message}")

def test_gpu_resource_config():
    """Test that GPU resource management config is properly loaded."""
    print_test("Testing GPU resource management configuration...")
    
    try:
        from core.config import performance_config
        gpu_config = performance_config.gpu_resource_management
        
        # Verify key settings
        ollama_reservation = gpu_config.get("ollama_vram_reservation_mb", 0)
        max_workers_per_gb = gpu_config.get("max_workers_per_gb", 0)
        min_vram_per_worker = gpu_config.get("min_vram_per_worker_gb", 0)
        
        if ollama_reservation >= 12288:  # 12GB
            print_success(f"✅ Ollama VRAM reservation: {ollama_reservation}MB (≥12GB)")
        else:
            print_error(f"❌ Ollama VRAM reservation too low: {ollama_reservation}MB (<12GB)")
            return False
        
        if max_workers_per_gb <= 0.5:  # Conservative scaling
            print_success(f"✅ Worker scaling factor: {max_workers_per_gb} workers/GB (conservative)")
        else:
            print_warning(f"⚠️ Worker scaling factor high: {max_workers_per_gb} workers/GB")
        
        if min_vram_per_worker >= 3.0:  # Minimum 3GB per worker
            print_success(f"✅ Minimum VRAM per worker: {min_vram_per_worker}GB")
        else:
            print_error(f"❌ Minimum VRAM per worker too low: {min_vram_per_worker}GB")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Failed to load GPU resource config: {e}")
        return False

def test_vram_calculation():
    """Test VRAM calculation and worker allocation logic."""
    print_test("Testing VRAM calculation and worker allocation...")
    
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print_warning("CUDA not available, skipping VRAM tests")
        return True
    
    try:
        from tests.test_robust_evaluation_framework import RobustEvaluationFramework
        
        # Create framework instance to test worker calculation
        framework = RobustEvaluationFramework(max_configs=1, clean_cache=False)
        
        # Test worker calculation
        max_workers, available_vram, can_use_gpu = framework._calculate_optimal_workers()
        
        print_success(f"Available VRAM: {available_vram:.1f}GB")
        print_success(f"Calculated workers: {max_workers}")
        print_success(f"GPU enabled: {can_use_gpu}")
        
        # Verify reasonable worker allocation (should be ≤2 to leave room for Ollama)
        if max_workers <= 2:
            print_success(f"✅ Worker count conservative: {max_workers} workers (≤2)")
        else:
            print_error(f"❌ Too many workers: {max_workers} (might displace Ollama)")
            return False
        
        # If GPU is enabled, verify per-worker allocation is reasonable
        if can_use_gpu and available_vram > 12:  # If we have enough VRAM
            vram_per_worker = (available_vram - 12 - 1) / max_workers  # Reserve 12GB for Ollama + 1GB buffer
            if vram_per_worker >= 3.0:
                print_success(f"✅ VRAM per worker: {vram_per_worker:.1f}GB (≥3GB)")
            else:
                print_warning(f"⚠️ Low VRAM per worker: {vram_per_worker:.1f}GB")
        
        return True
        
    except Exception as e:
        print_error(f"VRAM calculation test failed: {e}")
        return False

def test_ollama_connectivity():
    """Test Ollama server connectivity and model availability."""
    print_test("Testing Ollama connectivity...")
    
    try:
        # Test Ollama API connectivity
        response = requests.get("http://192.168.254.204:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            print_success("✅ Ollama server accessible")
            print_success(f"Available models: {len(model_names)}")
            
            # Check for the specific model we need
            qwen_models = [name for name in model_names if 'qwen2.5:14b-instruct' in name.lower()]
            if qwen_models:
                print_success(f"✅ Target model available: {qwen_models[0]}")
                return True
            else:
                print_warning(f"⚠️ qwen2.5:14b-instruct not found in: {model_names}")
                return False
        else:
            print_error(f"❌ Ollama API error: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"❌ Ollama connectivity failed: {e}")
        return False

def test_gpu_memory_status():
    """Test current GPU memory status."""
    print_test("Testing GPU memory status...")
    
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print_warning("CUDA not available")
        return True
    
    try:
        # Get GPU memory stats
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)  # GB
        
        available_memory = total_memory - max(allocated_memory, reserved_memory)
        
        print_success(f"GPU: {torch.cuda.get_device_name(0)}")
        print_success(f"Total VRAM: {total_memory:.1f}GB")
        print_success(f"Allocated: {allocated_memory:.1f}GB")
        print_success(f"Reserved: {reserved_memory:.1f}GB")
        print_success(f"Available: {available_memory:.1f}GB")
        
        # Check if there's enough room for both Ollama and testing
        if total_memory >= 20.0:  # At least 20GB total
            print_success("✅ Sufficient total VRAM for Ollama + testing")
        else:
            print_warning(f"⚠️ Limited total VRAM: {total_memory:.1f}GB")
        
        if available_memory >= 15.0:  # Enough for Ollama (12GB) + testing (3GB+)
            print_success("✅ Sufficient available VRAM for coexistence")
        else:
            print_warning(f"⚠️ Limited available VRAM: {available_memory:.1f}GB")
        
        return True
        
    except Exception as e:
        print_error(f"GPU memory status failed: {e}")
        return False

def simulate_worker_allocation():
    """Simulate the new worker allocation to show the improvement."""
    print_test("Simulating worker allocation comparison...")
    
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print_warning("CUDA not available, showing theoretical calculations")
        total_vram = 24.0  # RTX 3090 Ti
        available_vram = 22.0  # Assume 2GB in use
    else:
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        available_vram = total_vram - max(allocated, reserved)
    
    print_success(f"GPU Analysis: {total_vram:.1f}GB total, {available_vram:.1f}GB available")
    
    # OLD ALLOCATION (problematic)
    print_warning("❌ OLD ALLOCATION:")
    print_warning("  - 3 workers × 8GB = 24GB (monopolizes entire GPU)")
    print_warning("  - Ollama forced to CPU")
    print_warning("  - Poor LLM performance")
    
    # NEW ALLOCATION (fixed)
    ollama_reservation = 12.0  # GB
    safety_buffer = 1.0       # GB
    usable_for_testing = available_vram - ollama_reservation - safety_buffer
    
    if usable_for_testing > 0:
        new_workers = max(1, min(2, int(usable_for_testing * 0.25)))  # 0.25 workers per GB
        vram_per_worker = usable_for_testing / new_workers if new_workers > 0 else 0
        
        print_success("✅ NEW ALLOCATION:")
        print_success(f"  - Ollama reservation: {ollama_reservation:.1f}GB")
        print_success(f"  - Safety buffer: {safety_buffer:.1f}GB")
        print_success(f"  - Available for testing: {usable_for_testing:.1f}GB")
        print_success(f"  - Workers: {new_workers} × {vram_per_worker:.1f}GB = {new_workers * vram_per_worker:.1f}GB")
        print_success("  - Ollama stays on GPU ✅")
        print_success("  - Testing still GPU-accelerated ✅")
        
        return True
    else:
        print_error("❌ Not enough VRAM for coexistence - will fall back to CPU workers")
        return False

def main():
    """Run all GPU resource management tests."""
    print("=" * 60)
    print("🔧 GPU RESOURCE MANAGEMENT TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_gpu_resource_config),
        ("VRAM Calculation", test_vram_calculation),
        ("Ollama Connectivity", test_ollama_connectivity),
        ("GPU Memory Status", test_gpu_memory_status),
        ("Worker Allocation Simulation", simulate_worker_allocation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print()
        print(f"🧪 {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print_success(f"✅ {test_name} PASSED")
            else:
                print_error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print_error(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print()
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
    
    print("-" * 40)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print()
        print_success("🎯 ALL TESTS PASSED!")
        print_success("GPU resource management is working correctly.")
        print_success("Ollama should now be able to coexist with testing framework.")
    else:
        print()
        print_error("⚠️ SOME TESTS FAILED!")
        print_error("Review the failures above and check your configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
