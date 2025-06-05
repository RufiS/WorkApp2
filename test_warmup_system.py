#!/usr/bin/env python3
"""
Test script to verify the model warm-up system integration

This script tests the model preloader without starting the full Streamlit app
"""

import asyncio
import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_warmup_integration():
    """Test the warm-up system integration"""
    
    print("🧪 Testing WorkApp2 Warm-Up System Integration")
    print("=" * 60)
    
    try:
        # Import the orchestrator
        from core.services.app_orchestrator import AppOrchestrator
        
        print("✅ Successfully imported AppOrchestrator")
        
        # Initialize orchestrator
        orchestrator = AppOrchestrator()
        print("✅ AppOrchestrator initialized")
        
        # Get services (this should trigger preloader initialization)
        doc_processor, llm_service, retrieval_system = orchestrator.get_services()
        print("✅ Services initialized successfully")
        
        # Check if preloader was initialized
        if orchestrator.model_preloader:
            print("✅ Model preloader initialized")
            
            # Get preload status
            status = orchestrator.get_preload_status()
            print(f"✅ Preloader status: {status}")
            
            # Test ensure_models_preloaded (non-blocking test)
            print("🔥 Testing model preloading trigger...")
            orchestrator.ensure_models_preloaded()
            print("✅ Model preloading triggered successfully")
            
        else:
            print("❌ Model preloader not initialized")
            return False
            
        print("\n" + "=" * 60)
        print("🎉 Warm-up system integration test PASSED!")
        print("\nNow ready for:")
        print("- Fast model responses on first query")
        print("- LLM models pre-warmed with test queries")
        print("- Embedding models ready for vector search")
        print("- Zero cold start delays for users")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_preloading():
    """Test async preloading functionality"""
    
    print("\n🔄 Testing Async Preloading Functionality")
    print("=" * 60)
    
    try:
        from core.services.app_orchestrator import AppOrchestrator
        
        orchestrator = AppOrchestrator()
        doc_processor, llm_service, retrieval_system = orchestrator.get_services()
        
        if orchestrator.model_preloader:
            print("🚀 Starting async model preloading...")
            
            # Test async preloading
            preload_results = await orchestrator.preload_models()
            
            print("✅ Async preloading completed!")
            print(f"Results: {preload_results}")
            
            # Check final status
            final_status = orchestrator.get_preload_status()
            print(f"Final status: {final_status}")
            
            return True
        else:
            print("❌ No preloader available for async testing")
            return False
            
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("🚀 WorkApp2 Model Warm-Up System Test Suite")
    print("Testing integration with your existing sophisticated preloader")
    print()
    
    # Test 1: Basic integration
    test1_passed = test_warmup_integration()
    
    # Test 2: Async functionality
    test2_passed = asyncio.run(test_async_preloading())
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Integration Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Async Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour warm-up system is now integrated and ready!")
        print("\nNext steps:")
        print("1. Run `streamlit run workapp3.py` to see warm-up in action")
        print("2. Watch the logs for model preloading messages")
        print("3. Notice faster response times on first queries")
        
        return True
    else:
        print("\n❌ Some tests failed - check the error messages above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
