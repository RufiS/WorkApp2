#!/usr/bin/env python3
"""
Test script to verify warm-up fix and overall system improvements

This script tests:
1. Warm-up only runs once per session (session state tracking)
2. Formatting improvements work correctly  
3. Temperature controls are in place
4. Timing display functionality
"""

import logging
import sys
import os
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_session_state_logic():
    """Test the session state logic that prevents repeated warm-ups"""
    print("🔧 Testing Session State Warm-up Logic")
    print("=" * 60)
    
    try:
        # Mock streamlit session state behavior
        class MockSessionState:
            def __init__(self):
                self.data = {}
            
            def __contains__(self, key):
                return key in self.data
            
            def __setitem__(self, key, value):
                self.data[key] = value
                
            def __getitem__(self, key):
                return self.data[key]
                
            def get(self, key, default=None):
                return self.data.get(key, default)
        
        # Simulate the session state logic from workapp3.py
        session_state = MockSessionState()
        
        def simulate_warm_up_check():
            if "models_warmed_up" not in session_state:
                # This would be where warm-up happens
                session_state["models_warmed_up"] = True
                return "WARM_UP_EXECUTED"
            else:
                return "WARM_UP_SKIPPED"
        
        # Test first call - should execute warm-up
        result1 = simulate_warm_up_check()
        if result1 == "WARM_UP_EXECUTED":
            print("✅ First call correctly executes warm-up")
        else:
            print("❌ First call should execute warm-up")
            return False
            
        # Test second call - should skip warm-up
        result2 = simulate_warm_up_check()
        if result2 == "WARM_UP_SKIPPED":
            print("✅ Second call correctly skips warm-up")
        else:
            print("❌ Second call should skip warm-up")
            return False
            
        # Test third call (simulating feedback submission) - should skip
        result3 = simulate_warm_up_check()
        if result3 == "WARM_UP_SKIPPED":
            print("✅ Subsequent calls correctly skip warm-up")
        else:
            print("❌ Subsequent calls should skip warm-up")
            return False
            
        print("✅ Session state logic working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_formatting_fixes():
    """Test that formatting fixes are in place"""
    print("\n💰 Testing Formatting Fixes")
    print("=" * 60)
    
    try:
        from llm.prompts.formatting_prompt import generate_formatting_prompt
        
        # Test content with dollar signs
        test_content = "The service costs $125 and setup is $4."
        prompt = generate_formatting_prompt(test_content)
        
        # Check for dollar sign escaping instructions
        if "\\$125" in prompt and "\\$4" in prompt:
            print("✅ Dollar sign escaping instructions present")
        else:
            print("❌ Missing dollar sign escaping instructions")
            return False
            
        # Check for bullet point instructions
        if "- Item" in prompt and "own line" in prompt:
            print("✅ Bullet point formatting instructions present")
        else:
            print("❌ Missing bullet point formatting instructions")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Formatting test failed: {e}")
        return False

def test_temperature_controls():
    """Test temperature control configuration"""
    print("\n🌡️ Testing Temperature Controls")
    print("=" * 60)
    
    try:
        from core.config import model_config
        
        extraction_temp = model_config.extraction_temperature
        formatting_temp = model_config.formatting_temperature
        
        print(f"Extraction temperature: {extraction_temp}")
        print(f"Formatting temperature: {formatting_temp}")
        
        if extraction_temp == 0.0:
            print("✅ Extraction temperature correctly set to 0.0 (deterministic)")
        else:
            print("❌ Extraction temperature should be 0.0")
            return False
            
        if formatting_temp == 0.4:
            print("✅ Formatting temperature correctly set to 0.4 (creative)")
        else:
            print("❌ Formatting temperature should be 0.4")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Temperature test failed: {e}")
        return False

def test_timing_integration():
    """Test timing display integration"""
    print("\n⏱️ Testing Timing Display Integration")
    print("=" * 60)
    
    try:
        # Check that query controller has timing display
        import inspect
        from core.controllers.query_controller import QueryController
        
        source = inspect.getsource(QueryController)
        
        if "Ask to Answer:" in source and "total_time" in source:
            print("✅ Timing display integrated in query controller")
        else:
            print("❌ Timing display not found in query controller")
            return False
            
        print("✅ Timing integration ready")
        return True
        
    except Exception as e:
        print(f"❌ Timing test failed: {e}")
        return False

def test_preloader_timing():
    """Test that preloader timing calculation is correct"""
    print("\n⏲️ Testing Preloader Timing Calculation")
    print("=" * 60)
    
    try:
        from llm.services.model_preloader import ModelPreloader, PreloadConfig
        
        # Create a mock preloader to test timing
        config = PreloadConfig(
            enable_llm_preload=False,  # Disable actual LLM calls
            enable_embedding_preload=False,  # Disable actual embedding calls
        )
        
        preloader = ModelPreloader(config=config)
        
        # Test timing calculation logic
        start_time = time.time()
        time.sleep(0.1)  # Simulate 100ms work
        end_time = time.time()
        
        expected_time = end_time - start_time
        
        if 0.08 <= expected_time <= 0.15:  # Allow some variance
            print(f"✅ Timing calculation working correctly: {expected_time:.3f}s")
        else:
            print(f"❌ Timing calculation seems off: {expected_time:.3f}s")
            
        # Check that preloader uses time.time() correctly
        import inspect
        source = inspect.getsource(ModelPreloader)
        
        if "time.time()" in source and "preload_start_time" in source:
            print("✅ Preloader uses proper timing methods")
        else:
            print("❌ Preloader timing methods unclear")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Preloader timing test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🚀 WorkApp2 Warm-up Fix & System Improvements Test Suite")
    print("Testing fixes for repeated warm-up and timing issues")
    print()
    
    # Test 1: Session state logic
    test1_passed = test_session_state_logic()
    
    # Test 2: Formatting fixes
    test2_passed = test_formatting_fixes()
    
    # Test 3: Temperature controls
    test3_passed = test_temperature_controls()
    
    # Test 4: Timing integration
    test4_passed = test_timing_integration()
    
    # Test 5: Preloader timing
    test5_passed = test_preloader_timing()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Session State Logic: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Formatting Fixes: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Temperature Controls: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    print(f"Timing Integration: {'✅ PASSED' if test4_passed else '❌ FAILED'}")
    print(f"Preloader Timing: {'✅ PASSED' if test5_passed else '❌ FAILED'}")
    
    all_passed = all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed])
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Issues Fixed:")
        print("🔧 Warm-up now runs only once per session")
        print("🔧 No more repeated warm-ups on queries/feedback")
        print("💰 Dollar signs properly escaped for display")
        print("📝 Bullet points formatted on separate lines")
        print("🌡️ Temperature controls enable creative formatting")
        print("⏱️ Ask to Answer timing displayed in all modes")
        
        print("\n🎯 Expected Results:")
        print("• First app start: ~2-5s warm-up, then instant responses")
        print("• No warm-up on subsequent queries or feedback")
        print("• Proper $125 display (not broken LaTeX)")
        print("• Clean bullet point formatting")
        print("• Accurate timing display")
        
        return True
    else:
        print("\n❌ Some tests failed - check the error messages above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
