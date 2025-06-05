"""Test CLI startup after fixing missing orchestrator method."""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_app_orchestrator_has_splade_methods():
    """Test that AppOrchestrator has the required SPLADE methods."""
    from core.services.app_orchestrator import AppOrchestrator
    
    # Create orchestrator instance
    orchestrator = AppOrchestrator()
    
    # Verify both methods exist
    assert hasattr(orchestrator, 'set_splade_mode'), "set_splade_mode method should exist"
    assert hasattr(orchestrator, 'is_splade_mode_enabled'), "is_splade_mode_enabled method should exist"
    
    # Verify they are callable
    assert callable(orchestrator.set_splade_mode), "set_splade_mode should be callable"
    assert callable(orchestrator.is_splade_mode_enabled), "is_splade_mode_enabled should be callable"

def test_splade_mode_functionality():
    """Test that SPLADE mode can be set and retrieved."""
    import streamlit as st
    from core.services.app_orchestrator import AppOrchestrator
    
    # Create orchestrator instance
    orchestrator = AppOrchestrator()
    
    # Initially should be False (or default value)
    initial_state = orchestrator.is_splade_mode_enabled()
    assert isinstance(initial_state, bool), "is_splade_mode_enabled should return bool"
    
    # Test setting SPLADE mode to True
    orchestrator.set_splade_mode(True)
    assert orchestrator.is_splade_mode_enabled() == True, "SPLADE mode should be enabled"
    
    # Test setting SPLADE mode to False
    orchestrator.set_splade_mode(False)
    assert orchestrator.is_splade_mode_enabled() == False, "SPLADE mode should be disabled"

def test_config_sidebar_import():
    """Test that config sidebar can import without errors."""
    try:
        from utils.ui.config_sidebar import render_config_sidebar
        assert callable(render_config_sidebar), "render_config_sidebar should be callable"
        print("✅ Config sidebar import successful")
    except ImportError as e:
        pytest.fail(f"Failed to import config sidebar: {e}")

if __name__ == "__main__":
    print("=== Testing CLI Startup Fix ===")
    
    try:
        test_app_orchestrator_has_splade_methods()
        print("✅ AppOrchestrator SPLADE methods test passed")
        
        test_splade_mode_functionality() 
        print("✅ SPLADE mode functionality test passed")
        
        test_config_sidebar_import()
        print("✅ Config sidebar import test passed")
        
        print("\n🎯 All tests passed! CLI startup fix should work.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
