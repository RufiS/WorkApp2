#!/usr/bin/env python3
"""
Test script to verify API key loading from secrets/API-Keys.env file
"""

import os
import sys
import tempfile
import shutil

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_key_loading():
    """Test that API keys are loaded correctly from the API-Keys.env file"""
    
    print("🔑 Testing API Key Loading from secrets/API-Keys.env")
    print("=" * 60)
    
    try:
        # Import after setting up the path
        from core.config import app_config
        
        # Check if API keys are loaded
        api_keys = app_config.api_keys
        print(f"📋 API Keys Found: {list(api_keys.keys())}")
        
        # Check for OpenAI key specifically
        if 'openai' in api_keys:
            key = api_keys['openai']
            # Show first and last 4 characters for security
            masked_key = f"{key[:8]}...{key[-8:]}" if len(key) > 16 else "***masked***"
            print(f"✅ OpenAI API Key: {masked_key}")
            print(f"✅ Key length: {len(key)} characters")
        else:
            print("❌ OpenAI API Key not found")
            
        # Check if the file exists
        env_file_path = os.path.join("secrets", "API-Keys.env")
        if os.path.exists(env_file_path):
            print(f"✅ API Keys file exists: {env_file_path}")
        else:
            print(f"❌ API Keys file missing: {env_file_path}")
            
        # Check environment fallback
        env_key = os.getenv('OPENAI_API_KEY')
        if env_key:
            print(f"✅ Environment variable OPENAI_API_KEY is set")
        else:
            print(f"ℹ️  Environment variable OPENAI_API_KEY not set (using file)")
            
        return len(api_keys) > 0
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def test_config_no_api_keys():
    """Test that config.json no longer contains API keys"""
    
    print("\n🔒 Testing config.json Security")
    print("=" * 40)
    
    try:
        with open("config.json", "r") as f:
            config_content = f.read()
            
        if "api_key" in config_content.lower() or "openai" in config_content:
            print("❌ config.json still contains API key references")
            return False
        else:
            print("✅ config.json is clean (no API keys)")
            return True
            
    except Exception as e:
        print(f"❌ Error reading config.json: {e}")
        return False

if __name__ == "__main__":
    print("🚀 WorkApp2 API Key Security Test")
    print("=" * 80)
    
    test1_passed = test_api_key_loading()
    test2_passed = test_config_no_api_keys()
    
    print("\n" + "=" * 80)
    if test1_passed and test2_passed:
        print("✅ All tests passed! API key security migration successful.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the configuration.")
        sys.exit(1)
