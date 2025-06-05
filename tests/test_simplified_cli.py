"""Test the simplified command-line interface for WorkApp3."""

import argparse
import sys
from unittest.mock import patch
import pytest

# Import the parse_args function directly
sys.path.insert(0, '/workspace/llm/WorkApp2')
from workapp3 import parse_args


def test_default_arguments():
    """Test default argument parsing."""
    with patch('sys.argv', ['workapp3.py']):
        args = parse_args()
        assert args.mode == "development"
        assert args.features == []
        assert args.dry_run is False


def test_production_mode():
    """Test production mode positional argument."""
    with patch('sys.argv', ['workapp3.py', 'production']):
        args = parse_args()
        assert args.mode == "production"
        assert args.features == []


def test_development_with_splade():
    """Test development mode with SPLADE feature."""
    with patch('sys.argv', ['workapp3.py', 'development', 'splade']):
        args = parse_args()
        assert args.mode == "development"
        assert "splade" in args.features


def test_production_with_splade():
    """Test production mode with SPLADE feature."""
    with patch('sys.argv', ['workapp3.py', 'production', 'splade']):
        args = parse_args()
        assert args.mode == "production"
        assert "splade" in args.features




def test_dry_run_flag():
    """Test dry-run flag."""
    with patch('sys.argv', ['workapp3.py', '--dry-run', 'production']):
        args = parse_args()
        assert args.dry_run is True
        assert args.mode == "production"


if __name__ == "__main__":
    # Quick test runner
    print("Testing simplified CLI...")
    
    test_default_arguments()
    print("✅ Default arguments test passed")
    
    test_production_mode()
    print("✅ Production mode test passed")
    
    test_development_with_splade()
    print("✅ Development with SPLADE test passed")
    
    test_production_with_splade()
    print("✅ Production with SPLADE test passed")
    
    test_dry_run_flag()
    print("✅ Dry-run flag test passed")
    
    print("\n🎉 All simplified CLI tests passed!")
