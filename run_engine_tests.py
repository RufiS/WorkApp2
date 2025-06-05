#!/usr/bin/env python3
"""
Systematic Engine Testing Launcher
================================

Simple launcher script for the comprehensive engine testing framework.
Provides an easy interface to run tests and analyze results.
"""

import sys
import subprocess
import argparse
from pathlib import Path
import json
from datetime import datetime

def print_banner():
    """Print the testing framework banner."""
    print("=" * 60)
    print("   WorkApp2 Systematic Engine Testing Framework")
    print("=" * 60)
    print()

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'pandas', 'matplotlib', 'seaborn', 'numpy', 'transformers', 'torch'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def list_available_results():
    """List available test result files."""
    test_logs_dir = Path("test_logs")
    if not test_logs_dir.exists():
        print("❌ No test_logs directory found. Run tests first.")
        return []
    
    # Find JSON result files
    json_files = list(test_logs_dir.glob("*_summary_*.json"))
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not json_files:
        print("❌ No test result files found. Run tests first.")
        return []
    
    print("📊 Available test result files:")
    for i, file in enumerate(json_files):
        timestamp = file.stat().st_mtime
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        print(f"   {i+1}. {file.name} ({date_str})")
    
    return json_files

def run_quick_test():
    """Run quick evaluation test."""
    print("🚀 Running Quick Evaluation Test...")
    print("This tests 5 key configurations and takes 15-30 minutes.")
    print()
    
    try:
        result = subprocess.run([
            sys.executable, "tests/test_systematic_engine_evaluation.py", "--mode", "quick"
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Quick test completed successfully!")
            return True
        else:
            print(f"\n❌ Test failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive evaluation test."""
    print("🚀 Running Comprehensive Evaluation Test...")
    print("⚠️  WARNING: This will take 4-8 hours to complete!")
    print()
    
    confirm = input("Do you want to continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Test cancelled.")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, "tests/test_systematic_engine_evaluation.py", "--mode", "comprehensive"
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Comprehensive test completed successfully!")
            return True
        else:
            print(f"\n❌ Test failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def analyze_results(result_file=None):
    """Analyze test results."""
    if result_file is None:
        # Let user choose from available files
        files = list_available_results()
        if not files:
            return False
        
        print()
        try:
            choice = int(input("Select a file to analyze (number): ")) - 1
            if 0 <= choice < len(files):
                result_file = files[choice]
            else:
                print("❌ Invalid selection")
                return False
        except ValueError:
            print("❌ Invalid input")
            return False
    else:
        result_file = Path(result_file)
        if not result_file.exists():
            print(f"❌ File not found: {result_file}")
            return False
    
    print(f"📊 Analyzing results from: {result_file.name}")
    
    try:
        result = subprocess.run([
            sys.executable, "tests/test_results_analyzer.py", str(result_file)
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Analysis completed successfully!")
            print("📁 Check test_logs/analysis/ for detailed reports and visualizations")
            return True
        else:
            print(f"\n❌ Analysis failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False

def show_status():
    """Show testing framework status."""
    print("📋 Testing Framework Status")
    print()
    
    # Check test logs directory
    test_logs_dir = Path("test_logs")
    if test_logs_dir.exists():
        print("✅ Test logs directory exists")
        
        # Count result files
        json_files = list(test_logs_dir.glob("*_summary_*.json"))
        print(f"📊 {len(json_files)} test result files found")
        
        # Check analysis directory
        analysis_dir = test_logs_dir / "analysis"
        if analysis_dir.exists():
            reports = list(analysis_dir.glob("analysis_report_*.md"))
            plots = list(analysis_dir.glob("*.png"))
            print(f"📈 {len(reports)} analysis reports, {len(plots)} visualizations")
        else:
            print("📈 No analysis results yet")
    else:
        print("❌ Test logs directory not found")
    
    print()
    
    # Check main framework files
    framework_files = [
        "tests/test_systematic_engine_evaluation.py",
        "tests/test_results_analyzer.py", 
        "tests/QAexamples.json",
        "logs/feedback_detailed.log"
    ]
    
    print("🔧 Framework Components:")
    for file_path in framework_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (missing)")

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="WorkApp2 Systematic Engine Testing Framework Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_engine_tests.py test quick          # Run quick evaluation
  python run_engine_tests.py test comprehensive  # Run full evaluation
  python run_engine_tests.py analyze             # Analyze latest results
  python run_engine_tests.py status              # Show framework status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run engine tests')
    test_parser.add_argument('mode', choices=['quick', 'comprehensive'], 
                           help='Test mode to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze test results')
    analyze_parser.add_argument('--file', type=str, 
                               help='Specific result file to analyze')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show framework status')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check requirements')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.command == 'check':
        check_requirements()
    
    elif args.command == 'status':
        show_status()
    
    elif args.command == 'test':
        if not check_requirements():
            sys.exit(1)
        
        # Create test_logs directory if it doesn't exist
        Path("test_logs").mkdir(exist_ok=True)
        
        if args.mode == 'quick':
            success = run_quick_test()
        else:
            success = run_comprehensive_test()
        
        if success:
            print("\n🎯 Next steps:")
            print("   1. Analyze results: python run_engine_tests.py analyze")
            print("   2. Check test_logs/ directory for detailed files")
        
        sys.exit(0 if success else 1)
    
    elif args.command == 'analyze':
        if not check_requirements():
            sys.exit(1)
        
        success = analyze_results(args.file)
        sys.exit(0 if success else 1)
    
    else:
        print("Available commands:")
        print("  test quick          - Run quick evaluation (15-30 min)")
        print("  test comprehensive  - Run full evaluation (4-8 hours)")
        print("  analyze            - Analyze test results")
        print("  status             - Show framework status")
        print("  check              - Check requirements")
        print()
        print("For detailed help: python run_engine_tests.py --help")

if __name__ == "__main__":
    main()
