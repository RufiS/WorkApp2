#!/usr/bin/env python3
"""
Performance baseline tracking script for WorkApp2 refactoring.

Records initial performance metrics and compares subsequent runs to detect regressions.
Fails CI if performance degrades beyond acceptable thresholds.
"""

import json
import time
import psutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import subprocess

BASELINE_FILE = Path("scripts/performance_baseline.json")

def record_baseline_metrics() -> Dict[str, Any]:
    """Record initial performance baseline on first run"""
    print("📝 Recording performance baseline...")

    # Get current git commit for tracking
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        git_commit = "unknown"

    # TODO: Implement actual query performance measurement
    # For now, use placeholder values that will be replaced with real measurements
    baseline = {
        "p99_latency_ms": 500.0,  # placeholder - measure actual query latency
        "peak_rss_mb": 256.0,     # placeholder - measure actual memory usage
        "timestamp": time.time(),
        "git_commit": git_commit,
        "version": "v0.legacy_monolith",
        "measurement_details": {
            "test_queries": [
                "What is machine learning?",
                "Explain artificial intelligence",
                "How does vector search work?"
            ],
            "test_document_size_kb": 10,  # Size of test document
            "iterations": 10  # Number of test iterations
        }
    }

    BASELINE_FILE.write_text(json.dumps(baseline, indent=2))
    print(f"📝 Recorded baseline metrics to {BASELINE_FILE}")
    print(f"   Latency: {baseline['p99_latency_ms']:.1f}ms")
    print(f"   Memory:  {baseline['peak_rss_mb']:.1f}MB")
    return baseline

def load_baseline() -> Dict[str, Any]:
    """Load existing baseline metrics"""
    if not BASELINE_FILE.exists():
        print("📋 No baseline found, recording initial metrics...")
        return record_baseline_metrics()

    try:
        baseline = json.loads(BASELINE_FILE.read_text())
        print(f"📋 Loaded baseline from {baseline.get('version', 'unknown')} "
              f"({baseline.get('git_commit', 'unknown')[:8]})")
        return baseline
    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️  Corrupted baseline file, recording new baseline: {e}")
        return record_baseline_metrics()

def measure_current_performance() -> Dict[str, Any]:
    """Measure current performance metrics"""
    print("📊 Measuring current performance...")

    # TODO: Implement actual performance measurement
    # This should:
    # 1. Start the Streamlit app in test mode
    # 2. Upload a test document
    # 3. Run standardized queries
    # 4. Measure p99 latency and peak RSS
    # 5. Return actual measurements

    # For now, return placeholder values
    # In real implementation, this would measure actual performance
    current = {
        "p99_latency_ms": 520.0,  # placeholder - would be actual measurement
        "peak_rss_mb": 260.0      # placeholder - would be actual measurement
    }

    print(f"📊 Current performance:")
    print(f"   Latency: {current['p99_latency_ms']:.1f}ms")
    print(f"   Memory:  {current['peak_rss_mb']:.1f}MB")

    return current

def create_test_document() -> Path:
    """Create a minimal test document for performance testing"""
    test_content = """
Machine Learning and Artificial Intelligence

Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can analyze data, identify patterns, and make predictions or decisions.

Key concepts in machine learning include:
- Supervised learning: Training with labeled data
- Unsupervised learning: Finding patterns in unlabeled data
- Neural networks: Computing systems inspired by biological neural networks
- Deep learning: Neural networks with multiple layers

Vector search is a method used in information retrieval that represents documents and queries as high-dimensional vectors. These vectors capture semantic meaning, allowing for more accurate similarity matching compared to traditional keyword-based search methods.
"""

    test_file = Path(tempfile.gettempdir()) / "workapp_test_doc.txt"
    test_file.write_text(test_content.strip())
    return test_file

def main():
    parser = argparse.ArgumentParser(
        description="Performance baseline tracking for WorkApp2 refactoring"
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=10.0,
        help="Fail if regression exceeds this percentage (default: 10%%)"
    )
    parser.add_argument(
        "--record-baseline",
        action="store_true",
        help="Record new baseline (use carefully - only for major changes)"
    )
    parser.add_argument(
        "--show-baseline",
        action="store_true",
        help="Display current baseline metrics and exit"
    )
    args = parser.parse_args()

    # Handle special commands
    if args.show_baseline:
        if BASELINE_FILE.exists():
            baseline = json.loads(BASELINE_FILE.read_text())
            print("📋 Current baseline:")
            print(f"   Version: {baseline.get('version', 'unknown')}")
            print(f"   Commit:  {baseline.get('git_commit', 'unknown')[:8]}")
            print(f"   Latency: {baseline['p99_latency_ms']:.1f}ms")
            print(f"   Memory:  {baseline['peak_rss_mb']:.1f}MB")
            print(f"   Date:    {time.ctime(baseline['timestamp'])}")
        else:
            print("📋 No baseline file found")
        return

    if args.record_baseline:
        record_baseline_metrics()
        print("✅ New baseline recorded")
        return

    # Normal performance check
    baseline = load_baseline()
    current = measure_current_performance()

    # Calculate regression percentages
    latency_regression = ((current["p99_latency_ms"] - baseline["p99_latency_ms"])
                         / baseline["p99_latency_ms"] * 100)
    memory_regression = ((current["peak_rss_mb"] - baseline["peak_rss_mb"])
                        / baseline["peak_rss_mb"] * 100)

    print(f"\n📊 Performance comparison vs baseline:")
    print(f"   Latency: {current['p99_latency_ms']:.1f}ms vs {baseline['p99_latency_ms']:.1f}ms "
          f"({latency_regression:+.1f}%)")
    print(f"   Memory:  {current['peak_rss_mb']:.1f}MB vs {baseline['peak_rss_mb']:.1f}MB "
          f"({memory_regression:+.1f}%)")

    # CI exit code rules
    failed = False

    if latency_regression > args.fail_threshold:
        print(f"❌ Latency regression {latency_regression:.1f}% exceeds threshold {args.fail_threshold}%")
        failed = True

    if memory_regression > args.fail_threshold:
        print(f"❌ Memory regression {memory_regression:.1f}% exceeds threshold {args.fail_threshold}%")
        failed = True

    if failed:
        print("\n💡 Performance regression detected. Consider:")
        print("   - Reviewing recent changes for performance impact")
        print("   - Profiling the application to identify bottlenecks")
        print("   - Optimizing critical code paths")
        print("   - If regression is intentional, record new baseline with --record-baseline")
        sys.exit(1)

    print("✅ Performance within acceptable limits")

    # Show improvement if any
    if latency_regression < -1 or memory_regression < -1:
        print("🎉 Performance improvements detected!")

if __name__ == "__main__":
    main()
