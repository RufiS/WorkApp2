"""Test script for Answer Quality Analyzer.

Quick validation that the analyzer can be imported and basic functionality works.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_analyzer_import():
    """Test that the analyzer can be imported."""
    try:
        from utils.testing.answer_quality_analyzer import AnswerQualityAnalyzer, run_text_message_analysis
        logger.info("✅ Successfully imported AnswerQualityAnalyzer")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import AnswerQualityAnalyzer: {e}")
        return False

def test_analyzer_initialization():
    """Test that analyzer can be initialized with mock objects."""
    try:
        from utils.testing.answer_quality_analyzer import AnswerQualityAnalyzer

        # Create realistic mock objects that match actual interfaces
        class MockRetrievalSystem:
            def retrieve(self, query):
                # Return realistic tuple: (context, retrieval_time, chunk_count, similarity_scores)
                mock_context = f"Mock context for query: {query}\n[Chunk 10]: RingCentral texting procedures\n[Chunk 56]: Text response workflow"
                return (mock_context, 0.25, 2, [0.85, 0.72])

            @property
            def document_processor(self):
                class MockDocProcessor:
                    texts = [
                        "RingCentral texting procedures for field engineers",
                        "SMS response workflow and ticket creation",
                        "Text message handling guidelines and best practices"
                    ]
                return MockDocProcessor()

        class MockLLMService:
            def generate_answer(self, query, context):
                return {
                    'content': f"Mock answer for query: {query}. Based on context: {context[:50]}...",
                    'usage': {'total_tokens': 150}
                }

        # Initialize analyzer
        mock_retrieval = MockRetrievalSystem()
        mock_llm = MockLLMService()

        analyzer = AnswerQualityAnalyzer(mock_retrieval, mock_llm)
        logger.info("✅ Successfully initialized AnswerQualityAnalyzer with mock objects")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize AnswerQualityAnalyzer: {e}")
        return False

def test_basic_analysis():
    """Test basic analysis functionality."""
    try:
        from utils.testing.answer_quality_analyzer import AnswerQualityAnalyzer

        # Create enhanced mock objects for comprehensive testing
        class MockRetrievalSystem:
            def retrieve(self, query):
                # Return realistic retrieval data with specific chunks for text message query
                if "text message" in query.lower():
                    mock_context = """[Chunk 10]: RingCentral texting procedures for dispatch
[Chunk 56]: SMS response workflow documentation
[Chunk 11]: Text message handling guidelines for field engineers"""
                    return (mock_context, 0.35, 3, [0.92, 0.88, 0.83])
                else:
                    mock_context = "General mock context for testing"
                    return (mock_context, 0.25, 1, [0.75])

            @property
            def document_processor(self):
                class MockDocProcessor:
                    texts = [
                        "RingCentral texting procedures for dispatch operations",
                        "SMS response workflow documentation and guidelines",
                        "Text message handling for field engineer communication",
                        "Freshdesk ticket creation from text messages",
                        "Customer communication protocols via SMS"
                    ]
                return MockDocProcessor()

        class MockLLMService:
            def generate_answer(self, query, context):
                # Generate realistic answer based on context
                answer_content = f"""To respond to a text message:
1. Open RingCentral application
2. Navigate to SMS/texting section
3. Locate the customer's incoming message
4. Review message content and context
5. Compose appropriate response following company guidelines
6. Send the response to the customer
7. Create a ticket in Freshdesk for tracking and documentation

Based on the provided context: {context[:100]}..."""

                return {
                    'content': answer_content,
                    'usage': {'total_tokens': 180},
                    'model': 'mock-gpt-4'
                }

        # Run basic analysis
        mock_retrieval = MockRetrievalSystem()
        mock_llm = MockLLMService()

        analyzer = AnswerQualityAnalyzer(mock_retrieval, mock_llm)

        result = analyzer.analyze_answer_completeness(
            query="How do I respond to a text message",
            expected_chunks=[10, 11, 12],
            expected_content_areas=["RingCentral", "SMS", "Text Response"]
        )

        # Validate result structure
        required_keys = ["query", "timestamp", "retrieval_analysis", "answer_analysis", "gap_analysis", "user_impact", "recommendations"]

        for key in required_keys:
            if key not in result:
                logger.error(f"❌ Missing key in result: {key}")
                return False

        logger.info("✅ Basic analysis completed successfully")
        logger.info(f"   - Query: {result['query']}")
        logger.info(f"   - Recommendations: {len(result.get('recommendations', []))} generated")

        return True

    except Exception as e:
        logger.error(f"❌ Basic analysis failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Testing Answer Quality Analyzer...")

    tests = [
        ("Import Test", test_analyzer_import),
        ("Initialization Test", test_analyzer_initialization),
        ("Basic Analysis Test", test_basic_analysis)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            logger.error(f"Test failed: {test_name}")

    logger.info(f"\n🎯 Test Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✅ All tests passed! Answer Quality Analyzer is ready for use.")
        return True
    else:
        logger.error("❌ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
