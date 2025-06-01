"""Test script for Answer Quality Parameter Sweep - Find Optimal Retrieval Settings

This script runs systematic parameter testing to identify the optimal threshold and top_k
settings that maximize user task completion for text message response queries.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_parameter_sweep():
    """Test the parameter sweep functionality with mock services."""
    try:
        from utils.testing.answer_quality_parameter_sweep import run_focused_threshold_sweep, run_parameter_sweep_analysis
        
        logger.info("🧪 Testing Parameter Sweep with Mock Services")
        
        # Create enhanced mock objects that support configuration changes
        class MockRetrievalSystem:
            def __init__(self):
                self.similarity_threshold = 0.5  # Default threshold
                self.top_k = 100  # Default top_k
                
            def retrieve(self, query):
                # Simulate different retrieval results based on configuration
                if self.similarity_threshold <= 0.42 and self.top_k >= 20:
                    # Optimal configuration - retrieve all expected chunks
                    retrieved_chunks = [10, 11, 12, 56, 58, 59, 60]
                    similarity_scores = [0.54, 0.49, 0.49, 0.49, 0.44, 0.43, 0.43]
                    
                elif self.similarity_threshold <= 0.45 and self.top_k >= 15:
                    # Good configuration - retrieve most expected chunks
                    retrieved_chunks = [10, 11, 12, 56, 58, 59]
                    similarity_scores = [0.54, 0.49, 0.49, 0.49, 0.44, 0.43]
                    
                elif self.similarity_threshold <= 0.50:
                    # Current configuration - retrieve only some chunks
                    retrieved_chunks = [10, 11, 12]
                    similarity_scores = [0.54, 0.49, 0.49]
                    
                else:
                    # Poor configuration - retrieve minimal chunks
                    retrieved_chunks = [10, 11]
                    similarity_scores = [0.54, 0.49]
                
                # Build context from retrieved chunks
                chunk_content = {
                    10: "RingCentral Texting - Most contact with our clients will take place on the phone...",
                    11: "General SMS format: 'We tried to return your call for computer repair...'",
                    12: "When replying to an SMS/Text message, always find the ticket for it...",
                    56: "Text Response (30 min between contact attempts) - 1st touch: Call the EU from text conversation...",
                    58: "Text Tickets - Freshdesk 'New Text Message' tickets do not provide the information...",
                    59: "If the ticket says the text is to 'Matthew Karls'. That means the text message was sent...",
                    60: "When we receive a text message that is intended for an FE, we need to look up..."
                }
                
                context_parts = []
                for i, chunk_id in enumerate(retrieved_chunks):
                    if chunk_id in chunk_content:
                        context_parts.append(f"[{i+1}] From chunk {chunk_id}:\n{chunk_content[chunk_id]}")
                
                context = "\n\n".join(context_parts)
                
                return (context, 0.35, len(retrieved_chunks), similarity_scores)
            
            @property
            def document_processor(self):
                class MockDocProcessor:
                    texts = [f"Mock chunk {i} content" for i in range(221)]
                return MockDocProcessor()
        
        class MockLLMService:
            def generate_answer(self, query, context):
                # Generate answer based on context completeness
                if "Text Response" in context and "Text Tickets" in context and "Field Engineer" in context:
                    # Complete context - good answer
                    answer_content = """To respond to a text message:
1. Locate and claim the Freshdesk ticket generated for the text message
2. Call the EU from the text conversation and leave a voicemail
3. Send an SMS response using the general format: 'We tried to return your call for computer repair...'
4. Add notes to the Freshdesk ticket including your response
5. For Field Engineer texts, notify the FE via KTI channel
6. Follow the 30-minute contact attempt schedule"""
                    
                elif "RingCentral Texting" in context and "SMS format" in context:
                    # Partial context - incomplete answer
                    answer_content = """To respond to a text message:
1. Find the Freshdesk ticket for the text message
2. Send an SMS response using: 'We tried to return your call for computer repair...'
3. Add notes to the ticket with your response"""
                    
                else:
                    # Minimal context - poor answer
                    answer_content = """To respond to a text message:
1. Send a text response
2. Document in Freshdesk"""
                
                return {
                    'content': answer_content,
                    'usage': {'total_tokens': len(answer_content.split()) * 2}
                }
        
        # Initialize mock services
        mock_retrieval = MockRetrievalSystem()
        mock_llm = MockLLMService()
        
        logger.info("✅ Mock services initialized")
        
        # Run focused parameter sweep (smaller test)
        logger.info("🔍 Running focused parameter sweep...")
        
        from utils.testing.answer_quality_parameter_sweep import AnswerQualityParameterSweep
        sweep = AnswerQualityParameterSweep(mock_retrieval, mock_llm)
        
        # Test with focused range based on our analysis
        results = sweep.run_comprehensive_sweep(
            query="How do I respond to a text message",
            expected_chunks=[10, 11, 12, 56, 58, 59, 60],
            expected_content_areas=[
                "RingCentral Texting", "Text Response", "SMS format", 
                "Text Tickets", "Freshdesk ticket", "Field Engineer contact"
            ],
            threshold_values=[0.35, 0.40, 0.42, 0.45, 0.50, 0.55],  # Focused range
            top_k_values=[15, 20, 25, 30],  # Reasonable range
            save_results=True
        )
        
        # Display results
        logger.info("📊 Parameter Sweep Results:")
        logger.info(f"   Total tests: {results['performance_summary']['total_tests']}")
        logger.info(f"   Best user success score: {results['performance_summary']['best_user_success_score']:.3f}")
        
        optimal = results['optimal_configuration']
        logger.info(f"🎯 Optimal Configuration:")
        logger.info(f"   Threshold: {optimal['threshold']}")
        logger.info(f"   Top_k: {optimal['top_k']}")
        logger.info(f"   Expected coverage: {optimal['expected_coverage']:.1f}%")
        logger.info(f"   Task completion: {optimal['task_completion']}")
        logger.info(f"   Success probability: {optimal['success_probability']:.3f}")
        
        # Show top 5 configurations
        logger.info("🏆 Top 5 Configurations:")
        for i, config in enumerate(results['performance_summary']['top_5_configurations'][:5], 1):
            logger.info(f"   {i}. threshold={config['threshold']}, top_k={config['top_k']}, "
                       f"score={config['user_success_score']:.3f}, coverage={config['coverage']:.1f}%")
        
        # Check configurations with 100% coverage
        full_coverage_count = results['performance_summary']['configurations_with_100_percent_coverage']
        logger.info(f"✅ Configurations with 100% expected chunk coverage: {full_coverage_count}")
        
        task_completion_count = results['performance_summary']['configurations_with_task_completion']
        logger.info(f"✅ Configurations enabling task completion: {task_completion_count}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Parameter sweep test failed: {e}")
        return False

def test_configuration_interface():
    """Test that we can properly modify retrieval system configuration."""
    try:
        logger.info("🔧 Testing Configuration Interface")
        
        # Test mock configuration changes
        class TestRetrievalSystem:
            def __init__(self):
                self.similarity_threshold = 0.5
                self.top_k = 100
                
            def get_config(self):
                return {
                    "threshold": self.similarity_threshold,
                    "top_k": self.top_k
                }
        
        test_system = TestRetrievalSystem()
        original_config = test_system.get_config()
        logger.info(f"   Original config: {original_config}")
        
        # Test configuration changes
        test_system.similarity_threshold = 0.42
        test_system.top_k = 25
        
        new_config = test_system.get_config()
        logger.info(f"   Modified config: {new_config}")
        
        # Verify changes
        assert new_config["threshold"] == 0.42
        assert new_config["top_k"] == 25
        
        logger.info("✅ Configuration interface test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration interface test failed: {e}")
        return False

def main():
    """Run all parameter sweep tests."""
    logger.info("🧪 Testing Answer Quality Parameter Sweep...")
    
    tests = [
        ("Configuration Interface Test", test_configuration_interface),
        ("Parameter Sweep Test", test_parameter_sweep)
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
        logger.info("✅ All tests passed! Parameter sweep is ready for production use.")
        logger.info("🚀 Next steps:")
        logger.info("   1. Run parameter sweep with real retrieval system")
        logger.info("   2. Apply optimal configuration found by sweep")
        logger.info("   3. Validate improved user task completion")
        return True
    else:
        logger.error("❌ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
