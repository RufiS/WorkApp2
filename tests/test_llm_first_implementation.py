"""
Comprehensive test suite for LLM-first implementation

Tests the regex-free JSON handling, formatting, and prompt generation
"""
import pytest
import json
from unittest.mock import Mock, patch

# Import the modules we're testing
from llm.pipeline.validation import validate_json_output
from llm.pipeline.llm_json_handler import LLMJSONHandler
from llm.prompts.formatting_prompt import generate_formatting_prompt, check_formatting_quality
from llm.prompt_generator import PromptGenerator


class TestJSONValidation:
    """Test the LLM-first JSON validation"""
    
    def test_valid_json_direct_parsing(self):
        """Test that valid JSON is parsed directly without LLM"""
        valid_json = '{"answer": "Test answer", "sources": [], "confidence": 0.9}'
        is_valid, parsed, error = validate_json_output(valid_json)
        
        assert is_valid is True
        assert parsed["answer"] == "Test answer"
        assert error is None
        
    def test_empty_content(self):
        """Test handling of empty content"""
        is_valid, parsed, error = validate_json_output("")
        
        assert is_valid is False
        assert parsed is None
        assert "Empty content" in error
        
    def test_malformed_json_with_llm_service(self):
        """Test LLM-based JSON repair"""
        # Mock LLM service
        mock_llm = Mock()
        mock_llm.generate.return_value = '{"answer": "Fixed answer", "sources": [], "confidence": 0.8}'
        
        malformed = 'answer: "Broken JSON", confidence: 0.8'
        is_valid, parsed, error = validate_json_output(malformed, llm_service=mock_llm)
        
        assert is_valid is True
        assert parsed["answer"] == "Fixed answer"
        assert mock_llm.generate.called
        
    def test_fallback_extraction(self):
        """Test fallback content extraction"""
        content = "This is just plain text, not JSON at all"
        is_valid, parsed, error = validate_json_output(content)
        
        assert is_valid is True
        assert "This is just plain text" in parsed["answer"]
        assert parsed["confidence"] == 0.5  # Low confidence for fallback
        
    def test_backward_compatibility(self):
        """Test that validate_json_output works without llm_service (backward compatibility)"""
        valid_json = '{"answer": "Test", "sources": [], "confidence": 0.9}'
        # Should work without llm_service parameter for backward compatibility
        is_valid, parsed, error = validate_json_output(valid_json)
        
        assert is_valid is True
        assert parsed["answer"] == "Test"


class TestLLMJSONHandler:
    """Test the pure LLM JSON handler"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.handler = LLMJSONHandler(self.mock_llm)
        
    def test_ensure_valid_json_direct_parse(self):
        """Test direct JSON parsing success"""
        valid_json = '{"answer": "Test", "sources": [], "confidence": 0.9}'
        success, parsed, error = self.handler.ensure_valid_json(valid_json, {})
        
        assert success is True
        assert parsed["answer"] == "Test"
        assert error is None
        
    def test_ensure_valid_json_with_repair(self):
        """Test JSON repair through LLM"""
        self.mock_llm.generate.return_value = '{"answer": "Repaired", "sources": [], "confidence": 0.7}'
        
        malformed = '{answer: "Broken", confidence: 0.7'
        success, parsed, error = self.handler.ensure_valid_json(malformed, {})
        
        assert success is True
        assert parsed["answer"] == "Repaired"
        assert self.mock_llm.generate.called
        
    def test_fallback_json_creation(self):
        """Test fallback JSON creation when all repairs fail"""
        self.mock_llm.generate.side_effect = Exception("LLM failure")
        
        content = "Some random text that's not JSON"
        success, parsed, error = self.handler.ensure_valid_json(content, {})
        
        assert success is True
        assert "random text" in parsed["answer"]
        assert parsed["confidence"] == 0.3  # Fallback confidence
        
    def test_validate_response_format(self):
        """Test response format validation"""
        # Valid response
        valid = {"answer": "Test", "sources": [], "confidence": 0.9}
        assert self.handler.validate_response_format(valid) is True
        
        # Missing answer
        invalid = {"sources": [], "confidence": 0.9}
        assert self.handler.validate_response_format(invalid) is False
        
        # Wrong type for answer
        invalid = {"answer": 123, "sources": [], "confidence": 0.9}
        assert self.handler.validate_response_format(invalid) is False
        
        # Invalid confidence
        invalid = {"answer": "Test", "sources": [], "confidence": 1.5}
        assert self.handler.validate_response_format(invalid) is False


class TestFormattingPrompt:
    """Test the simplified formatting prompt"""
    
    def test_generate_formatting_prompt(self):
        """Test basic formatting prompt generation"""
        raw_answer = "This is a test answer"
        prompt = generate_formatting_prompt(raw_answer)
        
        assert "formatting assistant" in prompt
        assert raw_answer in prompt
        assert len(prompt.split('\n')) < 20  # Should be concise
        
    def test_special_case_handling(self):
        """Test special case for 'Answer not found'"""
        special_answer = "Answer not found. Please contact a manager or fellow dispatcher."
        result = generate_formatting_prompt(special_answer)
        
        assert result == special_answer  # Should return unchanged
        
    def test_formatting_quality_check(self):
        """Test quality checking without regex"""
        raw = "This is a long answer with multiple important words and concepts"
        
        # Good formatting - preserves content
        good_formatted = "This is a long answer with multiple **important** words and concepts"
        assert check_formatting_quality(good_formatted, raw) is True
        
        # Bad formatting - too short
        bad_formatted = "Short answer"
        assert check_formatting_quality(bad_formatted, raw) is False
        
        # Bad formatting - missing key words
        bad_formatted = "This is different text entirely"
        assert check_formatting_quality(bad_formatted, raw) is False


class TestPromptGenerator:
    """Test the regex-free prompt generator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = PromptGenerator()
        
    def test_context_truncation(self):
        """Test smart context truncation without regex"""
        # Create a long context
        long_context = "This is sentence one. " * 1000  # Very long
        
        prompt = self.generator.generate_extraction_prompt("Question?", long_context)
        
        assert "[Note: Context has been truncated" in prompt
        assert len(prompt) < len(long_context)
        
    def test_sentence_splitting_without_regex(self):
        """Test that sentence splitting works without regex"""
        context = "First sentence. Second sentence! Third sentence? Fourth."
        truncated = self.generator._truncate_context_smartly(context, 50)
        
        # Should preserve sentence boundaries
        assert truncated.endswith('.') or truncated.endswith('!') or truncated.endswith('?')
        
    def test_context_stats_without_regex(self):
        """Test context statistics gathering without regex"""
        context = "This is a test. It has multiple sentences! Really?"
        stats = self.generator.get_context_stats(context)
        
        assert stats["word_count"] == 9
        assert stats["sentence_count"] == 3
        assert stats["character_count"] == len(context)
        
    def test_enhanced_extraction_prompt(self):
        """Test enhanced prompts with retry logic"""
        query = "Test question"
        context = "Test context"
        
        # Base prompt
        prompt0 = self.generator.generate_enhanced_extraction_prompt(query, context, 0)
        assert "IMPORTANT: You MUST respond" not in prompt0
        
        # First retry
        prompt1 = self.generator.generate_enhanced_extraction_prompt(query, context, 1)
        assert "IMPORTANT: You MUST respond with valid JSON" in prompt1
        
        # Second retry
        prompt2 = self.generator.generate_enhanced_extraction_prompt(query, context, 2)
        assert "CRITICAL: You MUST respond with ONLY valid JSON" in prompt2


class TestIntegration:
    """Integration tests for the complete LLM-first pipeline"""
    
    def test_full_pipeline_without_regex(self):
        """Test that the full pipeline works without any regex"""
        # This test verifies no regex imports are used
        import llm.pipeline.validation
        import llm.pipeline.llm_json_handler
        import llm.prompts.formatting_prompt
        import llm.prompt_generator
        
        # Check that 're' module is not imported in any of our modules
        assert not hasattr(llm.pipeline.validation, 're')
        assert not hasattr(llm.pipeline.llm_json_handler, 're')
        assert not hasattr(llm.prompts.formatting_prompt, 're')
        assert not hasattr(llm.prompt_generator, 're')
        
    def test_json_handling_robustness(self):
        """Test various JSON edge cases"""
        test_cases = [
            # Valid JSON
            ('{"answer": "Test"}', True),
            # Missing quotes
            ('{answer: "Test"}', False),  # Should need LLM repair
            # Trailing comma
            ('{"answer": "Test",}', False),  # Should need LLM repair
            # Single quotes
            ("{'answer': 'Test'}", False),  # Should need LLM repair
            # Plain text
            ("Just plain text", False),  # Should use fallback
        ]
        
        for content, should_parse_directly in test_cases:
            is_valid, parsed, _ = validate_json_output(content)
            assert is_valid is True  # All should eventually succeed
            assert "answer" in parsed  # All should have answer field


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
