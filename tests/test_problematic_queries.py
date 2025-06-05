"""
Test problematic queries with new LLM-first implementation

This script tests real-world problematic queries that caused issues in production
"""
import json
import sys
from datetime import datetime
from llm.pipeline.validation import validate_json_output
from llm.pipeline.llm_json_handler import LLMJSONHandler
from llm.prompts.formatting_prompt import generate_formatting_prompt, check_formatting_quality

# Problematic queries from production logs
test_queries = [
    {
        "name": "FFLBoss formatting issue",
        "content": 'If a client mentions that they are part of FFL Boss in any form, they should submit a ticket either in the FFL Boss app or on the website(s) fflboss.com',
        "expected": "Should format cleanly without echoing instructions"
    },
    {
        "name": "Malformed JSON with trailing comma",
        "content": '{"answer": "Same day cancellation policy...",}',
        "expected": "Should repair trailing comma"
    },
    {
        "name": "Missing quotes in JSON",
        "content": '{answer: "Customer concern process", confidence: 0.9}',
        "expected": "Should add missing quotes"
    },
    {
        "name": "Plain text response",
        "content": 'Just a plain text answer without any JSON structure',
        "expected": "Should create valid JSON from plain text"
    },
    {
        "name": "Instruction echoing test",
        "content": '{"answer": "FORMATTING RULES: The actual answer is here"}',
        "expected": "Should not echo formatting instructions"
    },
    {
        "name": "Complex malformed JSON",
        "content": "{'answer': 'Using single quotes', sources: [\"mixed quotes\"], 'confidence': .8}",
        "expected": "Should fix mixed quotes and decimal format"
    },
    {
        "name": "Escaped quotes in answer",
        "content": '{"answer": "He said \\"hello\\" to me", "confidence": 0.9}',
        "expected": "Should handle escaped quotes properly"
    },
    {
        "name": "Very long plain text",
        "content": "This is a very long answer that goes on and on " * 50,
        "expected": "Should handle long text gracefully"
    },
    {
        "name": "Empty content",
        "content": "",
        "expected": "Should handle empty content appropriately"
    },
    {
        "name": "Whitespace only",
        "content": "   \n\t   ",
        "expected": "Should handle whitespace-only content"
    }
]

def test_json_validation():
    """Test JSON validation for all problematic queries"""
    print("=" * 80)
    print("🧪 TESTING JSON VALIDATION")
    print("=" * 80)
    
    results = []
    
    for test in test_queries:
        print(f"\n📝 Testing: {test['name']}")
        print(f"Expected: {test['expected']}")
        print(f"Input: {test['content'][:100]}{'...' if len(test['content']) > 100 else ''}")
        
        # Test JSON validation
        is_valid, parsed, error = validate_json_output(test['content'])
        
        result = {
            "name": test['name'],
            "valid": is_valid,
            "error": error,
            "has_answer": parsed and 'answer' in parsed if parsed else False
        }
        
        print(f"✓ Valid: {is_valid}")
        if error:
            print(f"⚠️  Error: {error}")
        if parsed:
            print(f"✓ Parsed successfully")
            print(f"  - Has answer: {'answer' in parsed}")
            print(f"  - Answer preview: {str(parsed.get('answer', 'N/A'))[:100]}...")
            if 'confidence' in parsed:
                print(f"  - Confidence: {parsed['confidence']}")
        
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results if r['valid'])
    print(f"✓ Successful validations: {successful}/{len(results)}")
    print(f"✓ All have answers: {all(r['has_answer'] for r in results)}")
    
    return results

def test_formatting():
    """Test formatting prompt generation"""
    print("\n" + "=" * 80)
    print("🎨 TESTING FORMATTING PROMPTS")
    print("=" * 80)
    
    test_answers = [
        "Simple answer without any special formatting",
        "Answer not found. Please contact a manager or fellow dispatcher.",
        "Complex answer with **bold** text and • bullet points",
        "Phone number: 813-555-1234",
        "FORMATTING RULES: This should not appear in output"
    ]
    
    for answer in test_answers:
        print(f"\n📝 Testing answer: {answer[:50]}...")
        prompt = generate_formatting_prompt(answer)
        print(f"✓ Prompt length: {len(prompt)} characters")
        print(f"✓ Contains 'FORMATTING RULES': {'FORMATTING RULES' in prompt}")
        print(f"✓ Is concise: {len(prompt) < 1000}")
        
        # Check quality
        if answer != "Answer not found. Please contact a manager or fellow dispatcher.":
            quality = check_formatting_quality(answer, answer)
            print(f"✓ Quality check passed: {quality}")

def test_llm_json_handler():
    """Test LLM JSON handler with mock LLM"""
    print("\n" + "=" * 80)
    print("🤖 TESTING LLM JSON HANDLER")
    print("=" * 80)
    
    # Create mock LLM service
    class MockLLM:
        def generate(self, prompt, temperature=0, max_tokens=2000):
            # Simulate different responses based on prompt content
            if "malformed JSON" in prompt:
                return '{"answer": "Fixed JSON response", "sources": [], "confidence": 0.8}'
            elif "extract" in prompt.lower():
                return '{"answer": "Extracted answer", "sources": [], "confidence": 0.7}'
            else:
                return '{"answer": "Generic response", "sources": [], "confidence": 0.5}'
    
    handler = LLMJSONHandler(MockLLM())
    
    test_cases = [
        ('{"answer": "Valid JSON"}', "Valid JSON"),
        ('{answer: "Invalid JSON"}', "Invalid JSON needing repair"),
        ('Plain text response', "Plain text needing extraction"),
        ('', "Empty content")
    ]
    
    for content, description in test_cases:
        print(f"\n📝 Testing: {description}")
        success, parsed, error = handler.ensure_valid_json(content, {})
        print(f"✓ Success: {success}")
        if parsed:
            print(f"✓ Answer: {parsed.get('answer', 'N/A')}")
            print(f"✓ Confidence: {parsed.get('confidence', 'N/A')}")

def main():
    """Run all tests"""
    print("\n🚀 LLM-FIRST IMPLEMENTATION - PROBLEMATIC QUERY TESTING")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Run tests
    validation_results = test_json_validation()
    test_formatting()
    test_llm_json_handler()
    
    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE")
    print("=" * 80)
    
    # Return success code
    all_valid = all(r['valid'] and r['has_answer'] for r in validation_results)
    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())
