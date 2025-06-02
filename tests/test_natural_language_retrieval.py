"""
Natural Language Retrieval Completeness Testing
Tests how well the system handles diverse, unpredictable user questions without query engineering.
Focus: Clean execution, human-reviewable results, progressive complexity.
"""

import time
import json
from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.engines.reranking_engine import RerankingEngine
from core.document_processor import DocumentProcessor
from llm.pipeline.answer_pipeline import AnswerPipeline
from llm.services.llm_service import LLMService
from llm.prompt_generator import PromptGenerator


class NaturalLanguageRetrievalTester:
    def __init__(self):
        """Initialize with clean, minimal output setup"""
        self.doc_processor = DocumentProcessor()
        self.reranking_engine = RerankingEngine(self.doc_processor)
        
        # Initialize LLM components
        self.llm_service = LLMService()
        self.prompt_generator = PromptGenerator()
        self.answer_pipeline = AnswerPipeline(self.llm_service, self.prompt_generator)
        
        # Question categories for progressive testing
        self.test_questions = {
            "simple_baseline": [
                "How much do we charge?",
                "What is our hourly rate?", 
                "What payment methods do we accept?",
                "Are we licensed and insured?",
                "What is our main phone number?"
            ],
            "medium_complexity": [
                "How do I reschedule a customer?",
                "Can we replace a laptop screen?",
                "How do I create a customer concern?",
                "What devices do we work on?",
                "How do I handle a same-day cancellation?"
            ],
            "complex_workflows": [
                "How do I handle a Field Engineer calling out sick?",
                "What's the complete process for a same-day cancellation?",
                "How do I handle a client who wants a refund?",
                "What's the complete text messaging workflow?",
                "How do I optimize routes for Field Engineers?"
            ]
        }
        
    def test_single_question(self, question: str, category: str) -> Dict[str, Any]:
        """Test a single question with clean execution and detailed results"""
        print(f"\n🔍 Testing: '{question}'")
        
        start_time = time.time()
        
        try:
            # Retrieval phase
            context, search_time, num_chunks, scores = self.reranking_engine.search(
                question, top_k=15
            )
            
            # Answer generation phase  
            answer_response = self.answer_pipeline.generate_answer(question, context)
            answer = answer_response.get('answer', 'No answer generated')
            
            total_time = time.time() - start_time
            
            # Parse chunks for human review
            chunks = self._parse_chunks(context)
            
            result = {
                'question': question,
                'category': category,
                'retrieval_time': search_time,
                'total_time': total_time,
                'chunks_retrieved': num_chunks,
                'similarity_scores': scores[:5] if scores else [],
                'retrieved_chunks': chunks,
                'generated_answer': answer,
                'timestamp': datetime.now().isoformat(),
                'success': num_chunks > 0 and len(answer.strip()) > 0
            }
            
            # Clean status display
            status = "✅ Complete" if result['success'] else "❌ Failed"
            print(f"   {status} | {num_chunks} chunks | {total_time:.1f}s")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {
                'question': question,
                'category': category,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
    
    def _parse_chunks(self, context: str) -> List[Dict[str, str]]:
        """Parse formatted context into chunks for human review"""
        chunks = []
        if not context:
            return chunks
            
        # Split by chunk markers
        sections = context.split('\n\n')
        
        for i, section in enumerate(sections):
            if section.strip() and section.startswith('['):
                lines = section.split('\n')
                if len(lines) > 1:
                    source_line = lines[0].strip()
                    content = '\n'.join(lines[1:]).strip()
                    
                    chunks.append({
                        'chunk_id': i + 1,
                        'source': source_line,
                        'content': content[:300] + ('...' if len(content) > 300 else ''),
                        'full_content': content
                    })
        
        return chunks
    
    def run_category_tests(self, category: str, show_details: bool = True) -> List[Dict[str, Any]]:
        """Run all tests in a category with clean output"""
        questions = self.test_questions.get(category, [])
        if not questions:
            print(f"❌ Unknown category: {category}")
            return []
        
        print(f"\n📋 TESTING CATEGORY: {category.upper()}")
        print(f"Questions: {len(questions)}")
        print("=" * 60)
        
        results = []
        successful = 0
        
        for question in questions:
            result = self.test_single_question(question, category)
            results.append(result)
            
            if result.get('success', False):
                successful += 1
                
            # Show human-reviewable details if requested
            if show_details and result.get('success', False):
                self._display_human_review(result)
        
        # Category summary
        success_rate = (successful / len(questions)) * 100 if questions else 0
        print(f"\n📊 CATEGORY SUMMARY: {category}")
        print(f"Success Rate: {successful}/{len(questions)} ({success_rate:.1f}%)")
        
        return results
    
    def _display_human_review(self, result: Dict[str, Any]):
        """Display results in human-reviewable format"""
        print(f"\n📝 HUMAN REVIEW - {result['question']}")
        print("-" * 50)
        
        # Show retrieved chunks
        chunks = result.get('retrieved_chunks', [])
        print(f"📄 Retrieved {len(chunks)} chunks:")
        for chunk in chunks[:3]:  # Show top 3 chunks
            print(f"   [{chunk['chunk_id']}] {chunk['content']}")
        
        # Show generated answer
        answer = result.get('generated_answer', '')
        print(f"\n💬 Generated Answer:")
        print(f"   {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print()
    
    def run_comprehensive_test(self, save_results: bool = True) -> Dict[str, Any]:
        """Run comprehensive test across all categories"""
        print("🚀 NATURAL LANGUAGE RETRIEVAL COMPLETENESS TEST")
        print("=" * 60)
        print("Testing system's ability to handle diverse, unpredictable questions")
        print("Focus: No query engineering, natural user language only")
        
        all_results = {}
        overall_stats = {
            'total_questions': 0,
            'successful_questions': 0,
            'categories_tested': 0,
            'start_time': datetime.now().isoformat()
        }
        
        # Test each category progressively
        for category in ['simple_baseline', 'medium_complexity', 'complex_workflows']:
            category_results = self.run_category_tests(category, show_details=False)
            all_results[category] = category_results
            
            # Update stats
            successful_in_category = sum(1 for r in category_results if r.get('success', False))
            overall_stats['total_questions'] += len(category_results)
            overall_stats['successful_questions'] += successful_in_category
            overall_stats['categories_tested'] += 1
        
        # Overall summary
        overall_success_rate = (overall_stats['successful_questions'] / overall_stats['total_questions'] * 100) if overall_stats['total_questions'] > 0 else 0
        
        print(f"\n🎯 OVERALL RESULTS")
        print("=" * 60)
        print(f"Total Questions Tested: {overall_stats['total_questions']}")
        print(f"Successful Retrievals: {overall_stats['successful_questions']}")
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        
        # Save results for human review
        if save_results:
            timestamp = int(time.time())
            filename = f"natural_language_retrieval_test_{timestamp}.json"
            
            final_results = {
                'test_metadata': {
                    'test_type': 'natural_language_retrieval_completeness',
                    'timestamp': datetime.now().isoformat(),
                    'total_questions': overall_stats['total_questions'],
                    'success_rate': overall_success_rate
                },
                'category_results': all_results,
                'overall_stats': overall_stats
            }
            
            with open(filename, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            print(f"\n💾 Detailed results saved: {filename}")
            print("👀 Human review recommended for answer quality assessment")
        
        return all_results
    
    def show_sample_question_answers(self, category: str = "simple_baseline", limit: int = 3):
        """Show sample questions with full answers for human evaluation"""
        questions = self.test_questions.get(category, [])[:limit]
        
        print(f"\n🔍 SAMPLE ANSWERS - {category.upper()}")
        print("=" * 60)
        print("👀 Human review required to assess answer quality")
        
        for question in questions:
            result = self.test_single_question(question, category)
            
            if result.get('success', False):
                print(f"\n❓ Question: {question}")
                print(f"💬 Answer: {result.get('generated_answer', 'No answer')}")
                print(f"📊 Retrieved {result.get('chunks_retrieved', 0)} chunks")
                print("-" * 40)


def main():
    """Main execution - clean, minimal output"""
    tester = NaturalLanguageRetrievalTester()
    
    print("Natural Language Retrieval Completeness Testing")
    print("Testing system without query engineering or hardcoded hints")
    print("=" * 60)
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Show some sample answers for human review
    print(f"\n🔍 SAMPLE ANSWERS FOR HUMAN REVIEW")
    tester.show_sample_question_answers("simple_baseline", limit=2)


if __name__ == "__main__":
    main()
