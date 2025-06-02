"""
Natural Language Retrieval Testing - Simple Version
Tests retrieval completeness for diverse questions without query engineering.
Focus: Clean execution, human-reviewable chunk retrieval, no LLM dependencies.
"""

import time
import json
from typing import List, Dict, Any
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.engines.reranking_engine import RerankingEngine
from core.document_processor import DocumentProcessor


class SimpleRetrievalTester:
    def __init__(self):
        """Initialize with clean, minimal output setup"""
        self.doc_processor = DocumentProcessor()
        self.reranking_engine = RerankingEngine(self.doc_processor)
        
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
        """Test retrieval for a single question with detailed results"""
        print(f"\n🔍 Testing: '{question}'")
        
        start_time = time.time()
        
        try:
            # Retrieval phase only
            context, search_time, num_chunks, scores = self.reranking_engine.search(
                question, top_k=15
            )
            
            total_time = time.time() - start_time
            
            # Parse chunks for human review
            chunks = self._parse_chunks(context)
            
            # Assess retrieval quality
            quality_assessment = self._assess_retrieval_quality(question, chunks)
            
            result = {
                'question': question,
                'category': category,
                'retrieval_time': search_time,
                'total_time': total_time,
                'chunks_retrieved': num_chunks,
                'similarity_scores': scores[:5] if scores else [],
                'retrieved_chunks': chunks,
                'quality_assessment': quality_assessment,
                'timestamp': datetime.now().isoformat(),
                'success': num_chunks > 0
            }
            
            # Clean status display
            status = "✅ Retrieved" if result['success'] else "❌ Failed"
            quality = quality_assessment.get('overall_quality', 'UNKNOWN')
            print(f"   {status} | {num_chunks} chunks | {quality} quality | {total_time:.1f}s")
            
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
    
    def _assess_retrieval_quality(self, question: str, chunks: List[Dict[str, str]]) -> Dict[str, Any]:
        """Assess quality of retrieved chunks for the question"""
        if not chunks:
            return {
                'overall_quality': 'POOR',
                'issues': ['No chunks retrieved'],
                'relevance_indicators': []
            }
        
        # Look for question-specific relevance indicators
        relevance_indicators = []
        issues = []
        
        question_lower = question.lower()
        combined_content = ' '.join([chunk['full_content'].lower() for chunk in chunks])
        
        # Question-specific relevance checks
        if 'charge' in question_lower or 'hourly rate' in question_lower:
            if '$125' in combined_content or 'hourly rate' in combined_content:
                relevance_indicators.append('Found pricing information')
            else:
                issues.append('Missing pricing details')
                
        elif 'payment' in question_lower:
            if 'credit' in combined_content or 'card' in combined_content:
                relevance_indicators.append('Found payment method info')
            else:
                issues.append('Missing payment method details')
                
        elif 'reschedule' in question_lower:
            if 'reschedule' in combined_content or 'appointment' in combined_content:
                relevance_indicators.append('Found rescheduling procedures')
            else:
                issues.append('Missing rescheduling procedures')
                
        elif 'laptop screen' in question_lower or 'replace' in question_lower:
            if 'screen' in combined_content or 'replacement' in combined_content:
                relevance_indicators.append('Found hardware replacement info')
            else:
                issues.append('Missing hardware policy details')
                
        elif 'field engineer' in question_lower and 'sick' in question_lower:
            if 'field engineer' in combined_content or 'callout' in combined_content:
                relevance_indicators.append('Found FE callout procedures')
            else:
                issues.append('Missing FE callout procedures')
        
        # Overall quality assessment
        if len(relevance_indicators) >= 2 and len(issues) == 0:
            overall_quality = 'EXCELLENT'
        elif len(relevance_indicators) >= 1 and len(issues) <= 1:
            overall_quality = 'GOOD'
        elif len(relevance_indicators) >= 1:
            overall_quality = 'PARTIAL'
        else:
            overall_quality = 'POOR'
        
        return {
            'overall_quality': overall_quality,
            'relevance_indicators': relevance_indicators,
            'issues': issues,
            'chunk_count': len(chunks)
        }
    
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
        quality_counts = {'EXCELLENT': 0, 'GOOD': 0, 'PARTIAL': 0, 'POOR': 0}
        
        for question in questions:
            result = self.test_single_question(question, category)
            results.append(result)
            
            if result.get('success', False):
                successful += 1
                quality = result.get('quality_assessment', {}).get('overall_quality', 'UNKNOWN')
                if quality in quality_counts:
                    quality_counts[quality] += 1
                
            # Show human-reviewable details if requested
            if show_details and result.get('success', False):
                self._display_human_review(result)
        
        # Category summary
        success_rate = (successful / len(questions)) * 100 if questions else 0
        print(f"\n📊 CATEGORY SUMMARY: {category}")
        print(f"Success Rate: {successful}/{len(questions)} ({success_rate:.1f}%)")
        print(f"Quality Distribution: EXCELLENT:{quality_counts['EXCELLENT']} | GOOD:{quality_counts['GOOD']} | PARTIAL:{quality_counts['PARTIAL']} | POOR:{quality_counts['POOR']}")
        
        return results
    
    def _display_human_review(self, result: Dict[str, Any]):
        """Display results in human-reviewable format"""
        print(f"\n📝 HUMAN REVIEW - {result['question']}")
        print("-" * 50)
        
        # Show quality assessment
        quality = result.get('quality_assessment', {})
        print(f"🎯 Quality: {quality.get('overall_quality', 'UNKNOWN')}")
        
        indicators = quality.get('relevance_indicators', [])
        if indicators:
            print(f"✅ Found: {', '.join(indicators)}")
        
        issues = quality.get('issues', [])
        if issues:
            print(f"❌ Missing: {', '.join(issues)}")
        
        # Show top retrieved chunks
        chunks = result.get('retrieved_chunks', [])
        print(f"\n📄 Top 3 chunks (of {len(chunks)}):")
        for chunk in chunks[:3]:
            print(f"   [{chunk['chunk_id']}] {chunk['content']}")
        print()
    
    def run_comprehensive_test(self, save_results: bool = True) -> Dict[str, Any]:
        """Run comprehensive test across all categories"""
        print("🚀 NATURAL LANGUAGE RETRIEVAL TESTING (Simple Version)")
        print("=" * 60)
        print("Testing retrieval completeness for diverse, unpredictable questions")
        print("Focus: No query engineering, natural user language only")
        
        all_results = {}
        overall_stats = {
            'total_questions': 0,
            'successful_retrievals': 0,
            'quality_distribution': {'EXCELLENT': 0, 'GOOD': 0, 'PARTIAL': 0, 'POOR': 0},
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
            overall_stats['successful_retrievals'] += successful_in_category
            overall_stats['categories_tested'] += 1
            
            # Update quality distribution
            for result in category_results:
                if result.get('success', False):
                    quality = result.get('quality_assessment', {}).get('overall_quality', 'POOR')
                    if quality in overall_stats['quality_distribution']:
                        overall_stats['quality_distribution'][quality] += 1
        
        # Overall summary
        retrieval_success_rate = (overall_stats['successful_retrievals'] / overall_stats['total_questions'] * 100) if overall_stats['total_questions'] > 0 else 0
        
        print(f"\n🎯 OVERALL RESULTS")
        print("=" * 60)
        print(f"Total Questions Tested: {overall_stats['total_questions']}")
        print(f"Successful Retrievals: {overall_stats['successful_retrievals']}")
        print(f"Retrieval Success Rate: {retrieval_success_rate:.1f}%")
        
        quality_dist = overall_stats['quality_distribution']
        print(f"Quality Distribution:")
        print(f"  EXCELLENT: {quality_dist['EXCELLENT']} ({quality_dist['EXCELLENT']/overall_stats['successful_retrievals']*100:.1f}%)")
        print(f"  GOOD: {quality_dist['GOOD']} ({quality_dist['GOOD']/overall_stats['successful_retrievals']*100:.1f}%)")
        print(f"  PARTIAL: {quality_dist['PARTIAL']} ({quality_dist['PARTIAL']/overall_stats['successful_retrievals']*100:.1f}%)")
        print(f"  POOR: {quality_dist['POOR']} ({quality_dist['POOR']/overall_stats['successful_retrievals']*100:.1f}%)")
        
        # Save results for human review
        if save_results:
            timestamp = int(time.time())
            filename = f"natural_language_retrieval_simple_{timestamp}.json"
            
            final_results = {
                'test_metadata': {
                    'test_type': 'natural_language_retrieval_simple',
                    'timestamp': datetime.now().isoformat(),
                    'total_questions': overall_stats['total_questions'],
                    'retrieval_success_rate': retrieval_success_rate
                },
                'category_results': all_results,
                'overall_stats': overall_stats
            }
            
            with open(filename, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            print(f"\n💾 Detailed results saved: {filename}")
            print("👀 Human review recommended for retrieval quality assessment")
        
        return all_results
    
    def show_sample_retrievals(self, category: str = "simple_baseline", limit: int = 3):
        """Show sample questions with full retrievals for human evaluation"""
        questions = self.test_questions.get(category, [])[:limit]
        
        print(f"\n🔍 SAMPLE RETRIEVALS - {category.upper()}")
        print("=" * 60)
        print("👀 Human review of retrieved chunks")
        
        for question in questions:
            result = self.test_single_question(question, category)
            
            if result.get('success', False):
                print(f"\n❓ Question: {question}")
                quality = result.get('quality_assessment', {})
                print(f"🎯 Quality: {quality.get('overall_quality', 'UNKNOWN')}")
                print(f"📊 Retrieved {result.get('chunks_retrieved', 0)} chunks")
                
                # Show chunk previews
                chunks = result.get('retrieved_chunks', [])
                print("📄 Chunk Previews:")
                for chunk in chunks[:3]:
                    print(f"   [{chunk['chunk_id']}] {chunk['content']}")
                print("-" * 40)


def main():
    """Main execution - clean, minimal output"""
    tester = SimpleRetrievalTester()
    
    print("Natural Language Retrieval Testing - Simple Version")
    print("Testing retrieval completeness without query engineering")
    print("=" * 60)
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Show some sample retrievals for human review
    print(f"\n🔍 SAMPLE RETRIEVALS FOR HUMAN REVIEW")
    tester.show_sample_retrievals("simple_baseline", limit=2)


if __name__ == "__main__":
    main()
