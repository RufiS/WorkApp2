"""
Comprehensive Text Message Retrieval Test
Tests multi-query approach to achieve complete workflow coverage
"""

import time
from typing import List, Dict, Any, Set
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.engines.reranking_engine import RerankingEngine
from core.document_processor import DocumentProcessor


class ComprehensiveTextRetrieval:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.reranking_engine = RerankingEngine(self.doc_processor)
        
    def multi_query_search(self, base_query: str, related_queries: List[str], max_chunks: int = 20) -> Dict[str, Any]:
        """
        Perform multi-query search to achieve comprehensive coverage
        
        Args:
            base_query: Primary query
            related_queries: Additional queries to fill gaps
            max_chunks: Maximum total chunks to return
            
        Returns:
            Combined results with deduplication
        """
        print(f"🔍 Multi-Query Comprehensive Search")
        print(f"Base query: '{base_query}'")
        print(f"Related queries: {len(related_queries)}")
        
        all_results = []
        seen_texts = set()
        query_contributions = {}
        
        # Process base query first
        try:
            context, search_time, num_chunks, scores = self.reranking_engine.search(base_query, top_k=10)
            results = self._parse_context_to_results(context, scores)
            
            for result in results:
                text_key = self._normalize_text(result['text'])
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    result['source_query'] = base_query
                    result['query_type'] = 'primary'
                    all_results.append(result)
            
            query_contributions[base_query] = len(results)
            print(f"   Primary query contributed: {len(results)} chunks")
            
        except Exception as e:
            print(f"   ❌ Primary query failed: {e}")
            query_contributions[base_query] = 0
        
        # Process related queries
        for query in related_queries:
            try:
                context, search_time, num_chunks, scores = self.reranking_engine.search(query, top_k=8)
                results = self._parse_context_to_results(context, scores)
                
                new_chunks = 0
                for result in results:
                    text_key = self._normalize_text(result['text'])
                    if text_key not in seen_texts:
                        seen_texts.add(text_key)
                        result['source_query'] = query
                        result['query_type'] = 'supplementary'
                        all_results.append(result)
                        new_chunks += 1
                
                query_contributions[query] = new_chunks
                print(f"   '{query}' contributed: {new_chunks} new chunks")
                
            except Exception as e:
                print(f"   ❌ Query '{query}' failed: {e}")
                query_contributions[query] = 0
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = all_results[:max_chunks]
        
        # Combine into context
        combined_context = self._format_combined_context(final_results)
        
        return {
            'context': combined_context,
            'total_chunks': len(final_results),
            'unique_chunks': len(seen_texts),
            'query_contributions': query_contributions,
            'coverage_analysis': self._analyze_coverage(combined_context)
        }
    
    def _parse_context_to_results(self, context: str, scores: List[float]) -> List[Dict[str, Any]]:
        """Parse formatted context back into result objects"""
        results = []
        chunks = context.split('\n\n')
        
        for i, chunk in enumerate(chunks):
            if chunk.strip() and chunk.startswith('['):
                # Extract text after source info
                lines = chunk.split('\n')
                if len(lines) > 1:
                    text = '\n'.join(lines[1:]).strip()
                    score = scores[i] if i < len(scores) else 0.0
                    
                    results.append({
                        'text': text,
                        'score': score,
                        'chunk_id': i
                    })
        
        return results
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for deduplication"""
        return ' '.join(text.lower().split())[:100]  # First 100 chars, normalized
    
    def _format_combined_context(self, results: List[Dict[str, Any]]) -> str:
        """Format results into combined context"""
        formatted_chunks = []
        
        for i, result in enumerate(results):
            source_query = result.get('source_query', 'unknown')
            query_type = result.get('query_type', 'unknown')
            score = result.get('score', 0.0)
            
            formatted_chunk = f"[{i+1}] From {source_query} ({query_type}, score: {score:.3f}):\n{result['text']}\n"
            formatted_chunks.append(formatted_chunk)
        
        return '\n'.join(formatted_chunks)
    
    def _analyze_coverage(self, context: str) -> Dict[str, Any]:
        """Analyze coverage of critical text messaging procedures"""
        context_lower = context.lower()
        
        # Define coverage indicators
        coverage_indicators = {
            'ringcentral_texting': ['ringcentral', 'text message', 'sms'],
            'sms_templates': ['general sms format', 'appointment confirmation', 'template'],
            'three_touch_workflow': ['3 touches', '30 min', 'vm/sms', 'pending'],
            'ticket_identification': ['freshdesk', 'ticket', 'computer repair', 'matthew karls'],
            'fe_notification': ['field engineer', 'kti channel', '@fieldengineer'],
            'client_lookup': ['phone number', 'client info', 'scheduled with']
        }
        
        coverage_results = {}
        for category, indicators in coverage_indicators.items():
            found_indicators = sum(1 for indicator in indicators if indicator in context_lower)
            coverage_results[category] = {
                'indicators_found': found_indicators,
                'total_indicators': len(indicators),
                'coverage_percentage': (found_indicators / len(indicators)) * 100
            }
        
        # Overall coverage
        total_found = sum(result['indicators_found'] for result in coverage_results.values())
        total_possible = sum(result['total_indicators'] for result in coverage_results.values())
        overall_coverage = (total_found / total_possible) * 100 if total_possible > 0 else 0
        
        return {
            'category_coverage': coverage_results,
            'overall_coverage': overall_coverage,
            'categories_with_good_coverage': len([c for c in coverage_results.values() if c['coverage_percentage'] >= 60])
        }


def test_comprehensive_retrieval():
    """Test comprehensive multi-query retrieval approach"""
    retriever = ComprehensiveTextRetrieval()
    
    # Define comprehensive query set
    base_query = "How do I send a text message"
    
    related_queries = [
        "text message response workflow",
        "3 touch contact protocol", 
        "appointment confirmation text template",
        "freshdesk text ticket handling",
        "field engineer text notification",
        "VM and SMS response procedures",
        "30 minute contact intervals",
        "text message ticket identification",
        "RingCentral text procedures",
        "client text response protocol"
    ]
    
    print("COMPREHENSIVE TEXT MESSAGE RETRIEVAL TEST")
    print("=" * 60)
    
    start_time = time.time()
    results = retriever.multi_query_search(base_query, related_queries, max_chunks=25)
    total_time = time.time() - start_time
    
    print(f"\n📊 RESULTS SUMMARY")
    print(f"Total retrieval time: {total_time:.2f}s")
    print(f"Total chunks retrieved: {results['total_chunks']}")
    print(f"Unique content pieces: {results['unique_chunks']}")
    
    print(f"\n📈 QUERY CONTRIBUTIONS")
    for query, contribution in results['query_contributions'].items():
        print(f"   {contribution:2d} chunks: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    
    print(f"\n🎯 COVERAGE ANALYSIS")
    coverage = results['coverage_analysis']
    print(f"Overall coverage: {coverage['overall_coverage']:.1f}%")
    print(f"Categories with good coverage (≥60%): {coverage['categories_with_good_coverage']}/6")
    
    for category, data in coverage['category_coverage'].items():
        status = "✅" if data['coverage_percentage'] >= 60 else "⚠️" if data['coverage_percentage'] >= 30 else "❌"
        print(f"   {status} {category}: {data['coverage_percentage']:.1f}% ({data['indicators_found']}/{data['total_indicators']})")
    
    # Success criteria
    success = (
        coverage['overall_coverage'] >= 70 and 
        coverage['categories_with_good_coverage'] >= 4 and
        results['total_chunks'] >= 15
    )
    
    print(f"\n{'✅ SUCCESS' if success else '❌ NEEDS IMPROVEMENT'}: Multi-query approach {'achieved' if success else 'did not achieve'} comprehensive coverage")
    
    return results, success


if __name__ == "__main__":
    results, success = test_comprehensive_retrieval()
