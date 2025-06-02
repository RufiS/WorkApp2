"""
Phase 1: Semantic Bridging Analysis
Tests why natural language queries fail to connect to existing content.
Focus: Phone number retrieval failure investigation.
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


class SemanticBridgingAnalyzer:
    def __init__(self):
        """Initialize semantic bridging analyzer"""
        self.doc_processor = DocumentProcessor()
        self.reranking_engine = RerankingEngine(self.doc_processor)
        
        # Phase 1.1: Query Variation Testing for Phone Numbers
        self.phone_query_variations = [
            # Natural language variations
            "What is our main phone number?",
            "What is the main company phone number?", 
            "How do I contact the company?",
            "What is the company contact number?",
            "What phone number should I call?",
            
            # Direct search attempts
            "480-999-3046",
            "main company number 480-999-3046",
            "phone directory",
            "contact information",
            
            # Technical terminology  
            "dispatch phone numbers",
            "metro phone numbers",
            "local phone number",
            "phone number directory"
        ]
        
        # Phase 1.2: Text Messaging Workflow Variations
        self.text_workflow_variations = [
            # Natural workflow questions
            "What's the complete text messaging workflow?",
            "How do I handle text messages?",
            "How do I process SMS messages?",
            "What's the text message procedure?",
            "How do I respond to customer texts?",
            
            # Process-focused queries
            "text message process step by step",
            "SMS handling procedure", 
            "text ticket workflow",
            "complete text messaging process",
            
            # Technical terminology
            "SMS response protocol",
            "text message ticket handling",
            "Freshdesk text ticket process",
            "text message contact attempts"
        ]
        
        # Phase 1.3: Customer Concern Variations  
        self.concern_query_variations = [
            # Natural language questions
            "How do I create a customer concern?",
            "How do I handle customer complaints?",
            "What's the process for customer issues?",
            "How do I submit a customer concern ticket?",
            "How do I escalate customer problems?",
            
            # Process-focused queries
            "customer concern ticket creation process",
            "complaint handling procedure",
            "customer concern workflow steps",
            "Freshdesk customer concern process",
            
            # Technical terminology
            "concern ticket development",
            "customer concern template",
            "helpdesk concern submission",
            "customer concern Freshdesk procedure"
        ]
    
    def test_query_variation_set(self, query_set: List[str], category: str) -> Dict[str, Any]:
        """Test a set of query variations and analyze results"""
        print(f"\n🔍 TESTING {category.upper()} VARIATIONS")
        print("=" * 60)
        
        results = []
        retrieval_success_count = 0
        
        for i, query in enumerate(query_set, 1):
            print(f"\n[{i}/{len(query_set)}] Testing: '{query}'")
            
            start_time = time.time()
            
            try:
                # Retrieval test
                context, search_time, num_chunks, scores = self.reranking_engine.search(
                    query, top_k=15
                )
                
                total_time = time.time() - start_time
                
                # Parse retrieved content
                chunks = self._parse_chunks(context)
                
                # Analyze content relevance
                relevance_analysis = self._analyze_content_relevance(query, chunks, category)
                
                result = {
                    'query': query,
                    'category': category,
                    'chunks_retrieved': num_chunks,
                    'similarity_scores': scores[:3] if scores else [],
                    'search_time': search_time,
                    'total_time': total_time,
                    'chunks_preview': [chunk['content'][:100] + '...' for chunk in chunks[:2]],
                    'relevance_analysis': relevance_analysis,
                    'success': num_chunks > 0,
                    'timestamp': datetime.now().isoformat()
                }
                
                if result['success']:
                    retrieval_success_count += 1
                
                # Status display
                status = f"✅ {num_chunks} chunks" if num_chunks > 0 else "❌ 0 chunks"
                relevance = relevance_analysis.get('relevance_level', 'UNKNOWN')
                print(f"   {status} | {relevance} relevance | {total_time:.2f}s")
                
                # Show top similarity score for analysis
                if scores:
                    print(f"   Top similarity: {scores[0]:.3f}")
                
                results.append(result)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    'query': query,
                    'category': category,
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Category summary
        success_rate = (retrieval_success_count / len(query_set)) * 100
        print(f"\n📊 {category} SUMMARY:")
        print(f"Success Rate: {retrieval_success_count}/{len(query_set)} ({success_rate:.1f}%)")
        
        return {
            'category': category,
            'total_queries': len(query_set),
            'successful_retrievals': retrieval_success_count,
            'success_rate': success_rate,
            'detailed_results': results
        }
    
    def _parse_chunks(self, context: str) -> List[Dict[str, str]]:
        """Parse context into chunk information"""
        chunks = []
        if not context:
            return chunks
            
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
                        'content': content,
                        'content_preview': content[:200] + ('...' if len(content) > 200 else '')
                    })
        
        return chunks
    
    def _analyze_content_relevance(self, query: str, chunks: List[Dict], category: str) -> Dict[str, Any]:
        """Analyze how relevant retrieved content is to the query"""
        if not chunks:
            return {
                'relevance_level': 'NO_CONTENT',
                'analysis': 'No chunks retrieved',
                'missing_elements': ['All content missing']
            }
        
        query_lower = query.lower()
        combined_content = ' '.join([chunk['content'].lower() for chunk in chunks])
        
        relevance_indicators = []
        missing_elements = []
        
        # Category-specific relevance analysis
        if category == 'phone_numbers':
            # Check for phone number content
            if 'phone' in combined_content or 'number' in combined_content:
                relevance_indicators.append('Contains phone/number terminology')
            if '480-999-3046' in combined_content:
                relevance_indicators.append('Contains main company number')
            if any(metro in combined_content for metro in ['metro', 'atlanta', 'phoenix', 'dallas']):
                relevance_indicators.append('Contains metro information')
            
            # Check for missing elements
            if 'main' in query_lower and 'main' not in combined_content:
                missing_elements.append('Missing "main" designation')
            if 'company' in query_lower and 'company' not in combined_content:
                missing_elements.append('Missing "company" context')
                
        elif category == 'text_messaging':
            # Check for text/SMS content
            if any(term in combined_content for term in ['text', 'sms', 'message']):
                relevance_indicators.append('Contains text/SMS terminology')
            if any(term in combined_content for term in ['workflow', 'process', 'procedure', 'step']):
                relevance_indicators.append('Contains process terminology')
            if 'freshdesk' in combined_content:
                relevance_indicators.append('Contains system references')
                
            # Check for missing elements
            if 'complete' in query_lower and 'complete' not in combined_content:
                missing_elements.append('Missing complete workflow coverage')
            if 'workflow' in query_lower and 'workflow' not in combined_content:
                missing_elements.append('Missing workflow structure')
                
        elif category == 'customer_concerns':
            # Check for concern/complaint content
            if any(term in combined_content for term in ['concern', 'complaint', 'customer']):
                relevance_indicators.append('Contains concern terminology')
            if any(term in combined_content for term in ['ticket', 'freshdesk', 'helpdesk']):
                relevance_indicators.append('Contains ticketing system references')
            if any(term in combined_content for term in ['create', 'develop', 'submit']):
                relevance_indicators.append('Contains action terminology')
                
            # Check for missing elements
            if 'create' in query_lower and 'create' not in combined_content:
                missing_elements.append('Missing creation process')
            if 'how' in query_lower and not any(term in combined_content for term in ['step', 'process', 'procedure']):
                missing_elements.append('Missing procedural guidance')
        
        # Determine relevance level
        if len(relevance_indicators) >= 3:
            relevance_level = 'HIGH'
        elif len(relevance_indicators) >= 2:
            relevance_level = 'MEDIUM'
        elif len(relevance_indicators) >= 1:
            relevance_level = 'LOW'
        else:
            relevance_level = 'POOR'
        
        return {
            'relevance_level': relevance_level,
            'relevance_indicators': relevance_indicators,
            'missing_elements': missing_elements,
            'analysis': f'{len(relevance_indicators)} relevance indicators found'
        }
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive semantic bridging analysis"""
        print("🚀 SEMANTIC BRIDGING ANALYSIS - PHASE 1")
        print("=" * 60)
        print("Investigating why natural language queries fail to connect to existing content")
        
        analysis_results = {}
        
        # Phase 1.1: Phone Number Query Variations
        phone_results = self.test_query_variation_set(
            self.phone_query_variations, 
            'phone_numbers'
        )
        analysis_results['phone_numbers'] = phone_results
        
        # Phase 1.2: Text Messaging Workflow Variations  
        text_results = self.test_query_variation_set(
            self.text_workflow_variations,
            'text_messaging'
        )
        analysis_results['text_messaging'] = text_results
        
        # Phase 1.3: Customer Concern Variations
        concern_results = self.test_query_variation_set(
            self.concern_query_variations,
            'customer_concerns'  
        )
        analysis_results['customer_concerns'] = concern_results
        
        # Overall analysis summary
        total_queries = sum(result['total_queries'] for result in analysis_results.values())
        total_successful = sum(result['successful_retrievals'] for result in analysis_results.values())
        overall_success_rate = (total_successful / total_queries * 100) if total_queries > 0 else 0
        
        print(f"\n🎯 OVERALL SEMANTIC BRIDGING ANALYSIS")
        print("=" * 60)
        print(f"Total Query Variations Tested: {total_queries}")
        print(f"Successful Retrievals: {total_successful}")
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        
        # Category breakdown
        for category, results in analysis_results.items():
            print(f"  {category.replace('_', ' ').title()}: {results['success_rate']:.1f}%")
        
        # Save detailed results
        timestamp = int(time.time())
        filename = f"semantic_bridging_analysis_{timestamp}.json"
        
        final_results = {
            'analysis_metadata': {
                'analysis_type': 'semantic_bridging_phase_1',
                'timestamp': datetime.now().isoformat(),
                'total_queries': total_queries,
                'overall_success_rate': overall_success_rate
            },
            'category_analyses': analysis_results,
            'key_findings': self._generate_key_findings(analysis_results)
        }
        
        with open(filename, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Detailed analysis saved: {filename}")
        print("👀 Ready for Phase 2: Content Fragmentation Analysis")
        
        return final_results
    
    def _generate_key_findings(self, analysis_results: Dict) -> Dict[str, Any]:
        """Generate key findings from the analysis"""
        findings = {
            'semantic_gaps_identified': [],
            'successful_query_patterns': [],
            'failure_patterns': [],
            'recommendations': []
        }
        
        # Analyze patterns across categories
        for category, results in analysis_results.items():
            category_name = category.replace('_', ' ').title()
            
            if results['success_rate'] < 30:
                findings['semantic_gaps_identified'].append(
                    f"{category_name}: Major semantic gap - {results['success_rate']:.1f}% success rate"
                )
            
            # Find successful vs failed patterns
            successful_queries = [r for r in results['detailed_results'] if r.get('success', False)]
            failed_queries = [r for r in results['detailed_results'] if not r.get('success', False)]
            
            if successful_queries:
                findings['successful_query_patterns'].append(
                    f"{category_name}: {len(successful_queries)} successful queries found"
                )
            
            if len(failed_queries) > len(successful_queries):
                findings['failure_patterns'].append(
                    f"{category_name}: {len(failed_queries)} failed queries indicate terminology mismatch"
                )
        
        # Generate recommendations
        if any('phone' in gap for gap in findings['semantic_gaps_identified']):
            findings['recommendations'].append(
                "Phone number queries need direct number indexing or improved semantic bridging"
            )
        
        if any('text' in gap for gap in findings['semantic_gaps_identified']):
            findings['recommendations'].append(
                "Text messaging workflows need better procedural content organization"
            )
        
        return findings


def main():
    """Execute Phase 1 of semantic bridging analysis"""
    analyzer = SemanticBridgingAnalyzer()
    
    print("PHASE 1: SEMANTIC BRIDGING ANALYSIS")
    print("Investigating natural language query failures")
    print("=" * 60)
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    # Display key findings
    key_findings = results.get('key_findings', {})
    
    if key_findings.get('semantic_gaps_identified'):
        print(f"\n🔍 KEY SEMANTIC GAPS:")
        for gap in key_findings['semantic_gaps_identified']:
            print(f"  • {gap}")
    
    if key_findings.get('recommendations'):
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in key_findings['recommendations']:
            print(f"  • {rec}")


if __name__ == "__main__":
    main()
