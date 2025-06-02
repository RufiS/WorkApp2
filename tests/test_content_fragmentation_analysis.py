"""
Phase 2: Content Fragmentation Analysis
Examines how content is chunked and why workflows get fragmented.
Focus: Understanding chunk boundaries and content distribution.
"""

import json
import re
from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ContentFragmentationAnalyzer:
    def __init__(self):
        """Initialize content fragmentation analyzer"""
        self.chunk_file_path = "current_index/chunks.txt"
        self.chunks = []
        self.analysis_results = {}
    
    def load_chunks(self) -> bool:
        """Load and parse chunks from the index file"""
        try:
            with open(self.chunk_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse chunks - they're formatted as "Chunk X:" followed by content
            chunk_sections = re.split(r'\nChunk \d+:\n', content)
            
            # First section before "Chunk 0:" might contain metadata, skip it
            chunk_start = 1 if chunk_sections[0].strip() and not chunk_sections[0].startswith('Chunk') else 0
            
            for i, section in enumerate(chunk_sections[chunk_start:], 0):
                if section.strip():
                    self.chunks.append({
                        'chunk_id': i,
                        'content': section.strip(),
                        'length': len(section.strip()),
                        'lines': len(section.strip().split('\n'))
                    })
            
            print(f"✅ Loaded {len(self.chunks)} chunks from index")
            return True
            
        except FileNotFoundError:
            print(f"❌ Chunk file not found: {self.chunk_file_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading chunks: {e}")
            return False
    
    def analyze_phone_number_distribution(self) -> Dict[str, Any]:
        """Analyze how phone number content is distributed across chunks"""
        print(f"\n🔍 ANALYZING PHONE NUMBER CONTENT DISTRIBUTION")
        print("=" * 60)
        
        phone_pattern = r'\d{3}-\d{3}-\d{4}'
        main_company_pattern = r'480-999-3046'
        
        phone_chunks = []
        main_company_chunks = []
        metro_directory_chunks = []
        
        for chunk in self.chunks:
            content_lower = chunk['content'].lower()
            
            # Find chunks with phone numbers
            phone_numbers = re.findall(phone_pattern, chunk['content'])
            if phone_numbers:
                phone_chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'phone_count': len(phone_numbers),
                    'phone_numbers': phone_numbers,
                    'content_preview': chunk['content'][:200] + '...',
                    'contains_main_company': main_company_pattern in chunk['content'],
                    'contains_metro_terms': any(term in content_lower for term in ['metro', 'areas serviced', 'phone number:']),
                    'contains_directory_structure': 'phone number:' in content_lower or 'metro:' in content_lower
                })
            
            # Find main company number specifically
            if main_company_pattern in chunk['content']:
                main_company_chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'content_preview': chunk['content'][:300] + '...',
                    'context_terms': [term for term in ['main', 'company', 'number', 'primary'] if term in content_lower]
                })
            
            # Find metro directory structure
            if any(term in content_lower for term in ['metro:', 'phone number:', 'areas serviced']):
                metro_directory_chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'content_preview': chunk['content'][:200] + '...',
                    'structure_indicators': [term for term in ['metro:', 'phone number:', 'areas serviced'] if term in content_lower]
                })
        
        analysis = {
            'total_phone_chunks': len(phone_chunks),
            'total_phone_numbers_found': sum(chunk['phone_count'] for chunk in phone_chunks),
            'main_company_chunks': len(main_company_chunks),
            'metro_directory_chunks': len(metro_directory_chunks),
            'phone_chunks_details': phone_chunks,
            'main_company_details': main_company_chunks,
            'metro_directory_details': metro_directory_chunks
        }
        
        print(f"📊 Phone Number Distribution:")
        print(f"  Total chunks with phone numbers: {analysis['total_phone_chunks']}")
        print(f"  Total phone numbers found: {analysis['total_phone_numbers_found']}")
        print(f"  Chunks with main company number: {analysis['main_company_chunks']}")
        print(f"  Chunks with metro directory structure: {analysis['metro_directory_chunks']}")
        
        # Show why "metro phone numbers" works but "main phone number" doesn't
        if metro_directory_chunks:
            print(f"\n✅ 'Metro phone numbers' succeeds because:")
            for chunk in metro_directory_chunks[:2]:
                print(f"  Chunk {chunk['chunk_id']}: {chunk['structure_indicators']}")
        
        if not any('main' in chunk.get('context_terms', []) for chunk in main_company_chunks):
            print(f"\n❌ 'Main phone number' fails because:")
            print(f"  Main company number (480-999-3046) exists but lacks 'main' context")
        
        return analysis
    
    def analyze_text_messaging_distribution(self) -> Dict[str, Any]:
        """Analyze how text messaging workflow content is distributed"""
        print(f"\n🔍 ANALYZING TEXT MESSAGING CONTENT DISTRIBUTION")
        print("=" * 60)
        
        text_terms = ['text', 'sms', 'message', 'messaging']
        workflow_terms = ['workflow', 'process', 'procedure', 'step', 'complete']
        system_terms = ['freshdesk', 'ticket', 'response', 'contact attempt']
        
        text_chunks = []
        workflow_chunks = []
        complete_workflow_chunks = []
        
        for chunk in self.chunks:
            content_lower = chunk['content'].lower()
            
            # Find chunks with text/SMS content
            text_matches = [term for term in text_terms if term in content_lower]
            workflow_matches = [term for term in workflow_terms if term in content_lower]
            system_matches = [term for term in system_terms if term in content_lower]
            
            if text_matches:
                text_chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'text_terms': text_matches,
                    'workflow_terms': workflow_matches,
                    'system_terms': system_matches,
                    'content_preview': chunk['content'][:200] + '...',
                    'has_procedural_content': any(term in content_lower for term in ['step', 'how to', 'process']),
                    'has_complete_workflow': len(workflow_matches) > 0 and any(term in content_lower for term in ['complete', 'full', 'entire'])
                })
            
            # Find chunks with workflow terminology
            if workflow_matches and text_matches:
                workflow_chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'content_preview': chunk['content'][:200] + '...',
                    'workflow_indicators': workflow_matches,
                    'text_indicators': text_matches
                })
            
            # Find chunks that might contain complete workflows
            if (len(text_matches) > 0 and len(workflow_matches) > 0 and 
                any(term in content_lower for term in ['complete', 'full', 'entire', 'step by step'])):
                complete_workflow_chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'content_preview': chunk['content'][:300] + '...',
                    'completeness_indicators': [term for term in ['complete', 'full', 'entire', 'step by step'] if term in content_lower]
                })
        
        analysis = {
            'total_text_chunks': len(text_chunks),
            'workflow_chunks': len(workflow_chunks),
            'complete_workflow_chunks': len(complete_workflow_chunks),
            'text_chunks_details': text_chunks,
            'workflow_chunks_details': workflow_chunks,
            'complete_workflow_details': complete_workflow_chunks,
            'fragmentation_analysis': self._analyze_workflow_fragmentation(text_chunks)
        }
        
        print(f"📊 Text Messaging Content Distribution:")
        print(f"  Total chunks with text/SMS terms: {analysis['total_text_chunks']}")
        print(f"  Chunks with both text+workflow terms: {analysis['workflow_chunks']}")
        print(f"  Chunks with complete workflow indicators: {analysis['complete_workflow_chunks']}")
        
        # Explain why queries fail
        if analysis['complete_workflow_chunks'] == 0:
            print(f"\n❌ 'Complete text messaging workflow' fails because:")
            print(f"  No chunks contain both text terms AND completeness indicators")
        
        if analysis['workflow_chunks'] < 3:
            print(f"\n⚠️ Text messaging workflows are fragmented:")
            print(f"  Only {analysis['workflow_chunks']} chunks contain both text and workflow terms")
        
        return analysis
    
    def _analyze_workflow_fragmentation(self, text_chunks: List[Dict]) -> Dict[str, Any]:
        """Analyze how workflow content is fragmented across chunks"""
        procedural_chunks = [chunk for chunk in text_chunks if chunk['has_procedural_content']]
        system_chunks = [chunk for chunk in text_chunks if chunk['system_terms']]
        
        return {
            'total_text_chunks': len(text_chunks),
            'procedural_chunks': len(procedural_chunks),
            'system_focused_chunks': len(system_chunks),
            'fragmentation_ratio': len(procedural_chunks) / len(text_chunks) if text_chunks else 0,
            'system_focus_ratio': len(system_chunks) / len(text_chunks) if text_chunks else 0
        }
    
    def analyze_customer_concern_distribution(self) -> Dict[str, Any]:
        """Analyze customer concern content distribution (for comparison)"""
        print(f"\n🔍 ANALYZING CUSTOMER CONCERN CONTENT DISTRIBUTION")
        print("=" * 60)
        
        concern_terms = ['concern', 'customer concern', 'complaint']
        process_terms = ['create', 'develop', 'submit', 'process', 'step', 'how to']
        system_terms = ['freshdesk', 'ticket', 'template', 'helpdesk']
        
        concern_chunks = []
        complete_process_chunks = []
        
        for chunk in self.chunks:
            content_lower = chunk['content'].lower()
            
            concern_matches = [term for term in concern_terms if term in content_lower]
            process_matches = [term for term in process_terms if term in content_lower]
            system_matches = [term for term in system_terms if term in content_lower]
            
            if concern_matches:
                concern_chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'concern_terms': concern_matches,
                    'process_terms': process_matches,
                    'system_terms': system_matches,
                    'content_preview': chunk['content'][:200] + '...',
                    'has_complete_process': len(process_matches) >= 2 and len(system_matches) >= 1
                })
            
            # Find chunks with complete processes
            if (concern_matches and len(process_matches) >= 2 and 
                any(term in content_lower for term in ['step', 'follow', 'how to'])):
                complete_process_chunks.append({
                    'chunk_id': chunk['chunk_id'],
                    'content_preview': chunk['content'][:300] + '...',
                    'process_indicators': process_matches
                })
        
        analysis = {
            'total_concern_chunks': len(concern_chunks),
            'complete_process_chunks': len(complete_process_chunks),
            'concern_chunks_details': concern_chunks,
            'complete_process_details': complete_process_chunks
        }
        
        print(f"📊 Customer Concern Content Distribution:")
        print(f"  Total chunks with concern terms: {analysis['total_concern_chunks']}")
        print(f"  Chunks with complete processes: {analysis['complete_process_chunks']}")
        
        # Explain why this category works better
        if analysis['complete_process_chunks'] > 0:
            print(f"\n✅ Customer concern queries succeed because:")
            print(f"  {analysis['complete_process_chunks']} chunks contain complete processes")
            print(f"  Content is organized around user tasks rather than system components")
        
        return analysis
    
    def analyze_chunk_structure_patterns(self) -> Dict[str, Any]:
        """Analyze overall chunk structure and organization patterns"""
        print(f"\n🔍 ANALYZING CHUNK STRUCTURE PATTERNS")
        print("=" * 60)
        
        # Analyze chunk sizes
        chunk_lengths = [chunk['length'] for chunk in self.chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        
        # Analyze content organization patterns
        directory_chunks = 0
        procedural_chunks = 0
        reference_chunks = 0
        
        for chunk in self.chunks:
            content_lower = chunk['content'].lower()
            
            # Directory-style content (lists, numbers, contact info)
            if (chunk['content'].count('\n') > 10 and 
                any(pattern in chunk['content'] for pattern in ['-', '•', '1.', '2.', 'Phone Number:', 'Metro:'])):
                directory_chunks += 1
            
            # Procedural content (how-to, steps, processes)
            elif any(term in content_lower for term in ['step', 'how to', 'process', 'procedure', 'follow']):
                procedural_chunks += 1
            
            # Reference content (definitions, policies, facts)
            else:
                reference_chunks += 1
        
        structure_analysis = {
            'total_chunks': len(self.chunks),
            'average_chunk_length': avg_length,
            'chunk_length_distribution': {
                'short_chunks': len([l for l in chunk_lengths if l < 500]),
                'medium_chunks': len([l for l in chunk_lengths if 500 <= l < 1500]),
                'long_chunks': len([l for l in chunk_lengths if l >= 1500])
            },
            'content_organization': {
                'directory_chunks': directory_chunks,
                'procedural_chunks': procedural_chunks,
                'reference_chunks': reference_chunks
            },
            'organization_ratios': {
                'directory_ratio': directory_chunks / len(self.chunks) if self.chunks else 0,
                'procedural_ratio': procedural_chunks / len(self.chunks) if self.chunks else 0,
                'reference_ratio': reference_chunks / len(self.chunks) if self.chunks else 0
            }
        }
        
        print(f"📊 Chunk Structure Analysis:")
        print(f"  Total chunks: {structure_analysis['total_chunks']}")
        print(f"  Average length: {avg_length:.0f} characters")
        print(f"  Directory-style chunks: {directory_chunks} ({directory_chunks/len(self.chunks)*100:.1f}%)")
        print(f"  Procedural chunks: {procedural_chunks} ({procedural_chunks/len(self.chunks)*100:.1f}%)")
        print(f"  Reference chunks: {reference_chunks} ({reference_chunks/len(self.chunks)*100:.1f}%)")
        
        return structure_analysis
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive content fragmentation analysis"""
        print("🚀 CONTENT FRAGMENTATION ANALYSIS - PHASE 2")
        print("=" * 60)
        print("Examining how content is chunked and why workflows get fragmented")
        
        if not self.load_chunks():
            return {'error': 'Failed to load chunks'}
        
        # Run all analyses
        phone_analysis = self.analyze_phone_number_distribution()
        text_analysis = self.analyze_text_messaging_distribution()
        concern_analysis = self.analyze_customer_concern_distribution()
        structure_analysis = self.analyze_chunk_structure_patterns()
        
        # Generate key insights
        insights = self._generate_fragmentation_insights(
            phone_analysis, text_analysis, concern_analysis, structure_analysis
        )
        
        final_results = {
            'analysis_metadata': {
                'analysis_type': 'content_fragmentation_phase_2',
                'timestamp': datetime.now().isoformat(),
                'total_chunks_analyzed': len(self.chunks)
            },
            'phone_number_analysis': phone_analysis,
            'text_messaging_analysis': text_analysis,
            'customer_concern_analysis': concern_analysis,
            'chunk_structure_analysis': structure_analysis,
            'key_insights': insights
        }
        
        # Save results
        timestamp = int(time.time())
        filename = f"content_fragmentation_analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Analysis saved: {filename}")
        print("👀 Ready for Phase 3: Systematic Fix Development")
        
        return final_results
    
    def _generate_fragmentation_insights(self, phone_analysis, text_analysis, concern_analysis, structure_analysis) -> Dict[str, Any]:
        """Generate key insights about content fragmentation"""
        insights = {
            'fragmentation_problems': [],
            'organization_patterns': [],
            'retrieval_gaps': [],
            'recommendations': []
        }
        
        # Phone number fragmentation issues
        if phone_analysis['main_company_chunks'] > 0 and phone_analysis['main_company_chunks'] < 2:
            insights['fragmentation_problems'].append(
                "Main company number exists but lacks contextual terms like 'main' or 'primary'"
            )
        
        if phone_analysis['metro_directory_chunks'] > phone_analysis['main_company_chunks']:
            insights['organization_patterns'].append(
                "Content organized around metro directories rather than company-wide information"
            )
        
        # Text messaging fragmentation issues
        if text_analysis['complete_workflow_chunks'] == 0:
            insights['fragmentation_problems'].append(
                "Text messaging workflows are fragmented - no single chunk contains complete process"
            )
        
        fragmentation_ratio = text_analysis['fragmentation_analysis']['fragmentation_ratio']
        if fragmentation_ratio < 0.5:
            insights['fragmentation_problems'].append(
                f"Only {fragmentation_ratio:.1%} of text chunks contain procedural content"
            )
        
        # Comparison with successful category
        if concern_analysis['complete_process_chunks'] > text_analysis['complete_workflow_chunks']:
            insights['organization_patterns'].append(
                "Customer concerns organized as complete processes while text messaging is fragmented"
            )
        
        # Structural insights
        if structure_analysis['organization_ratios']['directory_ratio'] > 0.3:
            insights['organization_patterns'].append(
                "High proportion of directory-style chunks may not match natural language queries"
            )
        
        # Generate recommendations
        if phone_analysis['main_company_chunks'] > 0:
            insights['recommendations'].append(
                "Add contextual terms ('main', 'primary', 'company') to main phone number chunks"
            )
        
        if text_analysis['complete_workflow_chunks'] == 0:
            insights['recommendations'].append(
                "Reorganize text messaging content to include complete workflows in single chunks"
            )
        
        if concern_analysis['complete_process_chunks'] > text_analysis['complete_workflow_chunks']:
            insights['recommendations'].append(
                "Apply customer concern organization pattern to text messaging workflows"
            )
        
        return insights


def main():
    """Execute Phase 2 of content fragmentation analysis"""
    analyzer = ContentFragmentationAnalyzer()
    
    print("PHASE 2: CONTENT FRAGMENTATION ANALYSIS")
    print("Examining chunk boundaries and content distribution")
    print("=" * 60)
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    # Display key insights
    if 'key_insights' in results:
        insights = results['key_insights']
        
        if insights.get('fragmentation_problems'):
            print(f"\n🔍 FRAGMENTATION PROBLEMS:")
            for problem in insights['fragmentation_problems']:
                print(f"  • {problem}")
        
        if insights.get('organization_patterns'):
            print(f"\n📋 ORGANIZATION PATTERNS:")
            for pattern in insights['organization_patterns']:
                print(f"  • {pattern}")
        
        if insights.get('recommendations'):
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in insights['recommendations']:
                print(f"  • {rec}")


if __name__ == "__main__":
    import time
    main()
