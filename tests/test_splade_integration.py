"""Test SPLADE integration in WorkApp2.

Tests the experimental SPLADE sparse+dense hybrid retrieval system.
"""

import pytest
import sys
import logging
from unittest.mock import MagicMock, patch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, '.')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSpladeIntegration:
    """Test SPLADE engine integration."""

    @pytest.fixture
    def mock_doc_processor(self):
        """Create a mock document processor."""
        doc_processor = MagicMock()
        doc_processor.texts = [
            "The Tampa Metro area has phone number 813-400-2865",
            "St Petersburg / Tampa Metro Phone Number: 727-350-1090",
            "Spring Hill / Tampa Metro Phone Number: 352-794-1085",
            "Same day cancellation (SDC) fee is $191.50 for new clients"
        ]
        doc_processor.index = MagicMock()
        doc_processor._get_embedding = MagicMock(return_value=np.random.rand(384))
        return doc_processor

    def test_splade_engine_initialization(self, mock_doc_processor):
        """Test SPLADE engine can be initialized."""
        try:
            from retrieval.engines.splade_engine import SpladeEngine
            
            # Test initialization
            engine = SpladeEngine(mock_doc_processor)
            
            assert engine is not None
            assert engine.sparse_weight == 0.5
            assert engine.expansion_k == 100
            assert engine.max_sparse_length == 256
            
            logger.info("✅ SPLADE engine initialized successfully")
            
        except ImportError as e:
            pytest.skip(f"SPLADE dependencies not installed: {e}")
        except Exception as e:
            pytest.fail(f"SPLADE initialization failed: {e}")

    def test_splade_search_functionality(self, mock_doc_processor):
        """Test SPLADE search returns results."""
        try:
            from retrieval.engines.splade_engine import SpladeEngine
            
            # Initialize engine
            engine = SpladeEngine(mock_doc_processor)
            
            # Test search
            query = "Tampa phone number"
            context, retrieval_time, chunk_count, scores = engine.search(query, top_k=3)
            
            assert context is not None
            assert len(context) > 0
            assert chunk_count > 0
            assert len(scores) > 0
            assert retrieval_time > 0
            
            logger.info(f"✅ SPLADE search completed: {chunk_count} chunks, max score: {max(scores):.4f}")
            
        except ImportError as e:
            pytest.skip(f"SPLADE dependencies not installed: {e}")
        except Exception as e:
            pytest.fail(f"SPLADE search failed: {e}")

    def test_splade_term_expansion(self, mock_doc_processor):
        """Test SPLADE term expansion functionality."""
        try:
            from retrieval.engines.splade_engine import SpladeEngine
            
            # Initialize engine
            engine = SpladeEngine(mock_doc_processor)
            
            # Test term expansion
            text = "SDC fee cancellation"
            expansion = engine._generate_sparse_representation(text)
            
            assert isinstance(expansion, dict)
            assert len(expansion) > 0
            assert all(isinstance(term, str) for term in expansion.keys())
            assert all(isinstance(weight, float) for weight in expansion.values())
            
            logger.info(f"✅ SPLADE term expansion generated {len(expansion)} terms")
            
        except ImportError as e:
            pytest.skip(f"SPLADE dependencies not installed: {e}")
        except Exception as e:
            pytest.fail(f"SPLADE term expansion failed: {e}")

    def test_unified_retrieval_system_splade_routing(self, mock_doc_processor):
        """Test UnifiedRetrievalSystem routes to SPLADE when enabled."""
        try:
            from retrieval.retrieval_system import UnifiedRetrievalSystem
            
            # Initialize retrieval system
            retrieval_system = UnifiedRetrievalSystem(mock_doc_processor)
            
            # Check SPLADE engine availability
            if retrieval_system.splade_engine is None:
                pytest.skip("SPLADE engine not available in retrieval system")
            
            # Enable SPLADE mode
            retrieval_system.use_splade = True
            
            # Test retrieval routes to SPLADE
            query = "What is the Tampa phone number?"
            context, retrieval_time, chunk_count, scores = retrieval_system.retrieve(query)
            
            assert context is not None
            assert retrieval_time > 0
            
            logger.info("✅ Retrieval system successfully routes to SPLADE when enabled")
            
        except ImportError as e:
            pytest.skip(f"Required dependencies not installed: {e}")
        except Exception as e:
            pytest.fail(f"SPLADE routing test failed: {e}")

    def test_splade_config_update(self, mock_doc_processor):
        """Test SPLADE configuration can be updated."""
        try:
            from retrieval.engines.splade_engine import SpladeEngine
            
            # Initialize engine
            engine = SpladeEngine(mock_doc_processor)
            
            # Test config update
            engine.update_config(
                sparse_weight=0.7,
                expansion_k=200,
                max_sparse_length=512
            )
            
            assert engine.sparse_weight == 0.7
            assert engine.expansion_k == 200
            assert engine.max_sparse_length == 512
            
            logger.info("✅ SPLADE configuration updated successfully")
            
        except ImportError as e:
            pytest.skip(f"SPLADE dependencies not installed: {e}")
        except Exception as e:
            pytest.fail(f"SPLADE config update failed: {e}")

    def test_splade_cache_functionality(self, mock_doc_processor):
        """Test SPLADE document expansion caching."""
        try:
            from retrieval.engines.splade_engine import SpladeEngine
            
            # Initialize engine
            engine = SpladeEngine(mock_doc_processor)
            
            # Perform search to populate cache
            query = "Tampa"
            engine.search(query, top_k=2)
            
            # Check cache
            initial_cache_size = len(engine.doc_expansions_cache)
            assert initial_cache_size > 0
            
            # Clear cache
            engine.clear_cache()
            assert len(engine.doc_expansions_cache) == 0
            
            logger.info("✅ SPLADE cache functionality working correctly")
            
        except ImportError as e:
            pytest.skip(f"SPLADE dependencies not installed: {e}")
        except Exception as e:
            pytest.fail(f"SPLADE cache test failed: {e}")

    def test_splade_sparse_scoring(self, mock_doc_processor):
        """Test SPLADE sparse scoring calculation."""
        try:
            from retrieval.engines.splade_engine import SpladeEngine
            
            # Initialize engine
            engine = SpladeEngine(mock_doc_processor)
            
            # Create test expansions
            query_expansion = {
                "tampa": 0.8,
                "phone": 0.9,
                "number": 0.7,
                "metro": 0.5
            }
            
            doc_expansion = {
                "tampa": 0.7,
                "phone": 0.6,
                "contact": 0.4,
                "area": 0.3
            }
            
            # Calculate score
            score = engine._calculate_sparse_score(query_expansion, doc_expansion)
            
            assert isinstance(score, float)
            assert score > 0
            
            logger.info(f"✅ SPLADE sparse scoring calculated: {score:.4f}")
            
        except ImportError as e:
            pytest.skip(f"SPLADE dependencies not installed: {e}")
        except Exception as e:
            pytest.fail(f"SPLADE scoring test failed: {e}")


if __name__ == "__main__":
    # Run the tests
    test = TestSpladeIntegration()
    mock_processor = test.mock_doc_processor()
    
    print("\n🧪 Testing SPLADE Integration...")
    
    # Run each test
    test.test_splade_engine_initialization(mock_processor)
    test.test_splade_search_functionality(mock_processor)
    test.test_splade_term_expansion(mock_processor)
    test.test_unified_retrieval_system_splade_routing(mock_processor)
    test.test_splade_config_update(mock_processor)
    test.test_splade_cache_functionality(mock_processor)
    test.test_splade_sparse_scoring(mock_processor)
    
    print("\n✅ All SPLADE integration tests completed!")
