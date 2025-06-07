"""SPLADE Engine - Sparse Lexical and Dense Retrieval.

Experimental implementation of SPLADE (Sparse Lexical AnD Expansion) for improved
retrieval, especially for queries requiring information spread across documents.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from core.config import retrieval_config  # type: ignore[import]
from core.document_processor import DocumentProcessor  # type: ignore[import]
from utils.error_handling.enhanced_decorators import with_timing, with_error_tracking  # type: ignore[import]

logger = logging.getLogger(__name__)


class SpladeEngine:
    """SPLADE sparse+dense hybrid retrieval engine for experimental use."""

    def __init__(self, document_processor: DocumentProcessor) -> None:
        """Initialize SPLADE engine with document processor.
        
        Args:
            document_processor: Document processor instance for chunk access
        """
        self.document_processor = document_processor
        
        # SPLADE model configuration
        self.model_name = "naver/splade-cocondenser-ensembledistil"
        self.sparse_weight = 0.5  # Balance between sparse and dense
        self.expansion_k = 100  # Number of expansion terms
        self.max_sparse_length = 256  # Limit sparse vector size
        
        # Configuration-aware cache for document expansions
        self.doc_expansions_cache: Dict[str, Dict[str, float]] = {}
        self.current_config_hash = ""  # Track configuration changes
        
        # Initialize device and model with proper GPU setup
        self._setup_device_and_model()
        
        logger.info(f"SPLADE engine initialized with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")

    def _setup_device_and_model(self) -> None:
        """Setup device and initialize model with proper GPU handling for multiprocessing."""
        try:
            # Force GPU detection in multiprocessing environment
            if torch.cuda.is_available():
                # Verify GPU access in current process
                torch.cuda.synchronize()
                self.device = torch.device("cuda")
                logger.info(f"SPLADE using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.warning("SPLADE falling back to CPU - GPU not available")
            
            # Initialize model with device
            self._initialize_splade_model()
            
            # Update configuration hash after initialization
            self._update_config_hash()
            
        except Exception as e:
            logger.error(f"SPLADE device setup failed: {e}")
            # Fallback to CPU
            self.device = torch.device("cpu")
            self._initialize_splade_model()
            self._update_config_hash()

    def _update_config_hash(self) -> None:
        """Update configuration hash to track parameter changes."""
        import hashlib
        config_str = f"{self.sparse_weight}_{self.expansion_k}_{self.max_sparse_length}_{self.model_name}"
        self.current_config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        logger.debug(f"SPLADE config hash updated: {self.current_config_hash}")

    def _get_cache_key(self, doc_idx: int) -> str:
        """Generate configuration-aware cache key for document expansion."""
        return f"doc_{doc_idx}_{self.current_config_hash}"

    def _initialize_splade_model(self) -> None:
        """Initialize SPLADE model and tokenizer."""
        try:
            logger.info(f"Loading SPLADE model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("SPLADE model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SPLADE model: {e}")
            raise RuntimeError(f"SPLADE initialization failed: {e}")

    @with_timing(threshold=0.1)
    @with_error_tracking()
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> Tuple[str, float, int, List[float]]:
        """Search using SPLADE sparse+dense representations with optimized caching.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            Tuple of (formatted context, retrieval time, chunk count, similarity scores)
        """
        start_time = time.time()
        
        if not self.document_processor.texts:
            logger.warning("No documents indexed for SPLADE search")
            return "", 0.0, 0, []
        
        # Use provided threshold or fall back to config
        threshold = similarity_threshold or retrieval_config.similarity_threshold
        
        try:
            # CRITICAL PERFORMANCE FIX: Pre-build all document expansions on first search
            self._ensure_document_expansions_cached()
            
            # Generate query expansion (fast)
            query_expansion = self._generate_sparse_representation(query)
            logger.info(f"Generated query expansion with {len(query_expansion)} terms")
            
            # OPTIMIZED: Score using configuration-aware cached document expansions
            scores = []
            for idx in range(len(self.document_processor.texts)):
                cache_key = self._get_cache_key(idx)
                if cache_key not in self.doc_expansions_cache:
                    logger.warning(f"Missing cached expansion for document {idx}")
                    continue
                    
                doc_expansion = self.doc_expansions_cache[cache_key]
                
                # Calculate sparse score only (skip expensive dense scoring for speed)
                sparse_score = self._calculate_sparse_score(query_expansion, doc_expansion)
                
                # Use sparse score directly for much better performance
                scores.append((idx, sparse_score))
            
            # Sort by score and apply threshold
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by threshold and limit to top_k
            filtered_results = [
                (idx, score) for idx, score in scores[:top_k]
                if score >= threshold
            ]
            
            if not filtered_results:
                logger.info(f"No results above threshold {threshold}")
                return "", time.time() - start_time, 0, []
            
            # Format results
            contexts = []
            similarity_scores = []
            
            for idx, score in filtered_results:
                # CRITICAL FIX: Ensure we extract text content properly
                chunk = self.document_processor.texts[idx]
                if isinstance(chunk, dict):
                    # Extract text from dictionary chunk
                    text_content = chunk.get('text', str(chunk))
                elif isinstance(chunk, str):
                    text_content = chunk
                else:
                    # Fallback for other types
                    text_content = str(chunk) if chunk is not None else ""
                
                contexts.append(text_content)
                similarity_scores.append(float(score))
            
            # Join contexts (now guaranteed to be strings)
            formatted_context = "\n\n".join(contexts)
            
            # Trim if too long
            max_length = retrieval_config.max_context_length
            if len(formatted_context) > max_length:
                formatted_context = formatted_context[:max_length] + "..."
            
            retrieval_time = time.time() - start_time
            
            logger.info(
                f"SPLADE search completed: {len(filtered_results)} chunks, "
                f"max score: {similarity_scores[0]:.4f}, time: {retrieval_time:.2f}s"
            )
            
            return formatted_context, retrieval_time, len(filtered_results), similarity_scores
            
        except Exception as e:
            logger.error(f"SPLADE search failed: {e}")
            # Fallback to empty results
            return "", time.time() - start_time, 0, []

    def _ensure_document_expansions_cached(self) -> None:
        """Ensure all document expansions are pre-computed and cached for fast search."""
        # Count how many documents we have cached for current configuration
        current_config_cached = sum(1 for key in self.doc_expansions_cache.keys()
                                    if key.endswith(f"_{self.current_config_hash}"))
        
        if current_config_cached >= len(self.document_processor.texts):
            logger.debug(f"SPLADE cache hit: {current_config_cached} documents already cached for config {self.current_config_hash}")
            return  # Already cached for current configuration
        
        logger.info(f"Pre-computing SPLADE expansions for {len(self.document_processor.texts)} documents (config: {self.current_config_hash})...")
        start_time = time.time()
        
        # Only compute missing documents for current configuration
        computed = 0
        for idx, chunk in enumerate(self.document_processor.texts):
            cache_key = self._get_cache_key(idx)
            if cache_key not in self.doc_expansions_cache:
                # Process chunk text
                if isinstance(chunk, dict):
                    text = chunk.get('text', str(chunk))
                elif isinstance(chunk, str):
                    text = chunk
                else:
                    text = str(chunk) if chunk is not None else ""
                
                # Generate and cache expansion
                self.doc_expansions_cache[cache_key] = self._generate_sparse_representation(text)
                computed += 1
        
        cache_time = time.time() - start_time
        if computed > 0:
            logger.info(f"✅ Pre-computed {computed} new SPLADE expansions in {cache_time:.2f}s (total cache: {len(self.doc_expansions_cache)} entries)")
        else:
            logger.debug(f"✅ All SPLADE expansions already cached for config {self.current_config_hash}")

    def _generate_sparse_representation(self, text: str) -> Dict[str, float]:
        """Generate sparse representation with term expansions.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of term weights
        """
        # CRITICAL TYPE VALIDATION - Fix the type compatibility error
        if not isinstance(text, str):
            logger.error(f"SPLADE received invalid text type: {type(text)} - {text}")
            text = str(text) if text is not None else ""
        
        if not text or not text.strip():
            logger.warning("SPLADE received empty text, returning empty representation")
            return {}
        
        text = text.strip()
        
        with torch.no_grad():
            # Tokenize with validated string input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get model outputs
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply log-softmax and max pooling
            log_probs = torch.log_softmax(logits, dim=-1)
            max_log_probs, _ = torch.max(log_probs, dim=1)
            
            # Get top-k terms
            values, indices = torch.topk(max_log_probs[0], k=self.expansion_k)
            
        # Convert to dictionary
        sparse_rep = {}
        for value, idx in zip(values.cpu().numpy(), indices.cpu().numpy()):
            # CRITICAL FIX: Properly convert numpy scalar to Python int
            try:
                idx_int = int(idx.item()) if hasattr(idx, 'item') else int(idx)
                token = self.tokenizer.decode([idx_int])
                if token.strip() and not token.startswith("##"):
                    sparse_rep[token] = float(np.exp(value))
            except Exception as e:
                logger.debug(f"Failed to decode token at index {idx}: {e}")
                continue
        
        # CRITICAL FIX: Apply max_sparse_length limit - keep highest weighted terms
        if self.max_sparse_length and len(sparse_rep) > self.max_sparse_length:
            # Sort terms by weight (descending) and keep top max_sparse_length terms
            sorted_terms = sorted(sparse_rep.items(), key=lambda x: x[1], reverse=True)
            original_length = len(sparse_rep)
            sparse_rep = dict(sorted_terms[:self.max_sparse_length])
            logger.debug(f"Applied max_sparse_length limit: {original_length} → {len(sparse_rep)} terms (limit: {self.max_sparse_length})")
        
        return sparse_rep

    def _get_document_expansion(self, idx: int, text: str) -> Dict[str, float]:
        """Get or generate document expansion with caching.
        
        Args:
            idx: Document index
            text: Document text
            
        Returns:
            Sparse representation of document
        """
        # CRITICAL: Handle case where text might be a dict or other object
        if isinstance(text, dict):
            # Extract text from dictionary if it's a chunk object
            text = text.get('text', str(text))
        elif not isinstance(text, str):
            logger.warning(f"Document at index {idx} is not string: {type(text)}")
            text = str(text) if text is not None else ""
        
        cache_key = f"doc_{idx}"
        
        if cache_key not in self.doc_expansions_cache:
            self.doc_expansions_cache[cache_key] = self._generate_sparse_representation(text)
        
        return self.doc_expansions_cache[cache_key]

    def _calculate_sparse_score(
        self, 
        query_expansion: Dict[str, float], 
        doc_expansion: Dict[str, float]
    ) -> float:
        """Calculate sparse similarity score between expansions.
        
        Args:
            query_expansion: Query sparse representation
            doc_expansion: Document sparse representation
            
        Returns:
            Similarity score
        """
        score = 0.0
        
        # Intersection of terms
        common_terms = set(query_expansion.keys()) & set(doc_expansion.keys())
        
        for term in common_terms:
            score += query_expansion[term] * doc_expansion[term]
        
        # Normalize by query magnitude
        query_norm = np.sqrt(sum(v**2 for v in query_expansion.values()))
        if query_norm > 0:
            score /= query_norm
        
        return score

    def _calculate_dense_score(self, query: str, document: str) -> float:
        """Calculate dense similarity score.
        
        For now, this uses the existing embedding service through document processor.
        In a full implementation, we'd use SPLADE's dense component.
        
        Args:
            query: Query text
            document: Document text
            
        Returns:
            Similarity score
        """
        try:
            # Use existing vector search as dense component
            if hasattr(self.document_processor, 'index') and self.document_processor.index:
                # Get query embedding
                query_embedding = self.document_processor.embed_query(query)
                
                # Find document in index
                doc_idx = self.document_processor.texts.index(document)
                if doc_idx < len(self.document_processor.texts):
                    # Get document embedding from index
                    doc_embedding = self.document_processor.index.reconstruct(doc_idx)
                    
                    # Cosine similarity
                    score = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    return float(score)
        except Exception as e:
            logger.debug(f"Dense scoring fallback: {e}")
        
        # Fallback to simple overlap
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms & doc_terms) / len(query_terms)
        return overlap

    def clear_cache(self) -> None:
        """Clear document expansion cache."""
        self.doc_expansions_cache.clear()
        logger.info("SPLADE document expansion cache cleared")

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get SPLADE engine statistics.
        
        Returns:
            Dictionary of engine statistics
        """
        return {
            "model": self.model_name,
            "device": str(self.device),
            "sparse_weight": self.sparse_weight,
            "expansion_k": self.expansion_k,
            "cache_size": len(self.doc_expansions_cache),
            "max_sparse_length": self.max_sparse_length
        }

    def update_config(
        self,
        sparse_weight: Optional[float] = None,
        expansion_k: Optional[int] = None,
        max_sparse_length: Optional[int] = None
    ) -> None:
        """Update SPLADE configuration parameters.
        
        Args:
            sparse_weight: New sparse/dense balance (0-1)
            expansion_k: New number of expansion terms
            max_sparse_length: New max sparse vector length
        """
        if sparse_weight is not None:
            self.sparse_weight = max(0.0, min(1.0, sparse_weight))
            logger.info(f"Updated sparse_weight to {self.sparse_weight}")
        
        if expansion_k is not None:
            self.expansion_k = max(10, min(500, expansion_k))
            logger.info(f"Updated expansion_k to {self.expansion_k}")
        
        if max_sparse_length is not None:
            self.max_sparse_length = max(50, min(1000, max_sparse_length))
            logger.info(f"Updated max_sparse_length to {self.max_sparse_length}")
        
        # Update configuration hash to reflect changes
        self._update_config_hash()
        
        # Clear cache after config change - parameter changes require recomputing expansions
        # Configuration-aware caching will still allow efficient switching between configurations
        self.clear_cache()
        logger.info(f"SPLADE configuration updated, cache cleared for new hash: {self.current_config_hash}")

    def clear_gpu_memory(self) -> Dict[str, float]:
        """
        Clear GPU memory and return freed amounts
        
        Returns:
            Dictionary with freed memory amounts in MB
        """
        freed_memory = {"allocated": 0.0, "cached": 0.0}
        
        if self.device.type == 'cuda' and torch.cuda.is_available():
            try:
                # Record memory before clearing
                allocated_before = torch.cuda.memory_allocated(0) / (1024 * 1024)
                cached_before = torch.cuda.memory_reserved(0) / (1024 * 1024)
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Record memory after clearing
                allocated_after = torch.cuda.memory_allocated(0) / (1024 * 1024)
                cached_after = torch.cuda.memory_reserved(0) / (1024 * 1024)
                
                freed_memory = {
                    "allocated": allocated_before - allocated_after,
                    "cached": cached_before - cached_after
                }
                
                logger.info(f"SPLADE GPU memory cleared - Freed: {freed_memory['allocated']:.1f}MB allocated, {freed_memory['cached']:.1f}MB cached")
                
            except Exception as e:
                logger.error(f"Error clearing SPLADE GPU memory: {str(e)}")
                
        return freed_memory

    def unload_model(self) -> bool:
        """
        Unload the SPLADE model from memory to free GPU/CPU resources
        
        Returns:
            True if model was successfully unloaded
        """
        try:
            # Move models to CPU first if on GPU
            if hasattr(self, 'model') and self.model is not None:
                if self.device.type == 'cuda':
                    self.model.to('cpu')
                del self.model
                self.model = None
            
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear expansion cache
            self.clear_cache()
            
            # Clear GPU memory
            self.clear_gpu_memory()
            
            logger.info("Successfully unloaded SPLADE model")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading SPLADE model: {str(e)}")
            return False

    def reload_model(self) -> bool:
        """
        Reload the SPLADE model after it was unloaded
        
        Returns:
            True if model was successfully reloaded
        """
        try:
            if hasattr(self, 'model') and self.model is not None:
                logger.warning("SPLADE model is already loaded")
                return True
                
            # Reinitialize the model
            self._initialize_splade_model()
            logger.info("Successfully reloaded SPLADE model")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading SPLADE model: {str(e)}")
            return False

    def is_model_loaded(self) -> bool:
        """Check if the SPLADE model is currently loaded"""
        return (hasattr(self, 'model') and self.model is not None and 
                hasattr(self, 'tokenizer') and self.tokenizer is not None)

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage in MB
        
        Returns:
            Dictionary with memory usage stats
        """
        memory_stats = {"allocated": 0.0, "cached": 0.0, "total": 0.0}
        
        if self.device.type == 'cuda' and torch.cuda.is_available():
            try:
                memory_stats = {
                    "allocated": torch.cuda.memory_allocated(0) / (1024 * 1024),
                    "cached": torch.cuda.memory_reserved(0) / (1024 * 1024),
                    "total": torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                }
            except Exception as e:
                logger.warning(f"Error getting SPLADE memory usage: {str(e)}")
                
        return memory_stats
