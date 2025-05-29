# WorkApp2 Progress: Last completed Retrieval-3 (Retrieval System)
# Next pending LLM-1 (LLM Service)
# Enhanced retrieval module

import os
import time
import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.config_unified import retrieval_config, performance_config, resolve_path
from utils.error_handling.decorators import with_timing, with_advanced_retry
from utils.error_logging import query_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedRetrieval:
    """Enhanced retrieval system with hybrid search and reranking"""

    def __init__(self):
        """Initialize the enhanced retrieval system"""
        self.bm25_index: Optional[BM25Okapi] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.corpus: List[str] = []
        self.chunk_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.use_hybrid = retrieval_config.enhanced_mode
        self.use_mmr = performance_config.enable_reranking
        self.enable_keyword_fallback = performance_config.enable_keyword_fallback

    @with_timing(threshold=0.5)
    def build_bm25_index(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build a BM25 index from document chunks

        Args:
            chunks: List of document chunks
        """
        if chunks is None:
            logger.error("Cannot build BM25 index with None chunks")
            raise ValueError("chunks parameter cannot be None")
        if not isinstance(chunks, list):
            logger.error(f"chunks must be a list, got {type(chunks)}")
            raise TypeError(f"chunks must be a list, got {type(chunks)}")
        if not chunks:
            logger.warning("No chunks provided for BM25 indexing")
            return

        try:
            self.corpus.clear()
            self.chunk_ids.clear()
            invalid_chunks = 0

            for i, chunk in enumerate(chunks):
                if not isinstance(chunk, dict):
                    logger.warning(f"Chunk at index {i} is not a dict: {type(chunk)}")
                    invalid_chunks += 1
                    continue
                text = chunk.get("text")
                if not isinstance(text, str):
                    logger.warning(f"Chunk 'text' at index {i} invalid: {type(text)}")
                    invalid_chunks += 1
                    continue

                self.corpus.append(text)
                cid = chunk.get("id") or f"chunk-{i}"
                self.chunk_ids.append(cid)

            if invalid_chunks > 0:
                logger.warning(
                    f"Found {invalid_chunks} invalid chunks out of {len(chunks)}"
                )
                if invalid_chunks / len(chunks) > 0.5 and not self.corpus:
                    raise ValueError("No valid chunks for BM25 indexing")

            self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            empty_tokens = sum(1 for tokens in self.tokenized_corpus if not tokens)
            if empty_tokens:
                logger.warning(f"{empty_tokens} chunks tokenized to empty lists")

            self.bm25_index = BM25Okapi(self.tokenized_corpus)
            if not hasattr(self.bm25_index, "get_scores"):
                raise RuntimeError("BM25 index missing 'get_scores'")

            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_vectorizer.fit(self.corpus)
            if not hasattr(self.tfidf_vectorizer, "transform"):
                raise RuntimeError("TF-IDF vectorizer missing 'transform'")

            logger.info(f"Built BM25 index on {len(self.corpus)} documents")
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            raise

    @with_timing(threshold=0.5)
    def hybrid_search(
        self, query: str, vector_results: List[Dict[str, Any]], top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector search and BM25
        """
        start_time = time.time()
        preview = (query[:50] + "...") if len(query) > 50 else query
        logger.info(
            f"Hybrid search for '{preview}' with {len(vector_results)} vector hits, top_k={top_k}"
        )

        fallback_info = {
            "used_fallback": False,
            "fallback_reason": None,
            "original_method": "hybrid",
        }

        if not self.bm25_index or not self.corpus:
            logger.warning("BM25 index missing, vector-only fallback")
            safe_k = min(top_k, len(vector_results))
            results = vector_results[:safe_k]
            for r in results:
                r["fallback_info"] = {**fallback_info, "used_fallback": True,
                                      "fallback_reason": "bm25_missing"}
            latency = time.time() - start_time
            logger.info(f"Vector-only fallback returned {len(results)} in {latency:.4f}s")
            if performance_config.log_query_metrics:
                query_logger.log_query(
                    query=query,
                    latency=latency,
                    hit_count=len(results),
                    metadata={"search_type": "vector_only"}
                )
            return results

        try:
            tokenized = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized)
            max_b = float(max(bm25_scores)) if bm25_scores.any() else 1.0
            norm_bm25 = bm25_scores / max_b if max_b > 0 else bm25_scores

            bm25_map = dict(zip(self.chunk_ids, norm_bm25))
            vec_map = {
                r["id"]: r["score"]
                for r in (vector_results or [])
            }
            max_v = float(max(vec_map.values())) if vec_map else 1.0
            norm_vec = {k: v / max_v for k, v in vec_map.items()}

            α = retrieval_config.vector_weight
            β = 1.0 - α
            combined = []
            for res in vector_results:
                cid = res["id"]
                vs = norm_vec.get(cid, 0.0)
                bs = bm25_map.get(cid, 0.0)
                score = α * vs + β * bs
                entry = res.copy()
                entry.update({"score": score, "vector_score": vs, "bm25_score": bs,
                              "fallback_info": fallback_info})
                combined.append(entry)

            combined.sort(key=lambda x: x["score"], reverse=True)
            if self.use_mmr:
                combined = self.mmr_reranking(query, combined, top_k)
            else:
                combined = combined[: top_k]

            latency = time.time() - start_time
            logger.info(f"Hybrid search returned {len(combined)} in {latency:.4f}s")
            if performance_config.log_query_metrics:
                query_logger.log_query(
                    query=query,
                    latency=latency,
                    hit_count=len(combined),
                    metadata={
                        "search_type": "hybrid",
                        "vector_weight": retrieval_config.vector_weight,
                        "mmr": self.use_mmr
                    }
                )
            return combined

        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            safe_k = min(top_k, len(vector_results))
            fallback_info.update({"used_fallback": True, "fallback_reason": str(e)})
            results = vector_results[:safe_k]
            for r in results:
                r["fallback_info"] = fallback_info
            latency = time.time() - start_time
            logger.info(f"Fallback vector-only returned {len(results)} in {latency:.4f}s")
            if performance_config.log_query_metrics:
                query_logger.log_query(
                    query=query,
                    latency=latency,
                    hit_count=len(results),
                    metadata={"search_type": "vector_only", "error": True}
                )
            return results

    @with_timing(threshold=0.5)
    def mmr_reranking(
        self, query: str, results: List[Dict[str, Any]], top_k: int = 10,
        lambda_param: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance reranking to diversify results
        """
        start_time = time.time()
        preview = (query[:50] + "...") if len(query) > 50 else query
        logger.info(
            f"MMR rerank for '{preview}' with {len(results)} hits, top_k={top_k}"
        )
        if not results:
            logger.warning("No results to rerank")
            return []

        # Ensure TF-IDF vectorizer is initialized
        if self.tfidf_vectorizer is None:
            logger.error("TF-IDF vectorizer not initialized for MMR")
            raise RuntimeError("TF-IDF vectorizer not initialized")

        safe_k = min(top_k, len(results))
        if safe_k < top_k:
            logger.info(f"Adjusting top_k down to {safe_k}")

        try:
            texts = [r["text"] for r in results]
            vecs = np.array([r["score"] for r in results])

            tfidf_mat = self.tfidf_vectorizer.transform(texts).toarray()
            q_vec = self.tfidf_vectorizer.transform([query]).toarray()[0]

            q_norm = q_vec / np.linalg.norm(q_vec) if np.linalg.norm(q_vec) > 0 else q_vec
            mat_norm = tfidf_mat / np.linalg.norm(tfidf_mat, axis=1, keepdims=True)

            sim_q = mat_norm.dot(q_norm)

            selected = []
            pool = list(range(len(results)))

            # First pick
            first_idx = int(np.argmax(sim_q))
            selected.append(first_idx)
            pool.remove(first_idx)

            # MMR loop
            for _ in range(min(safe_k - 1, len(results) - 1)):
                best_score = -np.inf
                best_idx = -1
                for idx in pool:
                    rel = sim_q[idx]
                    if selected:
                        div = np.max(mat_norm[selected].dot(mat_norm[idx]))
                    else:
                        div = 0.0
                    mmr = lambda_param * rel - (1 - lambda_param) * div
                    if mmr > best_score:
                        best_score = mmr
                        best_idx = idx
                if best_idx >= 0:
                    selected.append(best_idx)
                    pool.remove(best_idx)

            reranked = [results[i] for i in selected]
            latency = time.time() - start_time
            logger.info(f"MMR selected {len(reranked)} in {latency:.4f}s")
            return reranked

        except Exception as e:
            logger.error(f"MMR error: {e}")
            return results[:safe_k]

    @with_timing(threshold=0.5)
    @with_advanced_retry(
        max_attempts=3, backoff_factor=1.5, exception_types=[IOError, OSError]
    )
    def keyword_fallback_search(
        self, query: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search as a fallback when vector search returns no results
        """
        start_time = time.time()
        preview = (query[:50] + "...") if len(query) > 50 else query
        logger.info(f"Keyword fallback for '{preview}', top_k={top_k}")

        # Validate inputs
        if not query or not isinstance(query, str):
            logger.error(f"Invalid query: {query}")
            raise ValueError(f"Invalid query: {query}")
        if not isinstance(top_k, int) or top_k <= 0:
            logger.error(f"Invalid top_k: {top_k}")
            raise ValueError(f"Invalid top_k: {top_k}")

        # Ensure BM25 index is built
        if not self.bm25_index or not self.corpus:
            logger.error("BM25 index not built for keyword fallback")
            raise ValueError("BM25 index not built")

        # Tokenize & score
        tokenized = query.lower().split()
        if not tokenized:
            logger.warning("Query tokenized to empty list")
            return []

        scores = self.bm25_index.get_scores(tokenized)
        if scores is None or not isinstance(scores, np.ndarray):
            raise RuntimeError("BM25 returned invalid scores")
        if len(scores) != len(self.corpus):
            raise RuntimeError("BM25 scores length mismatch")
        if not len(scores):
            return []

        # Build top-k results
        top_idxs = np.argsort(-scores)[:top_k]
        results: List[Dict[str, Any]] = []
        for idx in top_idxs:
            results.append({
                "id": self.chunk_ids[idx],
                "text": self.corpus[idx],
                "score": float(scores[idx]),
                "search_type": "keyword_fallback"
            })

        latency = time.time() - start_time
        logger.info(f"Keyword fallback returned {len(results)} in {latency:.4f}s")
        return results
