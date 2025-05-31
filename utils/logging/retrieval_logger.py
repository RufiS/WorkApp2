"""Enhanced Retrieval System Logger.

Provides comprehensive logging for retrieval operations including:
- Engine routing decisions
- Configuration snapshots
- Chunk-level scoring
- Context quality assessment
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class RetrievalLogger:
    """Enhanced logger for retrieval system operations."""
    
    def __init__(self, metrics_log_path: str = "logs/query_metrics.log"):
        """Initialize the retrieval logger.
        
        Args:
            metrics_log_path: Path to the metrics log file
        """
        self.metrics_log_path = metrics_log_path
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)
    
    def log_retrieval_operation(
        self,
        query: str,
        engine_used: str,
        config_snapshot: Dict[str, Any],
        retrieval_results: Tuple[str, float, int, List[float]],
        chunk_details: Optional[List[Dict[str, Any]]] = None,
        context_quality_score: Optional[float] = None,
        routing_reason: Optional[str] = None
    ) -> str:
        """Log a complete retrieval operation.
        
        Args:
            query: The search query
            engine_used: Which engine was used (vector/hybrid/reranking)
            config_snapshot: Current configuration state
            retrieval_results: Tuple of (context, time, count, scores)
            chunk_details: Detailed information about retrieved chunks
            context_quality_score: Assessed quality of retrieved context
            routing_reason: Why this engine was selected
            
        Returns:
            Session ID for tracking
        """
        session_id = str(uuid.uuid4())[:8]
        context, retrieval_time, chunk_count, similarity_scores = retrieval_results
        
        # Create comprehensive log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": session_id,
            "query": query,
            "query_hash": hash(query) % (10**8),  # For tracking identical queries
            "latency": retrieval_time,
            "hit_count": chunk_count,
            "metadata": {
                "search_type": engine_used,
                "top_k": config_snapshot.get("top_k", "unknown"),
                "similarity_threshold": config_snapshot.get("similarity_threshold", "unknown"),
                "fallback_used": False  # TODO: Implement fallback detection
            },
            "config_snapshot": config_snapshot,
            "engine_routing": {
                "selected_engine": engine_used,
                "routing_reason": routing_reason or f"Configuration-based selection: {engine_used}",
                "available_engines": ["vector", "hybrid", "reranking"]
            },
            "retrieval_details": {
                "chunks_retrieved": chunk_count,
                "similarity_scores": similarity_scores[:10] if similarity_scores else [],  # Top 10 scores
                "context_length": len(context),
                "context_preview": self._truncate_text(context, 200),
                "context_quality_score": context_quality_score,
                "avg_similarity": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0,
                "min_similarity": min(similarity_scores) if similarity_scores else 0.0,
                "max_similarity": max(similarity_scores) if similarity_scores else 0.0
            }
        }
        
        # Add chunk details if provided
        if chunk_details:
            log_entry["chunk_analysis"] = {
                "total_chunks_considered": len(chunk_details),
                "chunks_above_threshold": len([c for c in chunk_details if c.get("score", 0) >= config_snapshot.get("similarity_threshold", 0)]),
                "top_chunks": chunk_details[:5],  # Top 5 chunks with details
                "score_distribution": self._analyze_score_distribution(chunk_details)
            }
        
        # Write to enhanced query metrics log (all detailed info goes to file)
        self._write_to_metrics_log(log_entry)
        
        # Minimal console logging - no context content or detailed scores
        logger.debug(f"Retrieval logged: {session_id} - {engine_used} - {chunk_count} chunks")
        return session_id
    
    def log_engine_routing_decision(
        self,
        query: str,
        available_engines: List[str],
        selected_engine: str,
        config_state: Dict[str, Any],
        routing_logic: str
    ) -> None:
        """Log engine routing decision for debugging.
        
        Args:
            query: The search query
            available_engines: List of available engines
            selected_engine: Which engine was selected
            config_state: Current configuration state
            routing_logic: Explanation of routing logic
        """
        routing_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": "engine_routing",
            "query_preview": self._truncate_text(query, 50),
            "available_engines": available_engines,
            "selected_engine": selected_engine,
            "config_state": config_state,
            "routing_logic": routing_logic
        }
        
        # Append to metrics log
        try:
            with open(self.metrics_log_path, "a") as f:
                f.write(json.dumps(routing_entry, separators=(',', ':')) + "\n")
        except Exception as e:
            logger.error(f"Failed to write routing decision to log: {e}")
    
    def assess_context_quality(
        self,
        query: str,
        retrieved_context: str,
        chunk_scores: List[float]
    ) -> float:
        """Assess the quality of retrieved context.
        
        Args:
            query: The original query
            retrieved_context: The retrieved context
            chunk_scores: Similarity scores of retrieved chunks
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not retrieved_context or not chunk_scores:
            return 0.0
        
        # Simple heuristic-based quality assessment
        quality_factors = []
        
        # Factor 1: Average similarity score
        avg_score = sum(chunk_scores) / len(chunk_scores)
        quality_factors.append(avg_score)
        
        # Factor 2: Score consistency (lower variance = better)
        if len(chunk_scores) > 1:
            score_variance = sum((score - avg_score) ** 2 for score in chunk_scores) / len(chunk_scores)
            consistency_score = max(0, 1 - score_variance)
            quality_factors.append(consistency_score)
        
        # Factor 3: Context length appropriateness
        context_length = len(retrieved_context)
        if 500 <= context_length <= 5000:  # Reasonable context length
            length_score = 1.0
        elif context_length < 500:
            length_score = context_length / 500  # Penalize too short
        else:
            length_score = max(0.5, 1 - (context_length - 5000) / 10000)  # Penalize too long
        quality_factors.append(length_score)
        
        # Factor 4: Keyword overlap (simple check)
        query_words = set(query.lower().split())
        context_words = set(retrieved_context.lower().split())
        if query_words:
            keyword_overlap = len(query_words.intersection(context_words)) / len(query_words)
            quality_factors.append(keyword_overlap)
        
        # Calculate weighted average
        return sum(quality_factors) / len(quality_factors)
    
    def _write_to_metrics_log(self, log_entry: Dict[str, Any]) -> None:
        """Write log entry to metrics file.
        
        Args:
            log_entry: Dictionary containing log data
        """
        try:
            with open(self.metrics_log_path, "a") as f:
                f.write(json.dumps(log_entry, separators=(',', ':')) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to metrics log: {e}")
            # Try to create backup
            try:
                backup_path = f"logs/metrics_backup_{int(time.time())}.log"
                with open(backup_path, "w") as f:
                    f.write(json.dumps(log_entry, indent=2) + "\n")
                logger.info(f"Metrics saved to backup: {backup_path}")
            except Exception as backup_error:
                logger.error(f"Failed to create backup metrics log: {backup_error}")
    
    def _analyze_score_distribution(self, chunk_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of chunk scores.
        
        Args:
            chunk_details: List of chunk details with scores
            
        Returns:
            Dictionary with score distribution analysis
        """
        if not chunk_details:
            return {"error": "No chunk details provided"}
        
        scores = [chunk.get("score", 0) for chunk in chunk_details]
        if not scores:
            return {"error": "No scores found in chunk details"}
        
        # Calculate distribution statistics
        scores.sort(reverse=True)
        n = len(scores)
        
        return {
            "total_chunks": n,
            "highest_score": scores[0],
            "lowest_score": scores[-1],
            "median_score": scores[n // 2],
            "score_range": scores[0] - scores[-1],
            "top_quartile_avg": sum(scores[:n//4]) / (n//4) if n >= 4 else sum(scores) / n,
            "bottom_quartile_avg": sum(scores[-n//4:]) / (n//4) if n >= 4 else sum(scores) / n
        }
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text for logging.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            
        Returns:
            Truncated text with ellipsis if needed
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query entries from the log.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent log entries
        """
        try:
            entries = []
            if os.path.exists(self.metrics_log_path):
                with open(self.metrics_log_path, "r") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("event_type") != "engine_routing":  # Skip routing entries
                                entries.append(entry)
                        except json.JSONDecodeError:
                            continue
                            
            return entries[-limit:] if entries else []
        except Exception as e:
            logger.error(f"Failed to read recent queries: {e}")
            return []
    
    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in recent queries.
        
        Returns:
            Dictionary with query pattern analysis
        """
        recent_queries = self.get_recent_queries(50)
        if not recent_queries:
            return {"error": "No recent queries found"}
        
        # Analyze patterns
        engines_used = {}
        success_rates = {"total": 0, "successful": 0}
        avg_latency = []
        
        for query in recent_queries:
            # Engine usage
            engine = query.get("metadata", {}).get("search_type", "unknown")
            engines_used[engine] = engines_used.get(engine, 0) + 1
            
            # Success estimation (basic heuristic)
            success_rates["total"] += 1
            context_quality = query.get("retrieval_details", {}).get("context_quality_score", 0)
            if context_quality > 0.5:
                success_rates["successful"] += 1
            
            # Latency
            latency = query.get("latency", 0)
            if latency > 0:
                avg_latency.append(latency)
        
        return {
            "total_queries": len(recent_queries),
            "engines_used": engines_used,
            "estimated_success_rate": success_rates["successful"] / success_rates["total"] if success_rates["total"] > 0 else 0,
            "average_latency": sum(avg_latency) / len(avg_latency) if avg_latency else 0,
            "analysis_timestamp": datetime.utcnow().isoformat() + "Z"
        }


# Global logger instance
retrieval_logger = RetrievalLogger()
