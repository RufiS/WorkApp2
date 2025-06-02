"""Feedback service for WorkApp2.

Handles feedback data storage, retrieval, and logging operations.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import asyncio
from threading import Lock

from core.models.feedback_models import FeedbackEntry, FeedbackRequest
from utils.error_handling.enhanced_decorators import with_error_tracking


logger = logging.getLogger(__name__)


class FeedbackService:
    """Service for managing user feedback on query responses."""
    
    def __init__(self, log_directory: str = "logs"):
        """Initialize the feedback service.
        
        Args:
            log_directory: Directory to store feedback logs
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        self.feedback_log_path = self.log_directory / "feedback_detailed.log"
        self.summary_log_path = self.log_directory / "feedback_summary.log"
        
        # Thread-safe file writing
        self._write_lock = Lock()
        
        # In-memory feedback tracking for analytics
        self._recent_feedback = []
        self._max_recent_feedback = 100
        
        logger.info(f"FeedbackService initialized with log directory: {self.log_directory}")

    @with_error_tracking()
    def store_feedback(
        self,
        question: str,
        context: str,
        answer: str,
        feedback_request: FeedbackRequest,
        query_results: Dict[str, Any],
        config_snapshot: Dict[str, Any],
        session_context: Dict[str, Any],
        model_config: Dict[str, Any]
    ) -> str:
        """Store user feedback with complete context.
        
        Args:
            question: Original user question
            context: Retrieved context used for answer
            answer: Generated answer
            feedback_request: User feedback data
            query_results: Complete query processing results
            config_snapshot: Current configuration state
            session_context: Session information
            model_config: Model configuration
            
        Returns:
            Feedback ID for tracking
            
        Raises:
            Exception: If feedback storage fails
        """
        try:
            feedback_timestamp = datetime.utcnow()
            
            # Create comprehensive feedback entry
            feedback_entry = FeedbackEntry.create_from_query_results(
                question=question,
                context=context,
                answer=answer,
                feedback_request=feedback_request,
                query_results=query_results,
                config_snapshot=config_snapshot,
                session_context=session_context,
                model_config=model_config,
                feedback_timestamp=feedback_timestamp
            )
            
            # Store to detailed log
            self._write_detailed_log(feedback_entry)
            
            # Store to summary log
            self._write_summary_log(feedback_entry)
            
            # Add to recent feedback for analytics
            self._add_to_recent_feedback(feedback_entry)
            
            logger.info(f"Stored {feedback_request.feedback_type} feedback for query: {question[:50]}...")
            
            return feedback_entry.feedback_id
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {str(e)}", exc_info=True)
            raise

    def _write_detailed_log(self, feedback_entry: FeedbackEntry) -> None:
        """Write detailed feedback entry to log file.
        
        Args:
            feedback_entry: Complete feedback entry to log
        """
        with self._write_lock:
            try:
                # Convert to JSON with proper serialization
                feedback_data = feedback_entry.model_dump()
                
                # Ensure datetime objects are serialized properly
                if isinstance(feedback_data.get("timestamp"), datetime):
                    feedback_data["timestamp"] = feedback_data["timestamp"].isoformat()
                
                # Write as single line JSON for easy parsing
                with open(self.feedback_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
                    
            except Exception as e:
                logger.error(f"Failed to write detailed feedback log: {str(e)}")
                raise

    def _write_summary_log(self, feedback_entry: FeedbackEntry) -> None:
        """Write summary feedback entry to summary log file.
        
        Args:
            feedback_entry: Feedback entry to summarize
        """
        with self._write_lock:
            try:
                # Create condensed summary
                summary = {
                    "timestamp": feedback_entry.timestamp.isoformat(),
                    "feedback_id": feedback_entry.feedback_id,
                    "feedback_type": feedback_entry.feedback.get("type"),
                    "query_hash": feedback_entry.session_info.query_hash,
                    "question_preview": feedback_entry.question[:100],
                    "answer_length": feedback_entry.content_metrics.answer_length,
                    "retrieval_score": feedback_entry.retrieval_metrics.max_similarity,
                    "processing_time": feedback_entry.performance.total_processing_time,
                    "production_mode": feedback_entry.session_info.production_mode,
                    "engine_used": feedback_entry.retrieval_metrics.engine_used
                }
                
                with open(self.summary_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(summary, ensure_ascii=False) + "\n")
                    
            except Exception as e:
                logger.error(f"Failed to write summary feedback log: {str(e)}")
                # Don't raise here since detailed log is primary

    def _add_to_recent_feedback(self, feedback_entry: FeedbackEntry) -> None:
        """Add feedback to recent feedback list for analytics.
        
        Args:
            feedback_entry: Feedback entry to add
        """
        try:
            # Add to beginning of list
            self._recent_feedback.insert(0, feedback_entry)
            
            # Trim to max size
            if len(self._recent_feedback) > self._max_recent_feedback:
                self._recent_feedback = self._recent_feedback[:self._max_recent_feedback]
                
        except Exception as e:
            logger.warning(f"Failed to add to recent feedback: {str(e)}")

    def get_recent_feedback_summary(self, limit: int = 20) -> Dict[str, Any]:
        """Get summary of recent feedback for analytics.
        
        Args:
            limit: Maximum number of feedback entries to include
            
        Returns:
            Dictionary with feedback analytics
        """
        try:
            recent = self._recent_feedback[:limit]
            
            if not recent:
                return {"total": 0, "positive": 0, "negative": 0, "entries": []}
            
            positive = sum(1 for entry in recent if entry.feedback.get("type") == "positive")
            negative = sum(1 for entry in recent if entry.feedback.get("type") == "negative")
            
            # Create summary entries
            summary_entries = []
            for entry in recent:
                summary_entries.append({
                    "timestamp": entry.timestamp.isoformat(),
                    "feedback_type": entry.feedback.get("type"),
                    "question_preview": entry.question[:100],
                    "retrieval_score": entry.retrieval_metrics.max_similarity,
                    "processing_time": entry.performance.total_processing_time,
                    "feedback_delay": entry.performance.feedback_delay_seconds
                })
            
            return {
                "total": len(recent),
                "positive": positive,
                "negative": negative,
                "positive_rate": positive / len(recent) if recent else 0,
                "avg_retrieval_score": sum(e.retrieval_metrics.max_similarity for e in recent) / len(recent),
                "avg_processing_time": sum(e.performance.total_processing_time for e in recent) / len(recent),
                "entries": summary_entries
            }
            
        except Exception as e:
            logger.error(f"Failed to get recent feedback summary: {str(e)}")
            return {"error": str(e)}

    def load_feedback_from_logs(self, limit: Optional[int] = None) -> List[FeedbackEntry]:
        """Load feedback entries from log files.
        
        Args:
            limit: Maximum number of entries to load (most recent first)
            
        Returns:
            List of feedback entries
        """
        entries = []
        
        try:
            if not self.feedback_log_path.exists():
                return entries
            
            with open(self.feedback_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Process in reverse order to get most recent first
            for line in reversed(lines):
                if limit and len(entries) >= limit:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    entry = FeedbackEntry(**data)
                    entries.append(entry)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse feedback log line: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to load feedback from logs: {str(e)}")
            
        return entries

    def get_feedback_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive feedback analytics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with analytics data
        """
        try:
            # Load recent feedback from logs
            all_feedback = self.load_feedback_from_logs(limit=1000)
            
            # Filter by date range
            from datetime import timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            recent_feedback = [
                entry for entry in all_feedback 
                if entry.timestamp >= cutoff_date
            ]
            
            if not recent_feedback:
                return {"period_days": days, "total_feedback": 0}
            
            # Calculate metrics
            total = len(recent_feedback)
            positive = sum(1 for entry in recent_feedback if entry.feedback.get("type") == "positive")
            negative = sum(1 for entry in recent_feedback if entry.feedback.get("type") == "negative")
            
            # Performance correlations
            positive_entries = [e for e in recent_feedback if e.feedback.get("type") == "positive"]
            negative_entries = [e for e in recent_feedback if e.feedback.get("type") == "negative"]
            
            analytics = {
                "period_days": days,
                "total_feedback": total,
                "positive_count": positive,
                "negative_count": negative,
                "positive_rate": positive / total if total > 0 else 0,
                "avg_retrieval_score": sum(e.retrieval_metrics.max_similarity for e in recent_feedback) / total,
                "avg_processing_time": sum(e.performance.total_processing_time for e in recent_feedback) / total,
                "positive_avg_score": sum(e.retrieval_metrics.max_similarity for e in positive_entries) / len(positive_entries) if positive_entries else 0,
                "negative_avg_score": sum(e.retrieval_metrics.max_similarity for e in negative_entries) / len(negative_entries) if negative_entries else 0,
                "engine_breakdown": self._get_engine_breakdown(recent_feedback),
                "common_negative_patterns": self._analyze_negative_feedback(negative_entries[:10])
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get feedback analytics: {str(e)}")
            return {"error": str(e)}

    def _get_engine_breakdown(self, feedback_entries: List[FeedbackEntry]) -> Dict[str, Any]:
        """Analyze feedback by search engine type.
        
        Args:
            feedback_entries: List of feedback entries to analyze
            
        Returns:
            Dictionary with engine breakdown
        """
        engine_stats = {}
        
        for entry in feedback_entries:
            engine = entry.retrieval_metrics.engine_used
            if engine not in engine_stats:
                engine_stats[engine] = {"total": 0, "positive": 0, "negative": 0}
            
            engine_stats[engine]["total"] += 1
            if entry.feedback.get("type") == "positive":
                engine_stats[engine]["positive"] += 1
            elif entry.feedback.get("type") == "negative":
                engine_stats[engine]["negative"] += 1
        
        # Calculate rates
        for engine, stats in engine_stats.items():
            if stats["total"] > 0:
                stats["positive_rate"] = stats["positive"] / stats["total"]
        
        return engine_stats

    def _analyze_negative_feedback(self, negative_entries: List[FeedbackEntry]) -> List[Dict[str, Any]]:
        """Analyze patterns in negative feedback.
        
        Args:
            negative_entries: List of negative feedback entries
            
        Returns:
            List of analysis patterns
        """
        patterns = []
        
        for entry in negative_entries:
            pattern = {
                "question_preview": entry.question[:100],
                "retrieval_score": entry.retrieval_metrics.max_similarity,
                "chunks_found": entry.retrieval_metrics.chunks_retrieved,
                "answer_length": entry.content_metrics.answer_length,
                "user_comment": entry.feedback.get("text", ""),
                "processing_time": entry.performance.total_processing_time
            }
            patterns.append(pattern)
        
        return patterns

    async def store_feedback_async(
        self,
        question: str,
        context: str,
        answer: str,
        feedback_request: FeedbackRequest,
        query_results: Dict[str, Any],
        config_snapshot: Dict[str, Any],
        session_context: Dict[str, Any],
        model_config: Dict[str, Any]
    ) -> str:
        """Async version of store_feedback.
        
        This allows feedback storage without blocking the UI.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.store_feedback,
            question,
            context,
            answer,
            feedback_request,
            query_results,
            config_snapshot,
            session_context,
            model_config
        )
