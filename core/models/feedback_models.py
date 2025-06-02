"""Feedback data models for WorkApp2.

Models for capturing user feedback on query responses with comprehensive context.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import uuid
import hashlib


class FeedbackRequest(BaseModel):
    """User feedback input model."""
    
    feedback_type: str = Field(..., description="Type of feedback: 'positive' or 'negative'")
    feedback_text: Optional[str] = Field(None, description="Optional user comment")


class PerformanceMetrics(BaseModel):
    """Performance timing metrics for the query processing."""
    
    total_processing_time: float = Field(..., description="Total time from query to answer")
    retrieval_time: float = Field(..., description="Time spent on retrieval")
    extraction_time: Optional[float] = Field(None, description="Time spent on extraction")
    formatting_time: Optional[float] = Field(None, description="Time spent on formatting")
    feedback_delay_seconds: Optional[float] = Field(None, description="Time between answer and feedback")


class RetrievalMetrics(BaseModel):
    """Metrics about the retrieval process and quality."""
    
    engine_used: str = Field(..., description="Search engine type used")
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved")
    similarity_scores: List[float] = Field(default_factory=list, description="Similarity scores for retrieved chunks")
    avg_similarity: float = Field(..., description="Average similarity score")
    max_similarity: float = Field(..., description="Maximum similarity score")
    context_length: int = Field(..., description="Length of retrieved context in characters")
    context_quality_score: Optional[float] = Field(None, description="Quality score of context")
    fallback_used: bool = Field(False, description="Whether fallback retrieval was used")


class ConfigSnapshot(BaseModel):
    """Snapshot of configuration at query time."""
    
    search_engine: str = Field(..., description="Current search engine configuration")
    similarity_threshold: float = Field(..., description="Similarity threshold setting")
    top_k: int = Field(..., description="Top-k retrieval setting")
    enhanced_mode: bool = Field(..., description="Enhanced mode setting")
    enable_reranking: bool = Field(..., description="Reranking enabled setting")
    vector_weight: Optional[float] = Field(None, description="Vector weight in hybrid mode")


class SessionInfo(BaseModel):
    """Session and query context information."""
    
    session_id: str = Field(..., description="Unique session identifier")
    query_hash: int = Field(..., description="Hash of the query for deduplication")
    production_mode: bool = Field(..., description="Whether app is in production mode")
    query_sequence_in_session: int = Field(1, description="Query number in this session")
    time_since_last_query: Optional[float] = Field(None, description="Seconds since last query")


class AnswerStructure(BaseModel):
    """Analysis of answer structure and formatting."""
    
    has_numbered_steps: bool = Field(False, description="Answer contains numbered steps")
    has_bullet_points: bool = Field(False, description="Answer contains bullet points")
    has_examples: bool = Field(False, description="Answer contains examples")


class ContentMetrics(BaseModel):
    """Metrics about question, context, and answer content."""
    
    question_length: int = Field(..., description="Length of question in characters")
    context_length: int = Field(..., description="Length of context in characters")
    answer_length: int = Field(..., description="Length of answer in characters")
    answer_structure: AnswerStructure = Field(default_factory=AnswerStructure)
    context_sources: List[str] = Field(default_factory=list, description="Source documents for context")
    context_preview: str = Field(..., description="First 100 characters of context")


class ModelInfo(BaseModel):
    """Information about models used in processing."""
    
    extraction_model: str = Field(..., description="Model used for extraction")
    formatting_model: str = Field(..., description="Model used for formatting")
    embedding_model: str = Field(..., description="Model used for embeddings")
    reranker_model: Optional[str] = Field(None, description="Model used for reranking")


class ErrorInfo(BaseModel):
    """Error context if any errors occurred."""
    
    had_errors: bool = Field(False, description="Whether any errors occurred")
    error_stage: Optional[str] = Field(None, description="Stage where error occurred")
    error_message: Optional[str] = Field(None, description="Error message")
    recovery_attempted: bool = Field(False, description="Whether error recovery was attempted")


class FeedbackEntry(BaseModel):
    """Complete feedback entry with all context."""
    
    # Core required fields
    question: str = Field(..., description="User's original question")
    context: str = Field(..., description="Retrieved context used for answer")
    answer: str = Field(..., description="Generated answer")
    
    # Feedback data
    feedback: Dict[str, Any] = Field(..., description="User feedback data")
    
    # Additional context fields
    performance: PerformanceMetrics = Field(..., description="Performance metrics")
    retrieval_metrics: RetrievalMetrics = Field(..., description="Retrieval quality metrics")
    config_state: ConfigSnapshot = Field(..., description="Configuration snapshot")
    session_info: SessionInfo = Field(..., description="Session context")
    content_metrics: ContentMetrics = Field(..., description="Content analysis")
    model_info: ModelInfo = Field(..., description="Model information")
    error_info: ErrorInfo = Field(default_factory=ErrorInfo, description="Error context")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @staticmethod
    def generate_query_hash(query: str) -> int:
        """Generate a consistent hash for a query."""
        return hash(query.strip().lower())

    @staticmethod
    def analyze_answer_structure(answer: str) -> AnswerStructure:
        """Analyze the structure of an answer."""
        answer_lower = answer.lower()
        return AnswerStructure(
            has_numbered_steps=any(f"{i}." in answer or f"{i})" in answer for i in range(1, 11)),
            has_bullet_points="•" in answer or "*" in answer or "-" in answer.split("\n"),
            has_examples="example" in answer_lower or "for instance" in answer_lower
        )

    @classmethod
    def create_from_query_results(
        cls,
        question: str,
        context: str,
        answer: str,
        feedback_request: FeedbackRequest,
        query_results: Dict[str, Any],
        config_snapshot: Dict[str, Any],
        session_context: Dict[str, Any],
        model_config: Dict[str, Any],
        feedback_timestamp: Optional[datetime] = None
    ) -> "FeedbackEntry":
        """Create a FeedbackEntry from query processing results."""
        
        # Extract timing information
        total_time = query_results.get("total_time", 0.0)
        retrieval_time = query_results.get("retrieval", {}).get("time", 0.0)
        
        # Calculate feedback delay if provided
        feedback_delay = None
        if feedback_timestamp and "processing_start" in query_results:
            processing_start = query_results["processing_start"]
            if isinstance(processing_start, datetime):
                feedback_delay = (feedback_timestamp - processing_start).total_seconds()
        
        performance = PerformanceMetrics(
            total_processing_time=total_time,
            retrieval_time=retrieval_time,
            feedback_delay_seconds=feedback_delay
        )
        
        # Extract retrieval metrics
        retrieval_data = query_results.get("retrieval", {})
        scores = retrieval_data.get("scores", [])
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        
        retrieval_metrics = RetrievalMetrics(
            engine_used=config_snapshot.get("search_engine", "unknown"),
            chunks_retrieved=retrieval_data.get("chunks", 0),
            similarity_scores=scores,
            avg_similarity=avg_score,
            max_similarity=max_score,
            context_length=len(context)
        )
        
        # Create config snapshot
        config_state = ConfigSnapshot(
            search_engine=config_snapshot.get("search_engine", "unknown"),
            similarity_threshold=config_snapshot.get("similarity_threshold", 0.0),
            top_k=config_snapshot.get("top_k", 0),
            enhanced_mode=config_snapshot.get("enhanced_mode", False),
            enable_reranking=config_snapshot.get("enable_reranking", False),
            vector_weight=config_snapshot.get("vector_weight")
        )
        
        # Create session info
        session_info = SessionInfo(
            session_id=session_context.get("session_id", "unknown"),
            query_hash=cls.generate_query_hash(question),
            production_mode=session_context.get("production_mode", False),
            query_sequence_in_session=session_context.get("query_sequence", 1),
            time_since_last_query=session_context.get("time_since_last_query")
        )
        
        # Analyze content
        answer_structure = cls.analyze_answer_structure(answer)
        content_metrics = ContentMetrics(
            question_length=len(question),
            context_length=len(context),
            answer_length=len(answer),
            answer_structure=answer_structure,
            context_sources=["unknown"],  # TODO: Extract from retrieval results
            context_preview=context[:100] + "..." if len(context) > 100 else context
        )
        
        # Model information
        model_info = ModelInfo(
            extraction_model=model_config.get("extraction_model", "unknown"),
            formatting_model=model_config.get("formatting_model", "unknown"),
            embedding_model=model_config.get("embedding_model", "unknown"),
            reranker_model=model_config.get("reranker_model")
        )
        
        # Error information
        error_info = ErrorInfo(
            had_errors="error" in query_results,
            error_stage=query_results.get("error_stage"),
            error_message=query_results.get("error"),
            recovery_attempted=query_results.get("recovery_attempted", False)
        )
        
        # Create feedback data
        feedback_data = {
            "type": feedback_request.feedback_type,
            "text": feedback_request.feedback_text,
            "timestamp": (feedback_timestamp or datetime.utcnow()).isoformat()
        }
        
        return cls(
            question=question,
            context=context,
            answer=answer,
            feedback=feedback_data,
            performance=performance,
            retrieval_metrics=retrieval_metrics,
            config_state=config_state,
            session_info=session_info,
            content_metrics=content_metrics,
            model_info=model_info,
            error_info=error_info
        )
