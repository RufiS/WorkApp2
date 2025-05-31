"""Query Data Models for WorkApp2.

Pydantic models for query processing and search results.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
from .document_models import ChunkModel


class QueryRequest(BaseModel):
    """Model for query requests."""
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        validate_default=True
    )
    
    query: str = Field(..., description="The user's question or search query", min_length=1)
    query_id: Optional[str] = Field(None, description="Unique identifier for this query")
    search_method: Literal["vector", "hybrid", "reranking"] = Field(
        "vector", description="Search method to use"
    )
    top_k: int = Field(5, description="Number of results to return", ge=1, le=50)
    rerank_top_k: Optional[int] = Field(None, description="Number of results to rerank", ge=1, le=100)
    include_scores: bool = Field(True, description="Whether to include relevance scores")
    include_metadata: bool = Field(True, description="Whether to include chunk metadata")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional search filters")
    user_id: Optional[str] = Field(None, description="User identifier for tracking")
    session_id: Optional[str] = Field(None, description="Session identifier")


class SearchResult(BaseModel):
    """Model for individual search results."""
    model_config = ConfigDict(validate_assignment=True)
    
    chunk: ChunkModel = Field(..., description="The retrieved text chunk")
    relevance_score: float = Field(..., description="Relevance score for this result")
    rank: int = Field(..., description="Rank position in results", ge=1)
    search_method: str = Field(..., description="Method used to retrieve this result")
    retrieval_time: Optional[float] = Field(None, description="Time taken to retrieve this result")
    
    @property
    def text_preview(self) -> str:
        """Get a preview of the chunk text."""
        text = self.chunk.text
        return text[:200] + "..." if len(text) > 200 else text


class QueryResponse(BaseModel):
    """Model for query responses."""
    model_config = ConfigDict(validate_assignment=True)
    
    query_id: str = Field(..., description="Unique identifier for the query")
    original_query: str = Field(..., description="The original query text")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_method: str = Field(..., description="Search method used")
    processing_time: float = Field(..., description="Total processing time in seconds")
    retrieval_time: float = Field(..., description="Time spent on retrieval")
    reranking_time: Optional[float] = Field(None, description="Time spent on reranking if applicable")
    answer: Optional[str] = Field(None, description="Generated answer from LLM")
    answer_time: Optional[float] = Field(None, description="Time spent generating answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    @property
    def has_results(self) -> bool:
        """Check if the response has any results."""
        return len(self.results) > 0
    
    @property
    def avg_relevance_score(self) -> float:
        """Get average relevance score of results."""
        if not self.results:
            return 0.0
        return sum(result.relevance_score for result in self.results) / len(self.results)
    
    @property
    def max_relevance_score(self) -> float:
        """Get maximum relevance score."""
        if not self.results:
            return 0.0
        return max(result.relevance_score for result in self.results)
    
    def get_top_results(self, n: int = 3) -> List[SearchResult]:
        """Get top N results by relevance score."""
        return sorted(self.results, key=lambda x: x.relevance_score, reverse=True)[:n]


class QueryMetrics(BaseModel):
    """Model for query performance metrics."""
    model_config = ConfigDict(validate_assignment=True)
    
    query_id: str = Field(..., description="Query identifier")
    search_method: str = Field(..., description="Search method used")
    total_processing_time: float = Field(..., description="Total processing time")
    retrieval_time: float = Field(..., description="Retrieval time")
    reranking_time: Optional[float] = Field(None, description="Reranking time")
    answer_generation_time: Optional[float] = Field(None, description="Answer generation time")
    results_count: int = Field(..., description="Number of results returned")
    avg_score: float = Field(..., description="Average relevance score")
    max_score: float = Field(..., description="Maximum relevance score")
    user_satisfaction: Optional[int] = Field(None, description="User satisfaction rating 1-5")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")


class QuerySession(BaseModel):
    """Model for query sessions."""
    model_config = ConfigDict(validate_assignment=True)
    
    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    queries: List[QueryResponse] = Field(default_factory=list, description="Queries in this session")
    started_at: datetime = Field(default_factory=datetime.now, description="Session start time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")
    
    @property
    def query_count(self) -> int:
        """Get number of queries in this session."""
        return len(self.queries)
    
    @property
    def total_session_time(self) -> float:
        """Get total session duration in seconds."""
        return (self.last_activity - self.started_at).total_seconds()
    
    @property
    def avg_processing_time(self) -> float:
        """Get average query processing time."""
        if not self.queries:
            return 0.0
        return sum(q.processing_time for q in self.queries) / len(self.queries)
    
    def add_query(self, query_response: QueryResponse) -> None:
        """Add a query response to this session."""
        self.queries.append(query_response)
        self.last_activity = datetime.now()
