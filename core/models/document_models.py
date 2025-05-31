"""Document Data Models for WorkApp2.

Pydantic models for document processing, replacing dictionary usage.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class MetadataModel(BaseModel):
    """Model for document metadata."""
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        validate_default=True
    )
    
    source: str = Field(..., description="Source file path or identifier")
    page: Optional[int] = Field(None, description="Page number if applicable")
    section: Optional[str] = Field(None, description="Section or chapter identifier")
    title: Optional[str] = Field(None, description="Document or section title")
    author: Optional[str] = Field(None, description="Document author")
    created_date: Optional[datetime] = Field(None, description="Document creation date")
    modified_date: Optional[datetime] = Field(None, description="Last modification date")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    mime_type: Optional[str] = Field(None, description="MIME type of the document")
    language: Optional[str] = Field("en", description="Document language")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


class ChunkModel(BaseModel):
    """Model for text chunks from document processing."""
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        validate_default=True
    )
    
    text: str = Field(..., description="The text content of the chunk", min_length=1)
    chunk_id: Optional[str] = Field(None, description="Unique identifier for the chunk")
    start_index: Optional[int] = Field(None, description="Start character index in original document")
    end_index: Optional[int] = Field(None, description="End character index in original document")
    token_count: Optional[int] = Field(None, description="Number of tokens in the chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")
    score: Optional[float] = Field(None, description="Relevance score from search")
    metadata: MetadataModel = Field(..., description="Metadata associated with this chunk")
    
    @property
    def length(self) -> int:
        """Get the character length of the text."""
        return len(self.text)
    
    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return len(self.text.split())


class DocumentModel(BaseModel):
    """Model for complete documents."""
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        validate_default=True
    )
    
    document_id: str = Field(..., description="Unique identifier for the document")
    file_path: str = Field(..., description="Path to the source file")
    content: str = Field(..., description="Full text content of the document")
    chunks: List[ChunkModel] = Field(default_factory=list, description="Text chunks extracted from document")
    metadata: MetadataModel = Field(..., description="Document metadata")
    processing_stats: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")
    processed_at: datetime = Field(default_factory=datetime.now, description="When the document was processed")
    
    @property
    def chunk_count(self) -> int:
        """Get the number of chunks in this document."""
        return len(self.chunks)
    
    @property
    def total_length(self) -> int:
        """Get total character length of all chunks."""
        return sum(chunk.length for chunk in self.chunks)
    
    @property
    def has_embeddings(self) -> bool:
        """Check if all chunks have embeddings."""
        return all(chunk.embedding is not None for chunk in self.chunks)
    
    def get_chunks_by_score(self, min_score: float = 0.0) -> List[ChunkModel]:
        """Get chunks filtered by minimum score."""
        return [chunk for chunk in self.chunks if chunk.score and chunk.score >= min_score]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkModel]:
        """Get a specific chunk by its ID."""
        return next((chunk for chunk in self.chunks if chunk.chunk_id == chunk_id), None)


class DocumentCollectionModel(BaseModel):
    """Model for a collection of documents."""
    model_config = ConfigDict(validate_assignment=True)
    
    collection_id: str = Field(..., description="Unique identifier for the collection")
    name: str = Field(..., description="Human-readable name for the collection")
    description: Optional[str] = Field(None, description="Description of the collection")
    documents: List[DocumentModel] = Field(default_factory=list, description="Documents in the collection")
    created_at: datetime = Field(default_factory=datetime.now, description="Collection creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    
    @property
    def document_count(self) -> int:
        """Get the number of documents in the collection."""
        return len(self.documents)
    
    @property
    def total_chunk_count(self) -> int:
        """Get total number of chunks across all documents."""
        return sum(doc.chunk_count for doc in self.documents)
    
    def get_document_by_id(self, document_id: str) -> Optional[DocumentModel]:
        """Get a specific document by its ID."""
        return next((doc for doc in self.documents if doc.document_id == document_id), None)
    
    def get_documents_by_source(self, source_pattern: str) -> List[DocumentModel]:
        """Get documents matching a source pattern."""
        return [doc for doc in self.documents if source_pattern in doc.metadata.source]
