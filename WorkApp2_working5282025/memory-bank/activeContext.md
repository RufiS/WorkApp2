# WorkApp2 Project Context (Updated 5/26/2025)

## Core System Architecture
- **Document Processing Pipeline**: 
  - Uses `RecursiveCharacterTextSplitter` for chunking with configurable size/overlap parameters
  - Implements FAISS-based vector search with hybrid BM25/FAISS support
  - Supports PDF, TXT, DOCX document formats through unified processing
  - Includes PDF hyperlink extraction and cross-platform path resolution

## Key Components
1. **Retrieval System**:
   - Hybrid retrieval (BM25 + FAISS) with configurable vector-lexical weighting
   - Enhanced mode enables advanced reranking and freshness checks
   - Automatic index health monitoring and rebuild tools

2. **LLM Service Layer**:
   - Advanced retry decorators with backoff factors and error tracking
   - Streaming support for token-by-token updates
   - Configurable model selection (extraction vs formatting)

3. **Streamlit UI**:
   - Interactive chat interface with progress bars and confidence meters
   - Responsive layout with debug information expanders
   - Enhanced error messages with actionable suggestions

4. **Index Management**:
   - Automatic freshness checks and dimension-mismatch recovery
   - GPU/CPU toggle for FAISS operations
   - Configurable batch sizes and profiling hooks

2. **LLM Integration**:
   - Separate extraction and formatting models (currently using GPT-3.5 Turbo)
   - Async processing with progress tracking
   - Error handling with retry mechanisms

3. **UI Enhancements**:
   - Progress tracking with stage-based visualization
   - Confidence meter for answer quality
   - Debug information expanders
   - Enhanced error messages with suggestions

## Performance Features
- GPU acceleration for FAISS operations (configurable)
- Index freshness monitoring
- Memory-efficient chunk processing
- Configurable retrieval parameters (top_k, similarity threshold)

## Error Handling Improvements
- Advanced retry logic with backoff factors (v1.2)
- Comprehensive error tracking system with severity levels
- Fallback UI components implementation for all major browsers
- Centralized error logging system with automatic alerting

## Recent Updates
- Unified configuration management system (v2.1)
- Enhanced index manager with health checks and freshness monitoring
- Improved document processing pipeline with PDF hyperlink extraction
- Better progress visualization for users with stage-based tracking

## Current Configuration
```json
{
  "retrieval": {
    "top_k": 15,
    "similarity_threshold": 0.8,
    "chunk_size": 600,
    "chunk_overlap": 120,
    "enhanced_mode": true,
    "vector_weight": 0.75
  },
  "performance": {
    "use_gpu_for_faiss": true,
    "enable_reranking": true,
    "max_concurrent_queries": 5
  }
}
```

## System Status
- Index contains [dynamic value] chunks from [dynamic value] files
- Last index update: [dynamic value]
- Current model configurations: 
  - Extraction: gpt-3.5-turbo
  - Formatting: gpt-3.5-turbo
