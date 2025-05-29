# System Patterns

## System Architecture
The application follows a modular architecture with the following components:
1. **Document Ingestion**: Handles the processing and indexing of documents.
2. **Retrieval System**: Implements hybrid retrieval using BM25 and FAISS.
3. **Context Processing**: Cleans and enriches the retrieved context.
4. **LLM Service**: Interacts with the language model for generating answers.
5. **UI**: Provides a user-friendly interface using Streamlit.

## Key Technical Decisions
- **Hybrid Retrieval**: Combining BM25 and FAISS for better retrieval performance.
- **Modular Design**: Separating concerns into distinct modules for maintainability.
- **Error Handling**: Implementing retry mechanisms and error tracking.

## Design Patterns in Use
- **Decorator Pattern**: For adding functionality to existing methods (e.g., retry logic).
- **Singleton Pattern**: For ensuring a single instance of certain components (e.g., index manager).
- **Factory Pattern**: For creating objects without specifying the exact class (e.g., LLM service).

## Component Relationships
- **Document Ingestion** → **Retrieval System** → **Context Processing** → **LLM Service** → **UI**

## Critical Implementation Paths
- Document ingestion and indexing
- Hybrid retrieval mechanism
- Context processing pipeline
- LLM integration and error handling
