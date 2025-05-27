# Tech Context

## Technologies Used
- **Streamlit**: For building the user interface (v1.24.0)
- **FAISS**: For vector similarity search (v1.7.3)
- **BM25**: For lexical ranking (via `rank_bm25` library)
- **Python**: Primary programming language (v3.11.6)
- **LLM**: Anthropic Claude 3.5 Sonnet (with fallback to GPT-4o)

## Development Setup
- **Environment**: Python 3.11.6
- **Dependencies**: Managed via `requirements.txt` (includes Streamlit, FAISS, rank_bm25, langchain, anthropic, openai)
- **Tools**: VSCode with Python extension, Git for version control

## Technical Constraints
- **Cross-platform Support**: Linux/macOS/Windows compatibility ensured
- **Performance**: Optimized with GPU acceleration via CUDA 12.1
- **Scalability**: Designed for horizontal scaling with Docker Compose

## Dependencies
- **Streamlit**: For UI components and real-time visualization
- **FAISS**: For efficient vector search implementation
- **BM25**: For lexical search via `rank_bm25` library
- **LLM**: Anthropic Claude 3.5 Sonnet (primary) and OpenAI GPT-4o (fallback)

## Tool Usage Patterns
- **Streamlit**: Interactive dashboard with real-time query visualization
- **FAISS**: Vector similarity search with GPU acceleration
- **BM25**: Lexical ranking for document retrieval
- **LLM**: Answer generation with streaming support and retry decorators

## Development Setup
- **Environment**: Python 3.9+
- **Dependencies**: Listed in requirements.txt
- **Tools**: VSCode, Git

## Technical Constraints
- **Cross-platform Support**: Ensure compatibility across different operating systems
- **Performance**: Optimize for speed and efficiency
- **Scalability**: Design for handling large volumes of data

## Dependencies
- **Streamlit**: For UI components
- **FAISS**: For vector search implementation
- **BM25**: For lexical search implementation
- **LLM**: For natural language processing

## Tool Usage Patterns
- **Streamlit**: For creating interactive web applications
- **FAISS**: For efficient similarity search and clustering of dense vectors
- **BM25**: For effective information retrieval from text documents
- **LLM**: For generating human-like text based on input prompts
