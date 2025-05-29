A high-performance, production-grade question-answering application built on Streamlit, FAISS, BM25 and your choice of LLM. It indexes arbitrary text or PDF documents, retrieves relevant chunks, cleans and enriches context, then dispatches to an LLM with retry and error-tracking wrappers. Designed for reliability, transparency and easy extension.

---

## 🔍 Features

- **Robust document ingestion**: chunking, TTL-based caching, path-resolution fixes for cross-platform support  
- **Hybrid retrieval**: BM25 + FAISS vector search with configurable vector-vs-lexical weighting  
- **Index management**: automatic freshness checks, rebuild tools, dimension-mismatch recovery  
- **LLM service layer**: advanced retry decorators, error tracking, streaming support for token-by-token updates  
- **Streamlit UI**: interactive chat interface, progress bars, hyperlink extraction, responsive layout  
- **Performance tuning**: configurable batching, GPU/CPU toggles, profiling hooks  