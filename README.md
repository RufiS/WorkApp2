# WorkApp2 – Enhanced Streamlit Document QA App

A high-performance, production-grade question-answering application built on Streamlit, FAISS, BM25 and your choice of LLM. It indexes arbitrary text or PDF documents, retrieves relevant chunks, cleans and enriches context, then dispatches to an LLM with retry and error-tracking wrappers. Designed for reliability, transparency and easy extension.

---

## 🔍 Features

- **Robust document ingestion**: chunking, TTL-based caching, path-resolution fixes for cross-platform support  
- **Hybrid retrieval**: BM25 + FAISS vector search with configurable vector-vs-lexical weighting  
- **Index management**: automatic freshness checks, rebuild tools, dimension-mismatch recovery  
- **LLM service layer**: advanced retry decorators, error tracking, streaming support for token-by-token updates  
- **Streamlit UI**: interactive chat interface, progress bars, hyperlink extraction, responsive layout  
- **Performance tuning**: configurable batching, GPU/CPU toggles, profiling hooks  

---

## 📦 Installation

1. **Clone or unzip** the repository
   ```bash
   git clone <repo-url> WorkApp2
   cd WorkApp2
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure** (see below) and run
   ```bash
   streamlit run workapp3.py -- --rebuild-index
   ```

---

## ⚙️ Configuration

All runtime settings live in JSON and dataclasses under `utils/config_unified.py`. You can override defaults via:

- **`config.json`**  
- **`performance_config.json`**  
- **Environment variables** (read in via `config_manager`)

Key sections:

```jsonc
{
  "retrieval": {
    "vector_weight": 0.7,
    "bm25_k": 1.5,
    "bm25_b": 0.75
  },
  "model": {
    "provider": "openai",
    "name": "gpt-4",
    "temperature": 0.2,
    "max_tokens": 512
  },
  "ui": {
    "page_title": "Document QA",
    "icon": "📚",
    "subtitle": "Ask questions about your documents"
  },
  "performance": {
    "batch_size": 64,
    "n_gpu_layers": 20,
    "timeout_seconds": 60
  },
  "data_dir": "./data",
  "log_level": "INFO"
}
```

---

## 🚀 Usage

```bash
# rebuild FAISS index from scratch (slow)
streamlit run workapp3.py -- --rebuild-index

# dry-run index changes without persisting
streamlit run workapp3.py -- --dry-run

# verbose logging
streamlit run workapp3.py -- --verbose
```

Then point your browser at `http://localhost:8501`.

---

## 📁 Directory Structure

```
.
├── workapp3.py                # Main Streamlit entrypoint
├── requirements.txt
├── config.json                # Optional overrides
├── performance_config.json
├── Errors_Roadmap.md          # Pending fixes & progress tracker
├── current_index/             # Serialized chunk cache
│   └── chunks.txt
├── data/
│   └── index/                 # FAISS index & metadata
│       ├── index.faiss
│       ├── metadata.json
│       └── texts.npy
├── logs/
│   ├── query_metrics.log
│   └── workapp_errors.log
└── utils/
    ├── config_unified.py      # Dataclasses & config manager
    ├── document_processor_unified.py  # Chunking, caching
    ├── error_logging.py       # Standardized error handlers
    ├── enhanced_retrieval.py  # BM25 + vector retrieval helpers
    ├── fix_current_index.py   # Index-repair tools
    ├── llm_service.py         # Basic LLM calls & wrappers
    ├── llm_service_enhanced.py# Retry & error-tracking decorators
    ├── rebuild_index.py       # CLI index rebuild logic
    ├── unified_retrieval_system.py # End-to-end retrieval orchestration
    └── index_management/
        ├── index_manager_unified.py  # Freshness & rebuild logic
        ├── index_operations.py
        └── index_health.py
```

---

## 🔧 Module Breakdown

### `utils/config_unified.py`  
Defines `AppConfig`, `RetrievalConfig`, `ModelConfig`, `UIConfig`, `PerformanceConfig`. Loads/saves JSON, handles defaults and locks for thread safety.

### `utils/document_processor_unified.py`  
- **`DocumentProcessor`**:  
  - Reads files (PDFs, text)  
  - Splits into fixed-size chunks + metadata  
  - Caches splits with TTL and LRU eviction  
  - Ensures consistent path resolution  

### `utils/enhanced_retrieval.py`  
- Lexical (BM25) and vector search helpers  
- Score normalization and customizable weighting  

### `utils/index_management/index_manager_unified.py`  
- Detects stale or corrupted FAISS indices  
- Rebuilds index in batch, with backups  
- Monitors dimensions and health metrics  

### `utils/llm_service_enhanced.py`  
- **`LLMService`**:  
  - Wraps underlying LLM provider (OpenAI, local)  
  - Retry logic (`with_advanced_retry`) for API throttles  
  - Error tracking (`with_error_tracking`) for diagnostics  
  - Streaming callbacks for UI updates  

### `utils/unified_retrieval_system.py`  
- Orchestrates query ➔ BM25 + FAISS ➔ context cleaning (`clean_context`, `extract_hyperlinks`) ➔ LLM invocation ➔ result post-processing  

### `workapp3.py`  
- Parses CLI flags (`--rebuild-index`, `--dry-run`, `--verbose`)  
- Instantiates configs, processors, index manager, LLM service, retrieval system  
- Defines Streamlit layout: sidebar for settings, chat panel, progress bars, error display  

---

## 📈 Logging & Metrics

- **Query metrics** (latency, token counts) logged to `logs/query_metrics.log`  
- **Error stack traces** captured in `logs/workapp_errors.log`  
- Built-in decorators surface errors in UI and persist them for offline analysis  

---

## 🛠️ Development & Contribution

1. Follow the **Errors_Roadmap.md** to track and implement outstanding fixes.  
2. Write atomic patches—no stray files.  
3. Update dataclass defaults in `config_unified.py` when adding new settings.  
4. Maintain strict test coverage around index rebuild and retrieval logic.  
5. Use `with_error_tracking` for all new decorators to ensure visibility.  

---

## 📜 License

MIT License – see [LICENSE](./LICENSE) for details (or adapt as needed).

---

> This README is structured for an LLM: each section maps directly to code modules and data flows. Refer to line-numbers and docstrings in each `.py` within `utils/` for deeper details.
