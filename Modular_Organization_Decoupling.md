# Modular Organization & Decoupling Tasks

## 1. Structure Code by Feature
- Group modules into `core/`, `llm/`, `retrieval/`, `ui/`, and `error_handling/` packages
- Move modules accordingly and update imports

## 2. Decouple Index Management from Document Processing
- Introduce an `IndexManager` class handling FAISS save/load and verification
- `DocumentProcessor` should focus on ingestion and call `IndexManager`

## 3. Isolate the QA Pipeline Logic
- Create `core/qa_pipeline.py` with a function `answer_query(query)` that orchestrates retrieval, LLM call, and formatting
- Streamlit UI (`workapp3.py`) should delegate to this

## 4. Inject Configurations Cleanly
- Replace global `config_unified` imports with explicit config injection (constructor parameters)
- Consider renaming `config_unified.py` to `config.py`
