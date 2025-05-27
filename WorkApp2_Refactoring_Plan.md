# Refactoring Plan for WorkApp2 Codebase

## 1. Code Deduplication (Eliminate Redundant Logic)

- **Merge LLM Service modules (`llm_service.py` & `llm_service_enhanced.py`)** – *These two files implement the same `LLMService` with overlapping code.* For example, both define `validate_json_output` (duplicated JSON validation logic). **Why:** Removing one copy prevents divergence and eases maintenance. **How:** Create a single `llm_service.py` that incorporates all enhancements from the "enhanced" version (e.g. context truncation before prompts, `check_formatting_quality` usage, stricter input validation) into the base class. Update any imports (e.g. in `workapp3.py`) to use the unified module. Delete the old `llm_service_enhanced.py` to avoid confusion.

- **Unify Document Processing & File Handling** – *Consolidate the document ingestion code that currently exists in multiple places.* The class `DocumentProcessor` in **`document_processor_unified.py`** duplicates functionality from **`file_processing.py`** and **`caching.py`** (e.g. both define `ChunkCacheEntry`, chunking and hashing functions). **Why:** Keeping one implementation of file loading, chunk splitting, and caching simplifies debugging and ensures consistent behavior. **How:** Use the `DocumentProcessor` class as the single source of truth for document loading and chunking. Merge any unique logic from `file_processing.py` (such as file type handling or chunk post-processing) into `DocumentProcessor` methods (`_get_file_loader`, `load_and_chunk_document`, `_handle_small_chunks`, etc.). Likewise, use the `ChunkCacheEntry` defined in `DocumentProcessor` and remove the duplicate definition in `caching.py`. After integrating, remove the now-redundant `file_processing.py` and `utils/caching.py` modules. Ensure all code (e.g. Streamlit file uploads in `workapp3.py`) calls the unified `DocumentProcessor` instead of any standalone functions.

- **Consolidate Retrieval Logic (Hybrid Search)** – *There are two retrieval systems:* `EnhancedRetrieval` (in **`enhanced_retrieval.py`**) and `UnifiedRetrievalSystem` (in **`unified_retrieval_system.py`**). They overlap in purpose – both perform combined vector and keyword search – but maintain separate implementations (e.g. both handle BM25 vs. FAISS searches and re-ranking). **Why:** Maintaining two versions of retrieval code risks inconsistency (bugs fixed in one not in the other) and adds complexity. **How:** Merge these into a single retrieval module/class. Prefer the `UnifiedRetrievalSystem` class as the base (since the Streamlit app already uses it) and incorporate any features from `EnhancedRetrieval` not present in the unified version. For example, if `EnhancedRetrieval.hybrid_search` or `mmr_reranking` logic is superior, integrate that into `UnifiedRetrievalSystem.retrieve` or a new helper method `_rerank_results`. Likewise, include any "keyword fallback" search functionality so that the unified class covers all scenarios. Once unified, update references in code to use the one class and remove the outdated module. This results in one clear **Retrieval Engine** for the app.

- **Merge Prompt Generators (`formatting_prompt.py` & `formatting_prompt_enhanced.py`)** – *Both files define `generate_formatting_prompt` with minor differences, and the enhanced version adds `check_formatting_quality`.* **Why:** We should have one consistent prompt format for answer formatting to avoid confusion. **How:** Unify into a single `formatting_prompt.py` module. Take the base `generate_formatting_prompt` and extend it with the improvements from the enhanced version (for instance, regex-based post-processing or better handling of bullet points if present). If `check_formatting_quality` is a useful utility for evaluating the formatted answer, include it in this module (or integrate its logic directly into the formatting workflow if appropriate). Update `LLMService` to import from the unified prompt module. Remove `formatting_prompt_enhanced.py` after confirming the new unified prompt works as expected.

- **Use One Set of Error-Handling Decorators** – *The `utils/error_handling/` package has duplicate decorator logic:* `decorators.py` (original) vs `enhanced_decorators.py` (improved). For example, `with_retry` vs `with_advanced_retry`, and differing error tracking wrappers. **Why:** Standardizing on one robust set of decorators simplifies error handling and avoids confusion about which to use. **How:** Adopt the enhanced decorators as the default (they appear to provide advanced retry logic, timing, and error tracking). Refactor code to use these consistently: e.g. replace `@with_retry` with `@with_advanced_retry` and `@with_error_handling` with `@with_error_tracking` where applicable. If the older decorators have any functionality not in the new ones (such as specific exception filtering or recovery hooks), extend the enhanced versions to cover those needs. Once all usages are migrated, remove references to `decorators.py` and consider deleting it. This ensures all retry/error-handling logic is defined in one module (`enhanced_decorators.py`) with a consistent implementation.

- **Unify UI Helper Functions** – *The UI layer has duplicated component functions:* **`ui/components.py`** vs **`ui/enhanced_components.py`**. **Why:** Having two sets of Streamlit UI functions complicates the UI code and may lead to inconsistent user experience if some features use old components. **How:** Merge the relevant UI functions into a single module (e.g. keep `components.py` and add enhanced features to it). Identify any overlapping functionality, upgrade originals, migrate new ones, and remove the obsolete module.

## 2. Improve Modular Organization & Decoupling

- **Structure Code by Feature** – Group into `core/`, `llm/`, `retrieval/`, `ui/`, and `error_handling/` packages. Move modules accordingly and update imports.

- **Decouple Index Management from Document Processing** – Introduce an `IndexManager` class handling FAISS save/load and verification. `DocumentProcessor` should focus on ingestion and call `IndexManager`.

- **Isolate the QA Pipeline Logic** – Create `core/qa_pipeline.py` with a function `answer_query(query)` that orchestrates retrieval, LLM call, and formatting. Streamlit UI (`workapp3.py`) should delegate to this.

- **Inject Configurations Cleanly** – Replace global `config_unified` imports with explicit config injection (constructor parameters). Consider renaming `config_unified.py` to `config.py`.

## 3. Split and Streamline Large Modules

- **Split `document_processor_unified.py`**  
  1. **document_ingestion.py** – file loading, text extraction, chunking, cache.  
  2. **index_manager.py** – FAISS embedding, add/search, save/load.  
  3. Keep a thin `document_processor.py` as facade if desired.

- **Trim Down `workapp3.py`** – Move UI sections into `ui/layout.py`, strip logic, rely on unified components.

- **Right‑Size Other Modules** – Keep each file <500 lines; move examples/documentation to `docs/`.

## 4. Consistent Naming & Style

- **Drop “unified/enhanced” Prefixes** – Rename modules: `document_processor_unified.py` → `document_processor.py`, `unified_retrieval_system.py` → `retrieval_system.py`, etc.

- **Apply Naming Conventions** – snake_case for functions/vars, PascalCase for classes; align config keys.

- **Follow PEP8 & Auto‑format** – Run `black` (line length 100), fix linter warnings, ensure docstrings.

---

### Implementation Order

1. **Deduplicate code** (merge & delete duplicates)  
2. **Rename modules** to final names  
3. **Reorganize package structure**  
4. **Split oversized modules**  
5. **Inject configs & decouple pipeline**  
6. **Run formatter & linter**  
7. **Smoke‑test the app** (Streamlit) after each major stage.

> **Tip:** Commit after each step. Run unit tests (or at least a regression query) to confirm no behavior change before proceeding.

