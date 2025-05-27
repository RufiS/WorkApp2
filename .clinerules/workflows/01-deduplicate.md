# Workflow: Deduplicate & Merge Redundant Modules in WorkApp2

Trigger with `/01-deduplicate.md`.  
This workflow tackles **all known duplicate or overlapping code** and then runs a generic duplicate‑finder pass.

> **Context limit**: Each step is scoped so an LLM with 128 k tokens can complete it comfortably.

---

## A · High‑Priority Target Groups (one‑time merges)

Follow these sub‑workflows first. Each one creates a *single* authoritative module and deletes the obsolete copy/ies.

| # | Target | Actions |
|---|--------|---------|
| **1** | **LLM Service**<br/>`llm_service.py` + `llm_service_enhanced.py` | • Create unified **`llm_service.py`** with all enhancements.<br/>• Update imports (e.g. `workapp3.py`).<br/>• Delete `llm_service_enhanced.py`. |
| **2** | **Document Processing & File Handling**<br/>`DocumentProcessor` vs `file_processing.py` + `utils/caching.py` | • Keep `DocumentProcessor` as single source of truth.<br/>• Migrate unique logic (file‑type handling, post‑processing) from `file_processing.py`.<br/>• Use `ChunkCacheEntry` from `DocumentProcessor`; remove duplicate in `caching.py`.<br/>• Delete `file_processing.py` & `utils/caching.py`.<br/>• Update all imports to use `DocumentProcessor`. |
| **3** | **Retrieval Logic (Hybrid Search)**<br/>`EnhancedRetrieval` vs `UnifiedRetrievalSystem` | • Use `UnifiedRetrievalSystem` as base.<br/>• Integrate keyword‑fallback, MMR, or other extras from `EnhancedRetrieval`.<br/>• Update imports; delete `enhanced_retrieval.py`. |
| **4** | **Prompt Generators**<br/>`formatting_prompt.py` vs `formatting_prompt_enhanced.py` | • Merge into one `formatting_prompt.py`.<br/>• Extend `generate_formatting_prompt` with enhanced features.<br/>• Include `check_formatting_quality` helper if valuable.<br/>• Update `LLMService` import.<br/>• Delete `formatting_prompt_enhanced.py`. |
| **5** | **Error‑Handling Decorators**<br/>`decorators.py` vs `enhanced_decorators.py` | • Adopt enhanced decorators (`enhanced_decorators.py`).<br/>• Replace all `@with_retry`, `@with_error_handling`, etc. with their enhanced equivalents.<br/>• Delete `decorators.py`. |
| **6** | **UI Helper Functions**<br/>`ui/components.py` vs `ui/enhanced_components.py` | • Merge overlapping functions; upgrade originals with richer formatting.<br/>• Add any new utilities (e.g., `display_system_status`).<br/>• Delete `ui/enhanced_components.py`. |

**For each target group** use this pattern:

```xml
<read_file path="path/to/old_file.py"/>
<read_file path="path/to/new_file.py"/>
<edit_file path="path/to/new_file.py" instruction="Integrate improvements from old_file.py, resolve conflicts, add tests if needed."/>
<ask_followup_question question="Run unit tests now? (yes/no)"/>
```

After the merge, delete the redundant file **only after tests pass.**

---

## B · Generic Duplicate Function Scan

Once the six priority merges are complete, run an automated search to catch any remaining unnoticed duplicates.

### 1 · Search

```xml
<search_files>
  <path>.</path>
  <regex>def +(validate_json_output|generate_formatting_prompt|hybrid_search|[a-zA-Z0-9_]+)</regex>
  <file_pattern>*.py</file_pattern>
</search_files>
```

Group hits by identical **function names**.

### 2 · Choose Canonical Implementations

```xml
<ask_followup_question>
  <question>You found duplicates of **{function_name}** in:<br/>{file_list}<br/><br/>Which file should be canonical?</question>
  <options>{file_list}</options>
</ask_followup_question>
```

### 3 · Merge & Remove

```xml
<read_file path="{redundant_file}"/>
<read_file path="{canonical_file}"/>
<edit_file path="{canonical_file}" instruction="Integrate missing behavior from redundant_file, ensure no regression."/>
```

Delete `{redundant_file}` once tests are green.

### 4 · Run Tests & Commit

```bash
pytest -q
git add -A && git commit -m "Remove duplicates of {function_name}"
```

Loop until no duplicates remain.

---

## C · Final Sanity Check

1. **Run full test suite**  
   ```bash
   pytest -q
   ```
2. **Start Streamlit in headless mode** and verify health endpoint.  
   ```bash
   python -m streamlit run workapp3.py --headless --server.port 8501
   curl -s http://localhost:8501/_stcore/health
   ```
3. **Commit**  
   ```bash
   git add -A && git commit -m "Deduplication phase complete"
   ```