# Code Deduplication Tasks

## 1. Merge LLM Service modules
- Create a single `llm_service.py` that incorporates all enhancements from `llm_service_enhanced.py`
- Update any imports (e.g. in `workapp3.py`) to use the unified module
- Delete the old `llm_service_enhanced.py`

## 2. Unify Document Processing & File Handling
- Use `DocumentProcessor` as the single source of truth for document loading and chunking
- Merge unique logic from `file_processing.py` into `DocumentProcessor` methods
- Use the `ChunkCacheEntry` defined in `DocumentProcessor` and remove the duplicate in `caching.py`
- Remove redundant `file_processing.py` and `utils/caching.py` modules
- Update all code to call the unified `DocumentProcessor`

## 3. Consolidate Retrieval Logic (Hybrid Search)
- Merge `EnhancedRetrieval` and `UnifiedRetrievalSystem` into a single retrieval module/class
- Prefer `UnifiedRetrievalSystem` as the base and incorporate features from `EnhancedRetrieval`
- Update references in code to use the unified class
- Remove the outdated module

## 4. Merge Prompt Generators
- Unify `formatting_prompt.py` and `formatting_prompt_enhanced.py`
- Take the base `generate_formatting_prompt` and extend with improvements from the enhanced version
- Include `check_formatting_quality` if useful
- Update `LLMService` to import from the unified prompt module
- Remove `formatting_prompt_enhanced.py`

## 5. Use One Set of Error-Handling Decorators
- Adopt enhanced decorators as the default
- Refactor code to use these consistently
- Remove references to `decorators.py`

## 6. Unify UI Helper Functions
- Merge relevant UI functions into a single module
- Identify overlapping functionality and upgrade/migrate as needed
- Remove the obsolete module
