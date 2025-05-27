# Workflow: Modularize WorkApp2

Trigger with `/02-modularize.md`. Moves files into feature‑based packages.

---

## 1 · Create Packages

```bash
mkdir -p core llm retrieval ui error_handling
```

---

## 2 · Move Representative Files

```bash
git mv document_processor_unified.py core/
git mv llm_service.py llm/
git mv formatting_prompt.py llm/
git mv unified_retrieval_system.py retrieval/
git mv ui/components.py ui/
git mv enhanced_decorators.py error_handling/
```

---

## 3 · Fix Imports

```xml
<search_files>
  <path>.</path>
  <regex>from +(document_processor_unified|unified_retrieval_system|llm_service)</regex>
  <file_pattern>*.py</file_pattern>
</search_files>
```

Edit each hit to use the new package paths.

---

## 4 · Smoke Test

```bash
pytest -q
python -m streamlit run workapp3.py --headless --server.port 8501
```

If all good, commit:

```bash
git add -A && git commit -m "Modularize codebase"
```
