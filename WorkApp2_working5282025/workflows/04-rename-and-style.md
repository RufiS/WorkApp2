# Workflow: Rename Modules & Apply Style

Trigger with `/04-rename-and-style.md`.

---

## 1 · Rename Files

```bash
git mv config_unified.py config.py
git mv retrieval/unified_retrieval_system.py retrieval/retrieval_system.py
# ...other renames
```

---

## 2 · Update Imports

```xml
<search_files>
  <path>.</path>
  <regex>_unified|_enhanced</regex>
  <file_pattern>*.py</file_pattern>
</search_files>
```

Edit hits accordingly.

---

## 3 · Run Formatter & Linter

```bash
black . --line-length 100
flake8 .
```

Fix any errors via `edit_file`.

---

## 4 · Commit

```bash
git add -A && git commit -m "Rename modules & format code"
```
