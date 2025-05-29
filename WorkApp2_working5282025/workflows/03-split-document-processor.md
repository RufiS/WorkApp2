# Workflow: Split Document Processor

Trigger with `/03-split-document-processor.md`. Extract ingestion and index logic.

---

## 1 · Read Original File

```xml
<read_file><path>core/document_processor_unified.py</path></read_file>
```

---

## 2 · Create Ingestion Module

```xml
<create_file>
  <path>core/document_ingestion.py</path>
  <contents># Ingestion logic extracted from document_processor_unified.py</contents>
</create_file>
```

Copy the loader, chunking, and caching helpers via `edit_file`.

---

## 3 · Create Index Manager

```xml
<create_file>
  <path>core/index_manager.py</path>
  <contents># Embedding & FAISS operations</contents>
</create_file>
```

Copy embedding and FAISS functions.

---

## 4 · Add Thin Facade

```xml
<create_file>
  <path>core/document_processor.py</path>
  <contents>
"""Facade combining ingestion and index manager"""
from core.document_ingestion import DocumentIngestion
from core.index_manager import IndexManager
# TODO: wire up methods
  </contents>
</create_file>
```

---

## 5 · Update References & Tests

Search for `document_processor_unified` and replace imports. Run tests and commit.
