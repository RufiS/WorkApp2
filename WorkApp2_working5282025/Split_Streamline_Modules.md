# Split and Streamline Large Modules Tasks

## 1. Split `document_processor_unified.py`
- Create `document_ingestion.py` for file loading, text extraction, chunking, cache
- Create `index_manager.py` for FAISS embedding, add/search, save/load
- Keep a thin `document_processor.py` as facade if desired

## 2. Trim Down `workapp3.py`
- Move UI sections into `ui/layout.py`
- Strip logic and rely on unified components

## 3. Right-Size Other Modules
- Keep each file <500 lines
- Move examples/documentation to `docs/`
