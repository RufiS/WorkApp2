# Workflow: Post‑Refactor Smoke Test

Trigger with `/05-smoke-test.md`.

---

## 1 · Run Unit Tests

```bash
pytest -q
```

---

## 2 · Launch Streamlit UI

```bash
python -m streamlit run workapp3.py --headless --server.port 8501
```

---

## 3 · Health Check

```bash
curl -s http://localhost:8501/_stcore/health
```

---

## 4 · Manual Query Confirmation

```xml
<ask_followup_question>
  <question>Please enter a test question to verify answers are correct.</question>
</ask_followup_question>
```

---

## 5 · Commit Results

```bash
git add -A && git commit -m "Smoke test passed post‑refactor"
```
