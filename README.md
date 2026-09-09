# ForgeMind AI

Multi-agent RAG assistant built with FastAPI, LangGraph, and ChromaDB.

---

## Quality Gates & Testing

ForgeMind uses automated CI quality gates to verify security, reliability, API contracts, RAG evaluation logic, and maintenance operations.

### Running the Deterministic Test Suite Locally

To run the exact same test suite executed in CI:

```bash
# Using uv (canonical)
PYTHONPATH=backend:. uv run --project backend python -m unittest discover -s tests -v

# Or using the local virtual environment directly
PYTHONPATH=backend:. backend/.venv/bin/python -m unittest discover -s tests -v
```

All 115 unit and integration tests run offline without requiring external network connectivity, LLM API keys, or cloud services.

### Quality Gates Verified by CI

| Gate | Scope | Key Assertions |
| :--- | :--- | :--- |
| **Security** | Path traversal & validation | Rejects `..`, `/`, `\`, null bytes, non-PDFs, oversized uploads |
| **Document Lifecycle** | Consistency & 2-phase deletion | ChromaDB vectors deleted before physical files; no stale vectors |
| **RAG Relevance** | Retrieval thresholding | Score cutoff (`0.60`) filters non-relevant context before LLM |
| **Citations** | Attribution contract | In-text citations `[Source: filename \| Page: X]` formatting |
| **Observability** | Telemetry privacy | Trace metadata, request correlation, zero secret leaks |
| **Reliability** | Error & retry semantics | Bounded retries, fail-fast on auth/4xx, sanitized error envelopes |
| **API Boundary** | HTTP contract hardening | `X-Request-ID` tracing, Pydantic egress, 200/503 health probes |
| **Maintenance Agent** | System diagnostics & repair | Missing index detection (Case A), orphan vector purge (Case B) |

---

### Running Live RAG Evaluation (Manual / Optional)

The 25-question end-to-end benchmark evaluates live retrieval precision, recall, and LLM generation. Because it makes external LLM calls and consumes API tokens, it is kept separate from automated PR gates:

```bash
PYTHONPATH=backend:. python tests/evaluation/run_eval.py --dataset tests/evaluation/dataset.json
```
