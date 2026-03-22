# RigorousRAG — Dream Spec vs. Reality: Gap Analysis

> Reconstructed from: README feature lists (all commit versions), inline `Goal N` code annotations (Goals 13, 16, 19, 20), commit messages, `AUDIT_SUMMARY.md`, and full codebase review.

---

## Legend

| Status | Meaning |
|--------|---------|
| ✅ **Fully Done** | Implemented, real logic, no stubs |
| 🟡 **Partial** | Scaffolding/structure exists, core logic incomplete |
| ❌ **Stub / Missing** | Not implemented — placeholder, commented-out, or entirely absent |

---

## Layer 1 — Classic Academic Search Engine

| Feature | Status | Notes |
|---------|--------|-------|
| BFS crawler over trusted domains | ✅ **Fully Done** | `Crawler.py` — full robots.txt, domain allowlist, depth/quota limits |
| Curated domain allowlist (100+ domains, 10 categories) | ✅ **Fully Done** | `trusted_sources.py` — 10 source categories, `www.` deduplication |
| robots.txt compliance with per-domain cache | ✅ **Fully Done** | `_robots_cache` in `AcademicCrawler`, fail-open on error |
| Resumable crawl (saves/restores frontier + visited) | ✅ **Fully Done** | `storage.py` → `CrawlState` with full JSON persistence |
| TF-IDF inverted index with title boost | ✅ **Fully Done** | `Indexer.py` — +2 title token count boost, log-normalized TF |
| Smoothed IDF (`log((1+N)/(1+df))+1`) | ✅ **Fully Done** | Prevents zero IDF, floor of 1 |
| PageRank with dangling node handling | ✅ **Fully Done** | `Pagerank.py` — 20-iter power method, sink_distribution |
| Combined ranking: 85% cosine + 15% PageRank | ✅ **Fully Done** | `Searching.py::search()` |
| Persistent storage for index + PageRank | ✅ **Fully Done** | `storage.py` — JSON files in `data/` |
| Interactive CLI search loop | ✅ **Fully Done** | `Searching.py::interactive_loop()` |

---

## Layer 2 — Agentic RAG System

### Core Agent & Orchestration

| Feature | Status | Notes |
|---------|--------|-------|
| LLM agentic loop with OpenAI function calling | ✅ **Fully Done** | `search_agent.py::SearchAgent.run()` — up to 10 turns |
| Multi-tool dispatch | ✅ **Fully Done** | Handles 12 tools via `if/elif` chain |
| Structured `AgentAnswer` output (Pydantic) | ✅ **Fully Done** | `tools/models.py` — `answer + citations[]` |
| `Citation` model with label, title, url, source_type, snippet | ✅ **Fully Done** | `tools/models.py` |
| Ollama local LLM support (CLI + server) | ✅ **Fully Done** | `--local` / `--demo` flags, `api_key="ollama"`, OpenAI-compat endpoint |
| OpenAI + Ollama + Extractive fallback chain | ✅ **Fully Done** | `llm_agent.py` — 3-tier fallback |
| JSON response stripping (markdown code fences) | ✅ **Fully Done** | `search_agent.py` lines 137–146 |
| Per-run usage metrics logging (JSONL) | ✅ **Fully Done** | `tools/logger.py` → `usage_metrics.jsonl` |
| Citation audit / anti-hallucination post-processor | 🟡 **Partial** | `tools/verification.py::audit_hallucination` — checks `[n]` markers map to citation objects, but **keyword/semantic matching is a `pass` stub** |
| Parallel tool execution | ❌ **Missing** | System prompt says "parallel", but `_handle_tool_call` is a sequential loop — no `asyncio.gather` or threading |

---

### RAG & Vector Search (Goals 19, 20)

| Feature | Status | Notes |
|---------|--------|-------|
| ChromaDB persistent vector store | ✅ **Fully Done** | `tools/rag.py` — `PersistentClient`, `all-MiniLM-L6-v2` |
| Hierarchical parent-child chunking **(Goal 20)** | ✅ **Fully Done** | Child ~1k chars indexed; parent text (~3k) stored in child metadata |
| Parent context expansion on retrieval **(Goal 20)** | ✅ **Fully Done** | `rag_tool.py` — returns `parent_text` from metadata as snippet |
| HyDE (Hypothetical Document Embeddings) **(Goal 19)** | ✅ **Fully Done** | `rag_tool.py` + `rag.py::generate_hyde_query()` — GPT-4o-mini generates hypothetical para |
| Multi-query expansion | 🟡 **Partial** | `rag.py::generate_expanded_queries()` exists, but `rag_tool.py` calls with `use_multi_query=False` — **never actually activated** |
| Metadata filtering in ChromaDB queries | ✅ **Fully Done** | `where` param passed through; `compare_papers` uses `where={"doc_id": ...}` |
| Chunk deduplication + sort by score | ✅ **Fully Done** | `rag.py::query()` — dedup by text, sort by distance |

---

### Document Ingestion

| Feature | Status | Notes |
|---------|--------|-------|
| PDF ingestion (PyMuPDF) | ✅ **Fully Done** | `tools/ingestion.py::_ingest_pdf()` |
| Word (.docx) ingestion with heading detection | ✅ **Fully Done** | `tools/ingestion.py::_ingest_docx()` — heading style detection |
| Plain text / Markdown ingestion | ✅ **Fully Done** | `tools/ingestion.py::_ingest_text()` — UTF-8 with Latin-1 fallback |
| PII redaction — emails, phones **(Goal 16.2)** | ✅ **Fully Done** | `redact_text()` — 3 regex patterns |
| PII redaction — US addresses | 🟡 **Partial** | Pattern simplistic; multi-line or non-US addresses are missed |
| Owner scoping / metadata tagging **(Goal 16.1)** | ✅ **Fully Done** | `owner_id` stamped in `result.document.metadata` |
| Academic metadata extraction (DOI, year, title) | 🟡 **Partial** | Heuristic regex on first 2000 chars — no semantic/PDF-layout-aware extraction |
| Semantic chunking (paragraph → sentence fallback) | ✅ **Fully Done** | `_chunk_text_semantically()` |
| Pre-ingestion LLM 2-sentence summary **(Goal 19)** | 🟡 **Partial** | In CLI (`ingest_docs.py`) only — **`server.py::process_ingestion()` skips it entirely** |
| Section detection in PDFs (heading/font-based) | ❌ **Missing** | Creates sections per-page only; comment says "placeholder — in real impl we'd look for font sizes" |
| Multi-tenant access control (retrieval filtering) | ❌ **Missing** | `owner_id` is set in metadata but **never used as a filter during retrieval** |

---

### Scientific Integrity Suite

| Feature | Status | Notes |
|---------|--------|-------|
| 3-agent Scientific Debate (Advocate→Skeptic→Judge) | ✅ **Fully Done** | `tools/integrity.py::run_scientific_debate()` — 3 sequential GPT-4o calls |
| Conflict Detector | ✅ **Fully Done** | `tools/integrity.py::detect_conflicts()` — GPT-4o with JSON output |
| Limitation Extractor | ✅ **Fully Done** | `tools/integrity.py::extract_limitations()` — GPT-4o-mini |
| Cross-paper Comparison (narrative) | ✅ **Fully Done** | `compare_papers()` — fetches RAG chunks per doc, GPT-4o synthesis |
| Comparison Matrix (Markdown table) | ✅ **Fully Done** | `generate_comparison_matrix()` — per-cell RAG lookup + GPT-4o-mini extraction |
| Visual Entailment Check **(Goal 13.4)** | ❌ **Stub** | `check_visual_entailment()` — image bytes injection **commented out**; always text-hallucinates "confirms the claim" |
| Protocol Extraction (wet-lab methods parser) | ❌ **Stub** | `extract_protocol()` — returns one hardcoded step: `"Step 1 extracted from text. Automated extraction pending."` |
| BibTeX Export | ✅ **Fully Done** | `tools/bib.py::export_to_bibtex()` — `@article` format |

---

### External Web Search

| Feature | Status | Notes |
|---------|--------|-------|
| Live web search via Serper.dev | ✅ **Fully Done** | `tools/web_search.py` — POST to `google.serper.dev/search` |
| Domain-allowlist filtering for web results | ✅ **Fully Done** | `allowed_domains` substring-match filter |
| Graceful degradation when `SERPER_API_KEY` missing | ✅ **Fully Done** | Returns error `Citation` instead of crashing |

---

### API Server

| Feature | Status | Notes |
|---------|--------|-------|
| FastAPI `/query` endpoint | ✅ **Fully Done** | Returns `AgentAnswer` |
| FastAPI `/ingest` with async background task | ✅ **Fully Done** | `BackgroundTasks` used |
| Frontend served as static site | ✅ **Fully Done** | `StaticFiles(html=True)` at `/` |
| Per-request model override | ✅ **Fully Done** | `server.py::run_query()` sets `agent.model` |
| Job status polling endpoint | ❌ **Missing** | `job_id` is returned but there is **no `GET /status/{job_id}` endpoint** |
| Docker / Docker Compose | ✅ **Fully Done** | `Dockerfile` + `docker-compose.yml` exist |
| Authentication / API key protection | ❌ **Missing** | No auth middleware — any caller can query or ingest |

---

### Frontend

| Feature | Status | Notes |
|---------|--------|-------|
| Dark-theme enterprise chat UI | ✅ **Fully Done** | Custom CSS design system in `frontend/index.html` |
| Markdown rendering in responses | ✅ **Fully Done** | `marked.js` CDN — tables, code, etc. |
| Color-coded citations sidebar | ✅ **Fully Done** | CSS `[data-source]` selectors per source type |
| Drag-and-drop file upload | ✅ **Fully Done** | `dragover`/`drop` events + `FormData` POST to `/ingest` |
| Batch multi-file upload | ✅ **Fully Done** | Loops `files[i]` with per-file status |
| Quick-action prompt buttons | ✅ **Fully Done** | 6 buttons pre-filling the query input |
| Loading spinner | ✅ **Fully Done** | CSS animation + `sendBtn.disabled` guard |
| Session persistence (localStorage) | ❌ **Missing** | History is DOM-only — lost on refresh |
| Responsive / mobile layout | ❌ **Missing** | Hardcoded 380px sidebar, no CSS breakpoints |

---

## Summary Scorecard

| Category | ✅ Done | 🟡 Partial | ❌ Missing/Stub |
|----------|---------|-----------|----------------|
| Classic IR Search Engine | 10 | 0 | 0 |
| Core Agent & Orchestration | 8 | 1 | 1 |
| RAG & Vector Search | 6 | 1 | 0 |
| Document Ingestion | 6 | 3 | 2 |
| Scientific Integrity Suite | 5 | 0 | 2 |
| Web Search | 3 | 0 | 0 |
| API Server | 5 | 0 | 2 |
| Frontend | 7 | 0 | 2 |
| **TOTAL** | **50** | **5** | **9** |

> **~78% of the original dream spec is fully implemented. 8% is partially done. 14% is a stub or entirely missing.**

---

## Priority Gap List

### 🔴 High — Core Correctness is Broken

| # | Gap | File | Fix |
|---|-----|------|-----|
| 1 | **Visual Entailment never sees an actual image** | `tools/integrity.py::check_visual_entailment` | Fetch figure bytes from ChromaDB doc, base64-encode, uncomment `image_url` message part |
| 2 | **Protocol Extraction returns a hardcoded stub** | `tools/integrity.py::extract_protocol` | Replace stub with a real GPT-4o call using JSON schema matching `Protocol` |
| 3 | **Multi-tenant isolation absent at query time** | `tools/rag_tool.py::search_uploaded_docs` | Add `where={"owner_id": current_user}` filter to all RAG `.query()` calls |

### 🟡 Medium — Missing UX / Feature Completeness

| # | Gap | File | Fix |
|---|-----|------|-----|
| 4 | **No job-status endpoint for async ingestion** | `server.py` | Add `job_registry: Dict[str, str]` + `GET /status/{job_id}` endpoint |
| 5 | **LLM summary skipped on web upload** | `server.py::process_ingestion` | Move LLM summary generation into `process_ingestion` (mirrors `ingest_docs.py`) |
| 6 | **Multi-query expansion is wired but disabled** | `tools/rag_tool.py` | Change `use_multi_query=False` → `True` (or make it a config param) |
| 7 | **Parallel tool execution never happens** | `search_agent.py::run` | Use `asyncio.gather` over `msg.tool_calls` list |

### 🟢 Low — Polish & Robustness

| # | Gap | Fix |
|---|-----|-----|
| 8 | Citation semantic verification is a `pass` | Add keyword-overlap check between answer sentence and citation snippet |
| 9 | `search_handbook` returns full file text | Chunk handbook, run through RAG for semantic retrieval |
| 10 | PDF section detection is page-only | Parse font-size/bold markers via PyMuPDF to detect real headings |
| 11 | Address redaction is US-only and simplistic | Extend regex patterns, or use a NER library (`spaCy`) |
| 12 | Frontend session is not persisted | Serialize `messages` array to `localStorage` on each append |
| 13 | No responsive layout | Add CSS media queries for sidebar collapse on narrow screens |
