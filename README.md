# 🔬 RigorousRAG
**The Agentic Research Orchestrator for High-Precision Academic Synthesis.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Framework: Agentic RAG](https://img.shields.io/badge/Framework-Agentic_RAG-purple.svg)](#)
[![Code Quality: Audited](https://img.shields.io/badge/Code_Quality-Audited-brightgreen.svg)](#)

RigorousRAG is a professional-grade research assistant designed for scientists and academic researchers. It layers a sophisticated multi-tool reasoning agent on top of a hybrid "Internal Index + Web + RAG" architecture, focusing on **groundedness, traceability, and rigorous evidence synthesis.**

---

## 🎯 Code Quality Status

> **Last Audit:** February 6, 2026  
> **Status:** ✅ **Production Ready**  
> **Bugs Fixed:** 3 (1 Critical, 1 Medium, 1 Low)  
> **Files Audited:** 49 Python files  
> 
> See [AUDIT_SUMMARY.md](AUDIT_SUMMARY.md) for complete audit details.

---

## 🚀 Key Features

### 🧠 Agentic Orchestration
*   **Multi-Tool Reasoning**: Automatically chooses between Internal Indices, Web Search, Handbook Policies, and Uploaded Documents. Fully implemented via OpenAI Function Calling.
*   **Citation Auditor**: Cross-verifies every inline citation `[n]` against raw source snippets to prevent hallucinations.
*   **Scientific Debate**: Adversarial 3-agent chain (Advocate, Skeptic, Judge) to evaluate claims.
*   **Interactive CLI**: A purpose-built terminal interface (`search_agent_cli.py`) for complex research dialogues.

### 📑 Advanced RAG & Ingestion
*   **Semantic Chunking**: Intelligent document splitting that preserves technical context and section integrity.
*   **Multimodal Ingestion**: Native support for PDF, Word, and Markdown/Text.
*   **Hierarchical Retrieval**: Parent/Child indexing for deep contextual awareness during search (Goal 20).
*   **HyDe (Hypothetical Document Embeddings)**: Generates hypothetical answers to bridge semantic gaps during vector search.
*   **Smart Summarization**: LLM-powered 2-sentence technical summaries generated during ingestion.

### 🛡️ Scientific Integrity Suite
*   **Comparison Matrices**: Generates dynamic Markdown tables comparing specific metrics across papers via targeted RAG extraction.
*   **Conflict Detector**: LLM-powered contradiction analysis across source literature.
*   **Visual Entailment**: Logic structure for cross-linking claims with paper figures (Vision-LLM compatible).
*   **Limitation Extractor**: Deep extraction of structural weaknesses, disclaimers, and future work.
*   **BibTeX Export**: One-click generation of bibliography entries for LaTeX.

---

## 🛠️ Architecture

RigorousRAG is a production-grade system:

*   **Engine**: Hybrid TF-IDF + PageRank for offline academic search.
*   **Vector DB**: ChromaDB for persistent document embeddings.
*   **LLM Core**: Multi-agent orchestration using OpenAI GPT-4o.
*   **API Layer**: FastAPI for exposing agentic capabilities.
*   **Search**: Integrated with Serper.dev for live web grounding.

---

## 📖 Quick Start

### 1. Installation
```powershell
git clone https://github.com/Anurag9000/RigorousRAG
cd RigorousRAG
pip install -r requirements.txt
```

### 2. Basic Configuration
```powershell
$env:OPENAI_API_KEY="sk-..." # Required for agentic reasoning
```

### 3. Usage Examples
*   **Interactive Chat**: `python search_agent_cli.py`
*   **Batch Ingestion**: `python ingest_docs.py ./my_papers/*.pdf`
*   **Start Backend**: `python server.py`
*   **Run via Docker**: `docker-compose up --build`

---

## 📂 Project Structure
*   `tools/`: Modular tool implementations (RAG, BibTeX, Integrity, etc.).
*   `search_agent.py`: The core reasoning and orchestration logic.
*   `ingest_docs.py`: High-performance ingestion CLI.
*   `server.py`: REST API exposing the research suite.

---

## ⚖️ Ethics & Privacy
*   **Redaction Layer**: Automatically redacts emails and phone numbers from technical documents (Goal 16).
*   **Owner Scoping**: Metadata-based ownership for multi-tenant environments.

Happy researching!

---
*Created by [Anurag9000](https://github.com/Anurag9000)*
