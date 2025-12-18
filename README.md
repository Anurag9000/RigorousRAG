# Academic Search Engine & Agentic RAG

An advanced, agentic academic search platform that layers a modern RAG-style “search + internal knowledge + web” agent on top of an offline-first academic search engine.

## 🚀 Key Features

### 1. Agentic Orchestration
- **Search Agent (`search_agent.py`)**: A multi-tool reasoning agent that chooses the best source (Internal Index, Handbook, Web, or Uploaded Docs) to answer complex queries.
- **Interactive CLI (`search_agent_cli.py`)**: A dedicated terminal interface for chatting with the research agent.
- **Service API (`server.py`)**: A FastAPI-based backend exposing the agent and document ingestion over HTTP.

### 2. Multimodal RAG & Ingestion
- **Document Ingestion (`ingest_docs.py`)**: Support for PDF (PyMuPDF), Word (`python-docx`), and Markdown/Text files.
- **Vector Database (`tools/rag.py`)**: Powered by ChromaDB and `all-MiniLM-L6-v2` embeddings for semantically aware retrieval of local documents.

### 3. Scientific Integrity Suite
- **Visual Entailment**: Check if a paper's claims are supported by its figures (via Vision-LLM stubs).
- **Protocol Extraction**: Automatically extract structured wet-lab protocols from methods sections.
- **Adversarial Debate**: Run "Advocate vs. Skeptic" reasoning loops to stress-test claims.

### 4. Legacy Academic Search
- **Curated Crawling**: Focused on 100+ vetted academic and governmental domains.
- **Hybrid Ranking**: Combines TF-IDF with PageRank.
- **Offline Indexing**: Persistent state saved to disk for ultra-fast local search.

## 🛠️ Requirements & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**:
   - Set your OpenAI API key: `export OPENAI_API_KEY=sk-...`
   - (Optional) Pull local models for Ollama: `ollama pull qwen3:8b`

## 📖 Usage

### Interactive Research Agent
```bash
python search_agent_cli.py
```

### Batch Ingest Documents
```bash
python ingest_docs.py ./papers/*.pdf
```

### Start API Server
```bash
python server.py
```
*API docs available at `http://localhost:8000/docs`*

### Run via Docker
```bash
docker-compose up --build
```

## 📂 Project Structure
- `tools/`: Modular tool definitions (search, rag, ingestion, integrity).
- `search_agent.py`: Core reasoning logic.
- `ingest_docs.py`: Document processing CLI.
- `server.py`: FastAPI implementation.
- `Searching.py`: Legacy search engine CLI.

## ⚖️ Safety & Configuration
- **Domain Allowlist**: Derivatives of `trusted_sources.py`.
- **Handbook**: Reads from `handbook.md` for internal policy grounding.
- **Agent Prompting**: Configured in `search_agent.py` to prioritize grounded citations `[n]`.

Happy researching! Feel free to adapt the pipeline to your own academic or enterprise knowledge bases.

