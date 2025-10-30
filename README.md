# Academic Search Engine

An offline-first academic search engine that crawls a curated list of scholarly, educational, and governmental sites, ranks results with TF-IDF plus PageRank, and (optionally) summarises answers with an LLM including source citations.

## Features
- Curated allowlist covering 100+ vetted academic, medical, statistical, and governmental domains (`trusted_sources.py`).
- Respectful crawler with robots.txt compliance, MIME filtering, per-domain quotas, and configurable crawl depth/page limits.
- Tokenisation, stop-word filtering, TF-IDF weighting (with title boosts), and cosine similarity scoring layered with PageRank.
- CLI search interface (`Searching.py`) and AI-augmented assistant (`ai_search.py`) that produces citation-linked summaries via OpenAI or a local Ollama model, with extractive fallback when neural models are unavailable.
- Easily extensible source catalogue and summarisation workflow for custom research needs.

## Requirements
This project targets Python 3.10+.

Install dependencies (ensure `pip` or your environment's package manager is available):

```bash
python -m ensurepip --upgrade            # Only if pip is unavailable
python -m pip install -r requirements.txt
```

> **Notes**
> - Network access is required for crawling, OpenAI calls, and any remote Ollama hosts.
> - OpenAI integration expects `OPENAI_API_KEY` unless you provide `--api-key` on the command line.
> - Ollama support requires a running Ollama daemon (local or remote) with the desired model pulled, e.g. `ollama pull qwen3:8b`.

## Usage
### 1. Baseline crawl + search
```bash
python Searching.py --max-pages 150 --max-depth 2 --delay 1.0 --results 10
```

The script will crawl the trusted domains, build the index, and drop into an interactive prompt. Press enter on an empty line to exit.

### 2. AI-assisted research bot
```bash
# Optional: enable OpenAI
export OPENAI_API_KEY=sk-...

# Optional: ensure Ollama is serving and the model is present
ollama serve            # in another terminal if not already running
ollama pull qwen3:8b

python ai_search.py --query "climate change impact on coral reefs" --results 8
```

Without `--query`, the command enters an interactive loop (`Ask>` prompt). The output includes:
1. An LLM-generated summary with bracketed citations `[n]` (OpenAI if configured, otherwise Ollama, otherwise extractive).  
2. A source list matching the citation numbers.  
3. The underlying ranked documents.  

If neither OpenAI nor Ollama are available, the tool falls back to extractive summaries that still reference the retrieved documents.

## Customisation
- **Trusted sources**: Edit `trusted_sources.py` to add or remove categories, seed URLs, or domain quotas. The crawler automatically derives the allowlist from this module.
- **Crawler tuning**: Adjust limits via CLI flags (`--max-pages`, `--max-depth`, `--delay`) or by editing the defaults in `Crawler.py`.
- **Ranking**: Extend `Indexer.py` to add custom term weighting or BM25 logic; swap out PageRank damping/iterations in `Pagerank.py` as needed.
- **Summarisation**: Update `llm_agent.py` to target a different model endpoint or prompt style, or replace with your own agent implementation.

## Troubleshooting
- **`ModuleNotFoundError: No module named 'requests'`**: Install dependencies with `python -m pip install -r requirements.txt`. If `pip` is missing, bootstrap it via `python -m ensurepip --upgrade` first.
- **`openai` import errors**: Ensure the latest OpenAI Python SDK (`pip install -U openai`) and provide an API key. Without it, the search still works, but summaries fall back to extractive mode.
- **Ollama errors**: Confirm the Ollama daemon is running (`ollama serve`), the target model is pulled, and the `OLLAMA_HOST` environment variable (if used) points to a reachable endpoint.
- **Long crawl times**: Lower `--max-pages` and `--max-depth`, or increase `--delay` to remain friendly to remote servers.

Happy researching! Feel free to adapt the pipeline to your own academic or enterprise knowledge bases.
