"""AI-assisted academic search CLI."""

from __future__ import annotations

import argparse
import textwrap

from Searching import AcademicSearchEngine
from llm_agent import LLMAgent


def format_summary(summary: str) -> str:
    return "\n".join(textwrap.wrap(summary, width=100))


def run_query(
    engine: AcademicSearchEngine, agent: LLMAgent, query: str, limit: int
) -> None:
    hits = engine.search(query, limit=limit)
    if not hits:
        print("No results found.")
        return

    contexts = engine.gather_context(hits, max_chars=8000)
    summary = agent.summarise(query, hits, contexts)

    print("\n=== AI Summary ===")
    print(format_summary(summary.summary))
    print("\n=== Sources ===")
    for source in summary.sources:
        print(f"- {source}")
    print("\n=== Top Results ===")
    for hit in hits:
        snippet = hit.snippet[:200]
        if len(hit.snippet) > 200:
            snippet += "..."
        print(f"{hit.rank}. {hit.title} ({hit.score:.3f})")
        print(f"   {hit.url}")
        if snippet:
            print(f"   {snippet}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-assisted academic search and summarisation"
    )
    parser.add_argument("--query", type=str, help="Run a single query then exit.")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=200,
        help="Maximum number of pages to crawl (default: 200)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum crawl depth from each seed URL (default: 2)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between HTTP requests (default: 1.0)",
    )
    parser.add_argument(
        "--results",
        type=int,
        default=8,
        help="Number of search results to send to the LLM (default: 8)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model identifier (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Override OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="qwen3:8b",
        help="Ollama model to use when available (default: qwen3:8b).",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        help="Ollama host endpoint, e.g. http://localhost:11434 (defaults to OLLAMA_HOST).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = AcademicSearchEngine(
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        request_delay=args.delay,
    )

    print("Crawling trusted academic domains...")
    total_pages = engine.build()
    print(f"Crawl complete. Indexed {total_pages} pages.\n")

    agent = LLMAgent(
        model=args.model,
        api_key=args.api_key,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
    )

    if args.query:
        run_query(engine, agent, args.query, args.results)
        return

    print("Enter an empty line to exit.\n")
    while True:
        try:
            query = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        run_query(engine, agent, query, args.results)


if __name__ == "__main__":
    main()
