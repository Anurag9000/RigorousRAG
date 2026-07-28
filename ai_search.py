"""AI-assisted CLI over the persisted classic academic index."""

from __future__ import annotations

import argparse

from Searching import AcademicSearchEngine
from llm_agent import LLMAgent

_MAX_QUERY_CHARS = 2000


def format_summary(summary: str) -> str:
    """Preserve Markdown and line structure produced by the summariser."""

    return (summary or "").strip()


def _validated_query(value: str) -> str:
    query = (value or "").strip()
    if not query:
        raise ValueError("A research query is required.")
    if len(query) > _MAX_QUERY_CHARS:
        raise ValueError("Research queries may contain at most 2,000 characters.")
    return query


def run_query(
    engine: AcademicSearchEngine,
    agent: LLMAgent,
    query: str,
    limit: int,
) -> None:
    query = _validated_query(query)
    hits = engine.search(query, limit=max(1, min(limit, 20)))
    if not hits:
        print("No results found.")
        return
    contexts = engine.gather_context(hits, max_chars=24_000)
    summary = agent.summarise(query, hits, contexts)
    print("\n=== AI Summary ===")
    print(format_summary(summary.summary))
    if summary.warning:
        print(f"\nWarning: {summary.warning}")
    print("\n=== Sources ===")
    for source in summary.sources:
        print(f"- {source}")
    print("\n=== Top Results ===")
    for hit in hits:
        print(f"{hit.rank}. {hit.title} ({hit.score:.3f})")
        print(f"   {hit.url}")
        if hit.snippet:
            print(f"   {hit.snippet[:300]}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-assisted academic search and evidence summarisation"
    )
    parser.add_argument("--query", help="Run one query and exit.")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--results", type=int, default=8)
    parser.add_argument("--storage-dir", default="data")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--ollama-model", default="qwen3:8b")
    parser.add_argument("--ollama-host")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = AcademicSearchEngine(
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        request_delay=args.delay,
        storage_dir=args.storage_dir,
    )
    if args.rebuild or not engine.ready:
        print("Crawling and rebuilding the academic index...")
        total_pages = engine.build()
        print(f"Indexed {total_pages} pages.\n")
    else:
        print(f"Loaded {len(engine.index.documents)} indexed pages from disk.\n")
    agent = LLMAgent(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
    )
    if args.query:
        try:
            run_query(engine, agent, args.query, args.results)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
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
        try:
            run_query(engine, agent, query, args.results)
        except ValueError as exc:
            print(f"Invalid query: {exc}")


if __name__ == "__main__":
    main()
