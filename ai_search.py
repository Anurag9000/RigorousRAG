"""AI-assisted CLI over the persisted classic academic index."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Any

from Searching import AcademicSearchEngine, SearchHit
from llm_agent import CitationSummary, LLMAgent
from tools.privacy import mask_metadata_text

_MAX_QUERY_CHARS = 2000
_MAX_RESULTS = 20


def _bounded_limit(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("Result limit must be an integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Result limit must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("Result limit must be an integer.")
    if not 1 <= numeric <= _MAX_RESULTS:
        raise ValueError(f"Result limit must be between 1 and {_MAX_RESULTS}.")
    return numeric


def format_summary(summary: object) -> str:
    """Preserve bounded Markdown while masking private metadata in direct calls."""

    if not isinstance(summary, str):
        return "No reliable summary was produced."
    bounded = mask_metadata_text(summary.strip()[:20_000])
    return bounded or "No reliable summary was produced."


def _validated_query(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("A research query must be a string.")
    query = value.strip()
    if not query:
        raise ValueError("A research query is required.")
    if len(query) > _MAX_QUERY_CHARS:
        raise ValueError("Research queries may contain at most 2,000 characters.")
    if "\x00" in query:
        raise ValueError("Research queries contain invalid control characters.")
    return query


def _bounded_hits(value: Any, maximum: int) -> list[SearchHit]:
    if isinstance(value, (str, bytes, bytearray)):
        raise RuntimeError("The search backend returned an invalid result collection.")
    try:
        candidates = itertools.islice(iter(value), maximum)
        return [item for item in candidates if isinstance(item, SearchHit)]
    except Exception as exc:
        raise RuntimeError(
            "The search backend returned an invalid result collection."
        ) from exc


def run_query(
    engine: AcademicSearchEngine,
    agent: LLMAgent,
    query: str,
    limit: int,
) -> None:
    bounded_query = _validated_query(query)
    requested = _bounded_limit(limit)
    hits = _bounded_hits(engine.search(bounded_query, limit=requested), requested)
    if not hits:
        print("No results found.")
        return
    contexts = engine.gather_context(hits, max_chars=24_000)
    summary = agent.summarise(bounded_query, hits, contexts)
    if not isinstance(summary, CitationSummary):
        raise RuntimeError("The summarizer returned an invalid result.")
    print("\n=== AI Summary ===")
    print(format_summary(summary.summary))
    if summary.warning:
        print(f"\nWarning: {format_summary(summary.warning)}")
    print("\n=== Sources ===")
    for source in summary.sources[:20]:
        print(f"- {mask_metadata_text(source)[:5000]}")
    print("\n=== Top Results ===")
    for hit in hits:
        title = mask_metadata_text(hit.title)[:500]
        url = mask_metadata_text(hit.url)[:4096]
        snippet = mask_metadata_text(hit.snippet)[:300]
        print(f"{hit.rank}. {title} ({hit.score:.3f})")
        print(f"   {url}")
        if snippet:
            print(f"   {snippet}")
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
    parser.add_argument(
        "--storage-dir",
        default=os.getenv("CLASSIC_STORAGE_DIR", "data"),
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--ollama-model", default="qwen3:8b")
    parser.add_argument("--ollama-host")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        requested = _bounded_limit(args.results)
        with AcademicSearchEngine(
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            request_delay=args.delay,
            storage_dir=args.storage_dir,
        ) as engine:
            if args.rebuild or not engine.ready:
                print("Crawling and rebuilding the academic index...")
                total_pages = engine.build()
                print(f"Indexed {total_pages} pages.\n")
            else:
                print(
                    f"Loaded {len(engine.index.documents)} indexed pages from disk.\n"
                )
            agent = LLMAgent(
                model=args.model,
                api_key=args.api_key,
                base_url=args.base_url,
                ollama_model=args.ollama_model,
                ollama_host=args.ollama_host,
            )
            if args.query is not None:
                run_query(engine, agent, args.query, requested)
                return 0
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
                    run_query(engine, agent, query, requested)
                except ValueError as exc:
                    print(f"Invalid query: {exc}")
                except Exception:
                    print("The search request failed. Retry after checking local services.")
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        print(
            "The academic search service could not be initialized.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
