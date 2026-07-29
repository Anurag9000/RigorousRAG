"""Command-line interface for the owner-scoped research agent."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional

from search_agent import SearchAgent
from tools.models import AgentAnswer, Citation
from tools.privacy import mask_metadata_text

_MAX_QUERY_CHARS = 20_000
_MAX_DISPLAY_CHARS = 100_000
_MAX_CITATIONS = 100


def _bounded_query(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("The query must be a string.")
    query = value.strip()
    if not query:
        raise ValueError("The query may not be empty.")
    if len(query) > _MAX_QUERY_CHARS:
        raise ValueError(
            f"The query may contain at most {_MAX_QUERY_CHARS:,} characters."
        )
    if "\x00" in query:
        raise ValueError("The query contains an invalid null character.")
    return query


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Academic Agentic Search CLI")
    parser.add_argument("--query", "-q", help="Run a single query and exit.")
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI-compatible model name.",
    )
    parser.add_argument(
        "--owner-id",
        default=os.getenv("SINGLE_USER_OWNER_ID", "default_user"),
        help="Server-style owner identity used for local document retrieval.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--local",
        action="store_true",
        help="Use a local Ollama-compatible endpoint with llama3.1 by default.",
    )
    modes.add_argument(
        "--demo",
        action="store_true",
        help="Use a local Ollama-compatible endpoint with qwen2.5:0.5b.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _display(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return mask_metadata_text(value)[:limit]


def print_result(result: AgentAnswer) -> None:
    if not isinstance(result, AgentAnswer):
        raise ValueError("The research agent returned an invalid result.")
    print("\nAnswer:")
    print(_display(result.answer, _MAX_DISPLAY_CHARS) or "No answer was produced.")

    citations = result.citations[:_MAX_CITATIONS]
    if not citations:
        return
    print("\nCitations:")
    for citation in citations:
        if not isinstance(citation, Citation):
            continue
        label = _display(citation.label, 64)
        title = _display(citation.title, 500)
        source_type = _display(citation.source_type, 100)
        url = _display(citation.url, 4096)
        print(f"{label} {title} ({source_type})")
        print(f"    URL: {url}")
        if citation.snippet:
            snippet = " ".join(_display(citation.snippet, 4000).split())
            if len(snippet) > 100:
                snippet = snippet[:97] + "..."
            print(f"    Excerpt: {snippet}")
        print()


def _build_agent(args: argparse.Namespace) -> SearchAgent:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = args.model

    if args.local or args.demo:
        mode_name = "DEMO" if args.demo else "LOCAL"
        print(f"[INFO] Running in {mode_name} mode via a local endpoint.")
        api_key = "ollama"
        base_url = "http://localhost:11434/v1"
        if args.demo:
            model = "qwen2.5:0.5b"
        elif model == "gpt-4o":
            model = "llama3.1"
    elif not api_key and not base_url:
        raise RuntimeError(
            "No OpenAI-compatible provider is configured. Set OPENAI_API_KEY or "
            "OPENAI_BASE_URL, or use --local."
        )

    print("[INFO] Initializing the research agent.")
    return SearchAgent(
        model=model,
        api_key=api_key,
        base_url=base_url,
        owner_id=args.owner_id,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        agent = _build_agent(args)
        if args.query is not None:
            query = _bounded_query(args.query)
            result = agent.run(query)
            print_result(result)
            return 0

        print("Academic Search Agent (type 'exit' or 'quit' to stop)")
        print("-----------------------------------------------------")
        while True:
            try:
                raw_input = input("You> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if raw_input.strip().lower() in {"exit", "quit"}:
                break
            if not raw_input.strip():
                continue
            try:
                query = _bounded_query(raw_input)
                print("\nAgent: Thinking...")
                print_result(agent.run(query))
                print("-" * 40)
            except ValueError as exc:
                print(f"Invalid query: {exc}")
            except Exception:
                print("The research request failed. Check the configured local services.")
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        print(
            "The research agent could not be initialized. Check provider configuration.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
