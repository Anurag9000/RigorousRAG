"""LLM-enabled summarisation agent for academic search results."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence

from Searching import SearchHit

try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore[assignment]


DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class CitationSummary:
    summary: str
    sources: List[str]


class ExtractiveFallback:
    """Simple extractive summariser used when the LLM client is unavailable."""

    def summarise(
        self, query: str, hits: Sequence[SearchHit], contexts: Sequence[dict]
    ) -> CitationSummary:
        lines: List[str] = []
        sources: List[str] = []
        for idx, (hit, context) in enumerate(zip(hits, contexts), start=1):
            snippet = context.get("text", "")[:280]
            lines.append(
                f"[{idx}] {hit.title}: {snippet}..."
            )
            sources.append(f"[{idx}] {hit.title} — {hit.url}")

        if not lines:
            lines.append("No supporting documents were available to summarise.")

        summary = "Query: {query}\n\n".format(query=query)
        summary += "\n".join(lines)
        return CitationSummary(summary=summary, sources=sources)


class LLMAgent:
    """Thin wrapper around an OpenAI-compatible LLM for academic summarisation."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if OpenAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        self.fallback = ExtractiveFallback()

    def summarise(
        self, query: str, hits: Sequence[SearchHit], contexts: Sequence[dict]
    ) -> CitationSummary:
        if not hits or not contexts:
            return self.fallback.summarise(query, hits, contexts)
        if not self.client:
            return self.fallback.summarise(query, hits, contexts)

        prompt = self._build_prompt(query, hits, contexts)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                top_p=0.9,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an academic research assistant. "
                            "Summarise the provided sources into an impartial, concise answer. "
                            "Use numbered citations like [1], [2] referencing the source list. "
                            "Highlight consensus and note disagreements if relevant."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            message = response.choices[0].message.content or ""
            sources = [
                f"[{idx}] {hit.title} — {hit.url}"
                for idx, hit in enumerate(hits, start=1)
            ]
            return CitationSummary(summary=message.strip(), sources=sources)
        except Exception:
            return self.fallback.summarise(query, hits, contexts)

    def _build_prompt(
        self, query: str, hits: Sequence[SearchHit], contexts: Sequence[dict]
    ) -> str:
        lines = [
            f"User query: {query}",
            "",
            "Sources:",
        ]
        for idx, (hit, context) in enumerate(zip(hits, contexts), start=1):
            excerpt = context.get("text", "")
            lines.append(f"[{idx}] Title: {hit.title}")
            lines.append(f"     URL: {hit.url}")
            lines.append(f"     Excerpt: {excerpt}")
            lines.append("")
        lines.append(
            "Task: Produce a structured summary (2-4 short paragraphs) followed by "
            "a bulleted list of key facts. Every statement requiring evidence must cite "
            "sources with the [n] notation."
        )
        return "\n".join(lines)
