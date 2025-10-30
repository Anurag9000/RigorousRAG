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

try:
    import ollama
except ImportError:
    ollama = None


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"


@dataclass
class CitationSummary:
    summary: str
    sources: List[str]


class ExtractiveFallback:
    """Simple extractive summariser used when neural LLM clients are unavailable."""

    def summarise(
        self, query: str, hits: Sequence[SearchHit], contexts: Sequence[dict]
    ) -> CitationSummary:
        lines: List[str] = []
        sources: List[str] = []
        for idx, (hit, context) in enumerate(zip(hits, contexts), start=1):
            snippet = context.get("text", "")[:280]
            lines.append(f"[{idx}] {hit.title}: {snippet}...")
            sources.append(f"[{idx}] {hit.title} — {hit.url}")

        if not lines:
            lines.append("No supporting documents were available to summarise.")

        summary = f"Query: {query}\n\n" + "\n".join(lines)
        return CitationSummary(summary=summary, sources=sources)


class LLMAgent:
    """Summarisation orchestrator supporting OpenAI, Ollama, and extractive fallbacks."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = None,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        ollama_host: str | None = None,
    ) -> None:
        self.openai_model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST")

        self.openai_client = None
        if OpenAI and self.api_key:
            try:
                self.openai_client = OpenAI(api_key=self.api_key)
            except Exception:
                self.openai_client = None

        self.ollama_client = None
        if ollama:
            try:
                if hasattr(ollama, "Client"):
                    self.ollama_client = (
                        ollama.Client(host=self.ollama_host)
                        if self.ollama_host
                        else ollama.Client()
                    )
                else:
                    self.ollama_client = ollama
            except Exception:
                self.ollama_client = None

        self.fallback = ExtractiveFallback()

    def summarise(
        self, query: str, hits: Sequence[SearchHit], contexts: Sequence[dict]
    ) -> CitationSummary:
        if not hits or not contexts:
            return self.fallback.summarise(query, hits, contexts)

        prompt = self._build_prompt(query, hits, contexts)

        summary = self._summarise_with_openai(prompt, hits)
        if summary:
            return summary

        summary = self._summarise_with_ollama(prompt, hits)
        if summary:
            return summary

        return self.fallback.summarise(query, hits, contexts)

    def _summarise_with_openai(
        self, prompt: str, hits: Sequence[SearchHit]
    ) -> CitationSummary | None:
        if not self.openai_client:
            return None
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
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
                    {"role": "user", "content": prompt},
                ],
            )
            message = response.choices[0].message.content or ""
            return CitationSummary(
                summary=message.strip(),
                sources=[
                    f"[{idx}] {hit.title} — {hit.url}"
                    for idx, hit in enumerate(hits, start=1)
                ],
            )
        except Exception:
            return None

    def _summarise_with_ollama(
        self, prompt: str, hits: Sequence[SearchHit]
    ) -> CitationSummary | None:
        if not self.ollama_client:
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an academic research assistant. Provide a concise, neutral summary "
                    "with numbered citations like [1], [2] that refer to the provided sources."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            if hasattr(self.ollama_client, "chat"):
                response = self.ollama_client.chat(
                    model=self.ollama_model, messages=messages
                )
            else:
                # Module-level API (older python-ollama releases)
                chat_kwargs = {"model": self.ollama_model, "messages": messages}
                if self.ollama_host:
                    chat_kwargs["host"] = self.ollama_host
                response = ollama.chat(**chat_kwargs)  # type: ignore[misc]

            message = response.get("message", {}).get("content", "").strip()
            if not message:
                return None
            return CitationSummary(
                summary=message,
                sources=[
                    f"[{idx}] {hit.title} — {hit.url}"
                    for idx, hit in enumerate(hits, start=1)
                ],
            )
        except Exception:
            return None

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
