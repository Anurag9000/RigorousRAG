"""
Core search agent: multi-tool reasoning loop with parallel tool execution,
owner-scoped document isolation, and citation hallucination auditing.
"""

import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageToolCall
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]

from tools.bib import export_to_bibtex, BIBTEX_TOOL_DEF
from tools.handbook import search_handbook, HANDBOOK_TOOL_DEF
from tools.integrity import (
    COMPARISON_TOOL_DEF,
    CONFLICT_TOOL_DEF,
    DEBATE_TOOL_DEF,
    LIMITATIONS_TOOL_DEF,
    MATRIX_TOOL_DEF,
    PROTOCOL_EXTRACTION_TOOL_DEF,
    VISUAL_ENTAILMENT_TOOL_DEF,
    check_visual_entailment,
    compare_papers,
    detect_conflicts,
    extract_limitations,
    extract_protocol,
    generate_comparison_matrix,
    run_scientific_debate,
)
from tools.internal_search import search_internal, INTERNAL_SEARCH_TOOL_DEF
from tools.logger import log_agent_run, log_tool_call
from tools.models import AgentAnswer, Citation
from tools.rag_tool import search_uploaded_docs, RAG_SEARCH_TOOL_DEF
from tools.single_page import fetch_single_page
from tools.verification import audit_hallucination
from tools.web_search import web_search, WEB_SEARCH_TOOL_DEF

# ---------------------------------------------------------------------------
# Tool schema registry
# ---------------------------------------------------------------------------

TOOLS_SCHEMA = [
    WEB_SEARCH_TOOL_DEF,
    HANDBOOK_TOOL_DEF,
    INTERNAL_SEARCH_TOOL_DEF,
    RAG_SEARCH_TOOL_DEF,
    VISUAL_ENTAILMENT_TOOL_DEF,
    PROTOCOL_EXTRACTION_TOOL_DEF,
    DEBATE_TOOL_DEF,
    COMPARISON_TOOL_DEF,
    MATRIX_TOOL_DEF,
    CONFLICT_TOOL_DEF,
    LIMITATIONS_TOOL_DEF,
    BIBTEX_TOOL_DEF,
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch and extract text content from a specific URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."}
                },
                "required": ["url"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an advanced academic research agent.  Answer user questions \
by synthesizing information from multiple sources:

1. **Internal Academic Index** (`search_internal`) — best for specific indexed papers.
2. **Internal Handbook** (`search_handbook`) — best for policy and operational questions.
3. **Web Search** (`web_search`) — for broad / up-to-date information.
4. **Uploaded Documents** (`search_uploaded_docs`) — for user-uploaded files and papers.
5. **Scientific Integrity Tools:**
   - `check_visual_entailment` — verify a figure supports a claim (requires doc_id + figure label).
   - `extract_protocol` — extract a structured wet-lab protocol from a methods section.
   - `run_scientific_debate` — adversarially critique a claim with Advocate/Skeptic/Judge.
   - `compare_papers` — narrative comparison across multiple documents.
   - `generate_comparison_matrix` — Markdown table of metrics across documents.
   - `detect_conflicts` — find contradictions in the literature.
   - `extract_limitations` — isolate scope constraints and disclaimers.
6. **Fetch Page** (`fetch_page`) — read a specific URL in full.
7. **BibTeX Export** (`export_to_bibtex`) — generate LaTeX bibliography entries.

Guidelines:
- Plan which tool(s) are most relevant before calling them.
- You may dispatch multiple tools simultaneously when they are independent.
- Include inline citations in the format [n] and provide a `citations` list.
- Output your final answer as JSON: {"answer": "...", "citations": [...]}
"""


# ---------------------------------------------------------------------------
# SearchAgent
# ---------------------------------------------------------------------------


class SearchAgent:
    """
    Agentic reasoning loop over a set of research tools.

    Key capabilities:
    - Parallel tool execution via `ThreadPoolExecutor` when the LLM requests
      multiple tool calls in one turn.
    - Per-owner document isolation passed through to `search_uploaded_docs`.
    - Citation hallucination auditing with Jaccard-based semantic grounding.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        owner_id: str = "default_user",
    ) -> None:
        self.model = model
        self.owner_id = owner_id
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not self.api_key and not self.base_url:
            print(
                "WARNING: Neither OPENAI_API_KEY nor OPENAI_BASE_URL is set. "
                "Running in extraction-only mode."
            )

        self.client = None
        if OpenAI is not None:
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            elif self.base_url:
                self.client = OpenAI(api_key="local-no-key", base_url=self.base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> AgentAnswer:
        """
        Execute the multi-turn agentic loop for the given query.

        - Up to 10 turns of LLM interaction.
        - Multiple tool calls in one turn are dispatched in parallel via
          `ThreadPoolExecutor`; results are collected and appended to
          `messages` before the next LLM turn.
        - The final turn's text content is parsed as an `AgentAnswer` JSON;
          falls back to a plain-text answer when parsing fails.
        - Appends a citation audit note when hallucination risk is detected.
        """
        t_start = time.time()

        if not self.client:
            return AgentAnswer(
                answer="Error: OpenAI client is not initialised.", citations=[]
            )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        final_answer: Optional[AgentAnswer] = None

        for _turn in range(10):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                tools=TOOLS_SCHEMA,  # type: ignore[arg-type]
                tool_choice="auto",
            )
            msg = response.choices[0].message
            messages.append(msg)  # type: ignore[arg-type]

            if msg.tool_calls:
                # -------------------------------------------------------
                # Parallel tool execution (Gap 7)
                # -------------------------------------------------------
                if len(msg.tool_calls) == 1:
                    # Single call — skip thread overhead
                    tc = msg.tool_calls[0]
                    t0 = time.time()
                    tool_msg = self._execute_tool(tc)
                    log_tool_call(tc.function.name, time.time() - t0, True)
                    messages.append(tool_msg)
                else:
                    # Multiple independent calls — dispatch concurrently
                    results: List[Tuple[str, str, Dict, float]] = []
                    with ThreadPoolExecutor(
                        max_workers=min(len(msg.tool_calls), 8)
                    ) as pool:
                        future_to_tc: Dict[Future, ChatCompletionMessageToolCall] = {
                            pool.submit(self._execute_tool, tc): tc
                            for tc in msg.tool_calls
                        }
                        for fut in as_completed(future_to_tc):
                            tc = future_to_tc[fut]
                            t0 = time.time()
                            try:
                                tool_msg = fut.result()
                            except Exception as exc:
                                tool_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": f"Error: {exc}",
                                }
                            results.append(
                                (tc.id, tc.function.name, tool_msg, time.time() - t0)
                            )

                    # Append tool results in the original call order
                    tc_order = {tc.id: i for i, tc in enumerate(msg.tool_calls)}
                    results.sort(key=lambda r: tc_order.get(r[0], 0))
                    for _, tool_name, tool_msg, duration in results:
                        log_tool_call(tool_name, duration, True)
                        messages.append(tool_msg)

            else:
                # No tool calls — this is the final answer turn
                content = (msg.content or "").strip()
                if not content:
                    break

                # Strip optional markdown code fences
                if content.startswith("```json"):
                    content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                elif content.startswith("```"):
                    content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                content = content.strip()

                try:
                    data = json.loads(content)
                    final_answer = AgentAnswer(**data)
                except (json.JSONDecodeError, ValidationError):
                    final_answer = AgentAnswer(answer=content, citations=[])
                break

        if not final_answer:
            final_answer = AgentAnswer(
                answer="Error: Maximum reasoning turn limit reached without a final answer.",
                citations=[],
            )

        # --- Citation hallucination audit ---
        audit_note = audit_hallucination(final_answer)
        if "warning" in audit_note.lower():
            final_answer.answer += f"\n\n*{audit_note}*"

        log_agent_run(query, time.time() - t_start, len(final_answer.citations))
        return final_answer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_tool(
        self, tool_call: "ChatCompletionMessageToolCall"
    ) -> Dict[str, Any]:
        """
        Execute a single tool call and return an OpenAI-compatible
        'tool' role message dict.  Thread-safe — mutates no shared state.
        """
        tool_name = tool_call.function.name
        try:
            arguments: Dict[str, Any] = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"Error parsing tool arguments: {exc}",
            }

        try:
            result_content = self._dispatch(tool_name, arguments)
        except Exception as exc:
            result_content = f"Error executing tool '{tool_name}': {exc}"

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_content,
        }

    def _dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Route a tool call to the appropriate function.

        The `owner_id` is injected for `search_uploaded_docs` so that the
        agent never needs to know about tenant context — it flows through
        automatically.
        """
        if tool_name == "web_search":
            citations = web_search(**arguments)
            return json.dumps([c.model_dump() for c in citations])

        if tool_name == "search_handbook":
            return search_handbook(**arguments)

        if tool_name == "search_internal":
            citations = search_internal(**arguments)
            return json.dumps([c.model_dump() for c in citations])

        if tool_name == "search_uploaded_docs":
            # Inject owner_id for multi-tenant isolation (Gap 3)
            citations = search_uploaded_docs(
                owner_id=self.owner_id, **arguments
            )
            return json.dumps([c.model_dump() for c in citations])

        if tool_name == "fetch_page":
            page = fetch_single_page(**arguments)
            return page.model_dump_json()

        if tool_name == "check_visual_entailment":
            return check_visual_entailment(**arguments)

        if tool_name == "extract_protocol":
            return extract_protocol(**arguments)

        if tool_name == "run_scientific_debate":
            return run_scientific_debate(**arguments)

        if tool_name == "compare_papers":
            return compare_papers(**arguments)

        if tool_name == "generate_comparison_matrix":
            return generate_comparison_matrix(**arguments)

        if tool_name == "detect_conflicts":
            return detect_conflicts(**arguments)

        if tool_name == "extract_limitations":
            return extract_limitations(**arguments)

        if tool_name == "export_to_bibtex":
            return export_to_bibtex(**arguments)

        return f"Error: Unknown tool '{tool_name}'"
