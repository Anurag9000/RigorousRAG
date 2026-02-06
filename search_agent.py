import os
import json
from typing import List, Optional, Dict, Any, Union
from pydantic import ValidationError

try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageToolCall
except ImportError:
    OpenAI = None # type: ignore

from tools.models import AgentAnswer, Citation
from tools.single_page import fetch_single_page
from tools.web_search import web_search, WEB_SEARCH_TOOL_DEF
from tools.handbook import search_handbook, HANDBOOK_TOOL_DEF
from tools.internal_search import search_internal, INTERNAL_SEARCH_TOOL_DEF
from tools.rag_tool import search_uploaded_docs, RAG_SEARCH_TOOL_DEF
from tools.integrity import (
    check_visual_entailment, VISUAL_ENTAILMENT_TOOL_DEF,
    extract_protocol, PROTOCOL_EXTRACTION_TOOL_DEF,
    run_scientific_debate, DEBATE_TOOL_DEF,
    compare_papers, COMPARISON_TOOL_DEF,
    generate_comparison_matrix, MATRIX_TOOL_DEF,
    detect_conflicts, CONFLICT_TOOL_DEF,
    extract_limitations, LIMITATIONS_TOOL_DEF,
)
from tools.bib import export_to_bibtex, BIBTEX_TOOL_DEF

# Tool Definitions for OpenAI
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
            "description": "Fetch and extract text from a specific URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."}
                },
                "required": ["url"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an advanced academic search agent. Your goal is to answer user questions by synthesizing information from multiple sources:
1. Internal Academic Index (via `search_internal`) - Best for academic papers and specific indexed documents.
2. Internal Handbook (via `search_handbook`) - Best for policy, guidelines, and internal operational questions.
3. Web Search (via `web_search`) - Best for broad, up-to-date information, or when internal sources are insufficient.
4. Uploaded Documents (via `search_uploaded_docs`) - Use this when asking about user files, papers, or specific uploaded content.
5. Integrity & Analysis Tools:
   - `check_visual_entailment`: Use when users ask to verify if a figure supports a claim.
   - `extract_protocol`: Use when users want a step-by-step lab protocol from a methods section.
   - `run_scientific_debate`: Use when users explicitly ask to "debate" or "critique" a paper's claims rigorously.
6. Specific URLs (via `fetch_page`) - Use this if you find a relevant URL in search results and need to read its full content.

Plan your actions:
- Analyze the user's query.
- Decide which tool(s) are most relevant.
- You can call multiple tools in sequence or parallel.
- If one source is insufficient, try another.
- Always include citations in your final answer using the [n] format.
"""

from tools.verification import audit_hallucination
from tools.logger import log_agent_run, log_tool_call
import time

class SearchAgent:
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
             print("WARNING: OPENAI_API_KEY not found.")
        
        self.client = OpenAI(api_key=self.api_key) if (OpenAI and self.api_key) else None

    def run(self, query: str) -> AgentAnswer:
        start_time = time.time()
        if not self.client:
            return AgentAnswer(answer="Error: OpenAI client not initialized.", citations=[])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]

        max_turns = 10
        current_turn = 0

        final_answer = None

        while current_turn < max_turns:
            current_turn += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
            
            msg = response.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    t_start = time.time()
                    self._handle_tool_call(tool_call, messages)
                    log_tool_call(tool_call.function.name, time.time() - t_start, True)
            else:
                content = msg.content
                if not content: break
                
                try:
                    if content.startswith("```json"):
                        content = content[7:-3]
                    elif content.startswith("```"):
                        content = content[3:-3]
                    
                    data = json.loads(content)
                    final_answer = AgentAnswer(**data)
                    break 
                except (json.JSONDecodeError, ValidationError):
                    final_answer = AgentAnswer(answer=content, citations=[])
                    break
        
        if final_answer:
            # Reality check: Verify citations
            audit_note = audit_hallucination(final_answer)
            if "Warning" in audit_note:
                 final_answer.answer += f"\n\n*Note: {audit_note}*"
            
            log_agent_run(query, time.time() - start_time, len(final_answer.citations))
            return final_answer
        
        return AgentAnswer(answer="Error: Maximum turn limit reached.", citations=[])

    def _handle_tool_call(self, tool_call: ChatCompletionMessageToolCall, messages: List[Dict]):
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        result_content = ""
        
        try:
            if tool_name == "web_search":
                citations = web_search(**arguments)
                result_content = json.dumps([c.model_dump() for c in citations])
                
            elif tool_name == "search_handbook":
                result_content = search_handbook(**arguments)
                
            elif tool_name == "search_internal":
                citations = search_internal(**arguments)
                result_content = json.dumps([c.model_dump() for c in citations])
                
            elif tool_name == "search_uploaded_docs":
                citations = search_uploaded_docs(**arguments)
                result_content = json.dumps([c.model_dump() for c in citations])

            elif tool_name == "fetch_page":
                page = fetch_single_page(**arguments)
                result_content = page.model_dump_json()

            elif tool_name == "check_visual_entailment":
                result_content = check_visual_entailment(**arguments)

            elif tool_name == "extract_protocol":
                result_content = extract_protocol(**arguments)

            elif tool_name == "run_scientific_debate":
                result_content = run_scientific_debate(**arguments)
            
            elif tool_name == "compare_papers":
                result_content = compare_papers(**arguments)
            
            elif tool_name == "generate_comparison_matrix":
                result_content = generate_comparison_matrix(**arguments)

            elif tool_name == "detect_conflicts":
                result_content = detect_conflicts(**arguments)

            elif tool_name == "extract_limitations":
                result_content = extract_limitations(**arguments)

            elif tool_name == "export_to_bibtex":
                result_content = export_to_bibtex(**arguments)
                
            else:
                result_content = f"Error: Unknown tool '{tool_name}'"

        except Exception as e:
            result_content = f"Error executing tool '{tool_name}': {str(e)}"

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_content
        })
