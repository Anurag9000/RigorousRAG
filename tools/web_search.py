import os
import requests
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from tools.models import Citation

class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query.")
    allowed_domains: Optional[List[str]] = Field(None, description="List of allowed domains to filter results.")

# Tool definition for OpenAI
WEB_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Perform a broader web search for a given query, optionally filtered by domains.",
        "parameters": WebSearchInput.model_json_schema()
    }
}

def web_search(query: str, allowed_domains: Optional[List[str]] = None) -> List[Citation]:
    """
    Executes a real web search using Serper.dev or a similar provider.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return [
            Citation(
                label="[ERROR]",
                title="Web Search Not Configured",
                url="about:blank",
                source_type="web_search",
                snippet=f"Web search requires SERPER_API_KEY environment variable. Query: {query}"
            )
        ]

    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "num": 5
    })
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        data = response.json()
        
        citations = []
        # Process organic results
        for idx, result in enumerate(data.get("organic", []), start=1):
            # Check domain filter
            link = result.get("link", "")
            if allowed_domains:
                if not any(domain in link for domain in allowed_domains):
                    continue

            citations.append(Citation(
                label=f"[web-{idx}]",
                title=result.get("title", "No Title"),
                url=link,
                source_type="web_search",
                snippet=result.get("snippet", "")
            ))
        return citations
    except Exception as e:
        return [
            Citation(
                label="[ERROR]",
                title="Search API Error",
                url="error:search",
                source_type="web_search",
                snippet=str(e)
            )
        ]
