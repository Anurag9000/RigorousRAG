from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from tools.models import Citation, AgentAnswer

HANDBOOK_PATH = Path("handbook.md")

class HandbookInput(BaseModel):
    query: str = Field(..., description="Query to answer from the handbook.")

HANDBOOK_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_handbook",
        "description": "Retrieve information from the internal handbook.",
        "parameters": HandbookInput.model_json_schema()
    }
}

def search_handbook(query: str) -> str:
    """
    Reads the handbook and returns its content. 
    Ideally, this would perform a semantic search if the handbook were large.
    For now, it returns the full text for the LLM to process.
    """
    if not HANDBOOK_PATH.exists():
        return "Handbook file not found."
    
    try:
        content = HANDBOOK_PATH.read_text(encoding="utf-8")
        return content
    except Exception as e:
        return f"Error reading handbook: {str(e)}"
