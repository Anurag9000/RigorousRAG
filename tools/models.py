from typing import List, Literal, Optional
from pydantic import BaseModel, Field

SourceType = Literal["internal_index", "handbook", "web_page", "web_search", "unknown"]

class Citation(BaseModel):
    """
    Represents a single source citation.
    """
    label: str = Field(..., description="The inline citation label, e.g., '[1]'.")
    title: str = Field(..., description="Title of the source.")
    url: str = Field(..., description="URL or file path of the source.")
    source_type: SourceType = Field(..., description="Type of the source.")
    snippet: Optional[str] = Field(None, description="Relevant text snippet from the source.")

class AgentAnswer(BaseModel):
    """
    The standardized structured answer from the search agent.
    """
    answer: str = Field(..., description="The natural language answer with inline citations.")
    citations: List[Citation] = Field(default_factory=list, description="List of citations referenced in the answer.")
