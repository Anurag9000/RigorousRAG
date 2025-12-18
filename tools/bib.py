from typing import List, Dict
import json
from tools.models import Citation

BIBTEX_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "export_to_bibtex",
        "description": "Convert a list of citations into a BibTeX-formatted string for LaTeX.",
        "parameters": {
            "type": "object",
            "properties": {
                "citations": {
                    "type": "array", 
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "authors": {"type": "string"},
                            "year": {"type": "string"},
                            "doi": {"type": "string"},
                            "url": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["citations"]
        }
    }
}

def export_to_bibtex(citations: List[Dict]) -> str:
    """
    Format a list of metadata into BibTeX entries.
    """
    entries = []
    for i, c in enumerate(citations):
        cite_key = f"ref_{i+1}"
        entry = (
            f"@article{{{cite_key},\n"
            f"  title = {{{c.get('title', 'Untitled')}}},\n"
            f"  author = {{{c.get('authors', 'Unknown')}}},\n"
            f"  year = {{{c.get('year', 'n.d.')}}},\n"
            f"  url = {{{c.get('url', '')}}},\n"
            f"  doi = {{{c.get('doi', '')}}}\n"
            f"}}"
        )
        entries.append(entry)
    
    return "\n\n".join(entries)
