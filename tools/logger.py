import json
import time
import os
from datetime import datetime

LOG_FILE = "usage_metrics.jsonl"

def log_activity(activity_type: str, details: dict):
    """
    Logs an activity (tool call, query, error) to usage_metrics.jsonl.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": activity_type,
        "details": details
    }
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def log_tool_call(tool_name: str, duration: float, success: bool, tokens: int = 0):
    log_activity("tool_call", {
        "tool": tool_name,
        "duration_sec": round(duration, 3),
        "success": success,
        "estimated_tokens": tokens
    })

def log_agent_run(query: str, total_time: float, citation_count: int):
    log_activity("agent_run", {
        "query": query,
        "duration_sec": round(total_time, 3),
        "citations": citation_count
    })
