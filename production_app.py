"""Hardened API plus durable owner-scoped governance routes."""
from pathlib import Path
import os

import server as base
from tools.control_api import build_control_router
from tools.feedback_store import FeedbackStore
from tools.review_store import ReviewStore

root = Path(os.environ.get("CLASSIC_STORAGE_DIR", "data")).resolve() / "governance"
root.mkdir(parents=True, exist_ok=True)
reviews = ReviewStore(root / "reviews.sqlite3")
feedback = FeedbackStore(root / "feedback.sqlite3")
app = base.app

if not any(getattr(route, "path", None) == "/reviews" for route in app.routes):
    app.include_router(
        build_control_router(
            principal_dependency=base.get_principal,
            review_store=reviews,
            feedback_store=feedback,
        )
    )

__all__ = ["app", "feedback", "reviews"]
