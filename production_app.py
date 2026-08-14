"""Hardened API plus durable owner-scoped governance routes."""
from pathlib import Path
import os

import server as base
from fastapi import Request
from tools.control_api import build_control_router
from tools.feedback_store import FeedbackStore
from tools.review_store import ReviewStore

root = Path(os.environ.get("CLASSIC_STORAGE_DIR", "data")).resolve() / "governance"
root.mkdir(parents=True, exist_ok=True)
reviews = ReviewStore(root / "reviews.sqlite3")
feedback = FeedbackStore(root / "feedback.sqlite3")
app = base.app

_REQUIRED_GOVERNANCE_ROUTES = frozenset({"/reviews", "/reviews/claim", "/feedback"})


def _route_paths() -> set[str]:
    return {
        path
        for route in app.routes
        if isinstance((path := getattr(route, "path", None)), str)
    }


def _ensure_governance_routes() -> None:
    if _REQUIRED_GOVERNANCE_ROUTES.issubset(_route_paths()):
        return
    governance = build_control_router(
        principal_dependency=base.get_principal,
        review_store=reviews,
        feedback_store=feedback,
    )
    known_paths = _route_paths()
    for route in governance.routes:
        path = getattr(route, "path", None)
        if isinstance(path, str) and path not in known_paths:
            app.router.routes.append(route)
            known_paths.add(path)
    missing = _REQUIRED_GOVERNANCE_ROUTES.difference(_route_paths())
    if missing:
        raise RuntimeError(
            "Production governance routes failed to mount: " + ", ".join(sorted(missing))
        )


_ensure_governance_routes()


@app.middleware("http")
async def governance_no_store(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(("/reviews", "/feedback")):
        response.headers["Cache-Control"] = "no-store"
    return response


__all__ = ["app", "feedback", "reviews"]
