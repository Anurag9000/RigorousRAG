#!/usr/bin/env python3
"""High-signal semantic learner scan layered on the deterministic census.

This catches hand-written learners that do not call framework ``.fit`` APIs by
inspecting declared/called training symbols.  It is diagnostic until every
candidate is assigned a closed-world accounting disposition.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from tools.training_surface_census import _iter_files, _git_blob_sha1, build_report

SCHEMA = "rigorousrag.training_surface_semantic_scan.v1"
VERB_PREFIXES = (
    "fit_", "train_", "learn_", "calibrat", "optimiz", "finetun", "fine_tun",
)
EXACT_NAMES = frozenset({
    "fit", "train", "partial_fit", "fit_transform", "advance_training",
    "run_training", "training_step", "train_step", "backward",
})
CLASS_TOKENS = ("trainer", "learner", "fitter", "calibrator", "trainingengine", "optimizer")


def _name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    return ""


def _interesting(name: str) -> bool:
    low = name.lower().rsplit(".", 1)[-1]
    return low in EXACT_NAMES or low.startswith(VERB_PREFIXES)


def scan(root: Path) -> dict[str, object]:
    base = build_report(root)
    direct = set(base["direct_files"])
    semantic: dict[str, list[dict[str, object]]] = {}
    errors: list[str] = []
    for path in _iter_files(root):
        if path.suffix.lower() != ".py":
            continue
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=rel)
        except Exception as exc:
            errors.append(f"{rel}: {exc}")
            continue
        signals: list[dict[str, object]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _interesting(node.name):
                signals.append({"kind":"function","line":node.lineno,"name":node.name})
            elif isinstance(node, ast.ClassDef) and any(t in node.name.lower() for t in CLASS_TOKENS):
                signals.append({"kind":"class","line":node.lineno,"name":node.name})
            elif isinstance(node, ast.Call):
                called = _name(node.func)
                if called and _interesting(called):
                    signals.append({"kind":"call","line":getattr(node,"lineno",0),"name":called})
        if signals:
            semantic[rel] = sorted(signals, key=lambda x:(int(x["line"]),str(x["kind"]),str(x["name"])))
    candidates = sorted(direct | set(semantic))
    payload = {
        "schema": SCHEMA,
        "base_inventory_sha256": base["inventory_sha256"],
        "parse_errors": sorted(set(list(base["parse_errors"]) + errors)),
        "candidate_files": candidates,
        "direct_files": sorted(direct),
        "semantic_signals": {key: semantic[key] for key in sorted(semantic)},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["inventory_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def main() -> int:
    root = Path(".").resolve()
    payload = scan(root)
    out = root / ".training_control/training_surface_semantic_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps({"schema":payload["schema"],"candidate_count":len(payload["candidate_files"]),"inventory_sha256":payload["inventory_sha256"]}, sort_keys=True))
    print("[candidate_files]")
    for path in payload["candidate_files"]:
        print(path)
    if payload["parse_errors"]:
        print("[parse_errors]")
        for err in payload["parse_errors"]:
            print(err)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
