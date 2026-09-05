#!/usr/bin/env python3
"""Deterministic multi-language semantic census for learner and launch surfaces.

The lower-level :mod:`tools.training_surface_census` finds framework calls and
training-adjacent Python source.  This pass adds deliberately different
coverage: hand-written learner symbols plus non-Python optimization and launch
surfaces.  Evidence is typed so a shell/workflow that *launches* training is
not mistaken for an independent learner implementation.

The scanner has no third-party dependencies and is safe to execute directly as
``python tools/training_surface_semantic_scan.py`` from a repository checkout.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# Direct execution places ``tools/`` rather than the repository root on
# sys.path.  Make the package import deterministic without depending on an
# installed RigorousRAG wheel.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.training_surface_census import _git_blob_sha1, _iter_files, build_report

SCHEMA = "rigorousrag.training_surface_semantic_scan.v2"
VERB_PREFIXES = (
    "fit_",
    "train_",
    "learn_",
    "calibrat",
    "optimiz",
    "finetun",
    "fine_tun",
)
EXACT_NAMES = frozenset(
    {
        "fit",
        "train",
        "partial_fit",
        "fit_transform",
        "advance_training",
        "run_training",
        "training_step",
        "train_step",
        "backward",
    }
)
CLASS_TOKENS = (
    "trainer",
    "learner",
    "fitter",
    "calibrator",
    "trainingengine",
    "optimizer",
)

# Non-Python patterns intentionally require recognizable ML/optimization APIs;
# a generic method named ``fit`` in application JavaScript is not enough.
TEXT_LEARNER_PATTERNS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "javascript": (
        ("tensorflow-js-fit", re.compile(r"\b(?:model|network|net)\.fit(?:Dataset)?\s*\(")),
        ("tensorflow-js-train", re.compile(r"\btf\.train\.[A-Za-z_][\w]*\s*\(")),
        ("optimizer-minimize", re.compile(r"\b(?:optimizer|optimiser)\.minimize\s*\(")),
        ("brainjs-train", re.compile(r"\b(?:net|network)\.train\s*\(")),
    ),
    "r": (
        ("caret-train", re.compile(r"\bcaret::train\s*\(")),
        ("optim", re.compile(r"\boptim(?:x)?\s*\(")),
        ("glmnet", re.compile(r"\bglmnet::?glmnet\s*\(")),
        ("xgboost", re.compile(r"\bxgboost::?xgb\.(?:train|cv)\s*\(")),
        ("keras-fit", re.compile(r"\bfit\s*\(\s*(?:object\s*=\s*)?[A-Za-z_][\w.]*")),
        ("torch-r-backward", re.compile(r"\b(?:loss|objective)\$backward\s*\(")),
    ),
    "julia": (
        ("flux-train", re.compile(r"\bFlux\.train!\s*\(")),
        ("flux-setup", re.compile(r"\bFlux\.setup\s*\(")),
        ("optim-optimize", re.compile(r"\bOptim\.optimize\s*\(")),
        ("mlj-fit", re.compile(r"\bMLJ\.fit!\s*\(")),
        ("zygote-gradient", re.compile(r"\bZygote\.(?:gradient|withgradient)\s*\(")),
    ),
}

SHELL_LAUNCH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("torchrun", re.compile(r"(?:^|\s)torchrun(?:\s|$)")),
    ("accelerate-launch", re.compile(r"\baccelerate\s+launch\b")),
    ("deepspeed-launch", re.compile(r"(?:^|\s)deepspeed(?:\s|$)")),
    ("training-controller", re.compile(r"\brun_all_training\.py\b")),
    (
        "python-training-command",
        re.compile(r"\bpython(?:3(?:\.\d+)?)?\b[^\n]*\b(?:train|training|fit|finetun(?:e|ing))\b", re.I),
    ),
)
WORKFLOW_LAUNCH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = SHELL_LAUNCH_PATTERNS + (
    ("training-audit-command", re.compile(r"\btraining-control-audit\b")),
)

LANGUAGE_BY_SUFFIX = {
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
    ".r": "r",
    ".jl": "julia",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "shell",
}
WORKFLOW_SUFFIXES = frozenset({".yml", ".yaml"})


@dataclass(frozen=True)
class Evidence:
    category: str  # learner | launcher | workflow
    language: str
    kind: str
    line: int
    name: str


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


def _dedupe(items: Iterable[Evidence]) -> list[Evidence]:
    return sorted(
        set(items),
        key=lambda item: (item.line, item.category, item.language, item.kind, item.name),
    )


def _scan_python_text(text: str, rel: str, *, line_offset: int = 0) -> tuple[list[Evidence], str | None]:
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        return [], f"{rel}:{(exc.lineno or 0) + line_offset}: syntax error: {exc.msg}"
    evidence: list[Evidence] = []
    for node in ast.walk(tree):
        line = int(getattr(node, "lineno", 0)) + line_offset
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _interesting(node.name):
            evidence.append(Evidence("learner", "python", "function", line, node.name))
        elif isinstance(node, ast.ClassDef) and any(token in node.name.lower() for token in CLASS_TOKENS):
            evidence.append(Evidence("learner", "python", "class", line, node.name))
        elif isinstance(node, ast.Call):
            called = _name(node.func)
            if called and _interesting(called):
                evidence.append(Evidence("learner", "python", "call", line, called))
    return _dedupe(evidence), None


def _strip_text_comment(line: str, language: str) -> str:
    stripped = line.lstrip()
    if not stripped:
        return ""
    if language in {"shell", "r", "julia"} and stripped.startswith("#"):
        return ""
    if language == "javascript" and (stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*")):
        return ""
    return line


def _scan_text(text: str, language: str, category: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> list[Evidence]:
    evidence: list[Evidence] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_text_comment(raw, language)
        if not line:
            continue
        for kind, pattern in patterns:
            if pattern.search(line):
                evidence.append(Evidence(category, language, kind, lineno, line.strip()[:240]))
    return _dedupe(evidence)


def _scan_notebook(path: Path, rel: str) -> tuple[list[Evidence], str | None]:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], f"{rel}: notebook parse error: {exc}"
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells", []), list):
        return [], f"{rel}: invalid notebook structure"
    evidence: list[Evidence] = []
    pseudo_line = 0
    for index, cell in enumerate(notebook.get("cells", [])):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            text = "".join(str(item) for item in source)
        else:
            text = str(source)
        cell_evidence, error = _scan_python_text(text, f"{rel}#cell-{index}", line_offset=pseudo_line)
        if error:
            return [], error
        evidence.extend(cell_evidence)
        pseudo_line += max(1, len(text.splitlines()))
    return _dedupe(evidence), None


def _workflow_file(rel: str, suffix: str) -> bool:
    return suffix in WORKFLOW_SUFFIXES and rel.startswith(".github/workflows/")


def scan(root: Path) -> dict[str, object]:
    base = build_report(root)
    evidence_by_path: dict[str, list[Evidence]] = {}
    errors: list[str] = list(base["parse_errors"])
    scanned_by_language: dict[str, int] = {}

    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        evidence: list[Evidence] = []
        language = ""
        error: str | None = None

        if suffix == ".py":
            language = "python"
            try:
                text = path.read_text(encoding="utf-8")
            except Exception as exc:
                error = f"{rel}: utf-8 decode error: {exc}"
            else:
                evidence, error = _scan_python_text(text, rel)
        elif suffix == ".ipynb":
            language = "notebook"
            evidence, error = _scan_notebook(path, rel)
        elif suffix in LANGUAGE_BY_SUFFIX:
            language = LANGUAGE_BY_SUFFIX[suffix]
            try:
                text = path.read_text(encoding="utf-8")
            except Exception as exc:
                error = f"{rel}: utf-8 decode error: {exc}"
            else:
                if language == "shell":
                    evidence = _scan_text(text, language, "launcher", SHELL_LAUNCH_PATTERNS)
                else:
                    evidence = _scan_text(text, language, "learner", TEXT_LEARNER_PATTERNS[language])
        elif _workflow_file(rel, suffix):
            language = "workflow"
            try:
                text = path.read_text(encoding="utf-8")
            except Exception as exc:
                error = f"{rel}: utf-8 decode error: {exc}"
            else:
                evidence = _scan_text(text, language, "workflow", WORKFLOW_LAUNCH_PATTERNS)
        else:
            continue

        scanned_by_language[language] = scanned_by_language.get(language, 0) + 1
        if error:
            errors.append(error)
        if evidence:
            evidence_by_path[rel] = evidence

    # Base direct files are concrete mutating/fitting surfaces even if their
    # function names do not satisfy the semantic name heuristic.
    for rel in base["direct_files"]:
        existing = evidence_by_path.setdefault(rel, [])
        existing.append(Evidence("learner", "python", "base-direct-signal", 0, "training_surface_census.direct"))
        evidence_by_path[rel] = _dedupe(existing)

    learner_files = sorted(
        rel for rel, items in evidence_by_path.items() if any(item.category == "learner" for item in items)
    )
    launcher_files = sorted(
        rel for rel, items in evidence_by_path.items() if any(item.category == "launcher" for item in items)
    )
    workflow_files = sorted(
        rel for rel, items in evidence_by_path.items() if any(item.category == "workflow" for item in items)
    )
    candidate_files = sorted(evidence_by_path)

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "root": ".",
        "base_inventory_sha256": base["inventory_sha256"],
        "parse_errors": sorted(set(errors)),
        "scanned_by_language": dict(sorted(scanned_by_language.items())),
        "candidate_files": candidate_files,
        "learner_files": learner_files,
        "launcher_files": launcher_files,
        "workflow_files": workflow_files,
        "evidence": {
            rel: [asdict(item) for item in _dedupe(evidence_by_path[rel])]
            for rel in candidate_files
        },
        "summary": {
            "candidate_files": len(candidate_files),
            "learner_files": len(learner_files),
            "launcher_files": len(launcher_files),
            "workflow_files": len(workflow_files),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["inventory_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=".training_control/training_surface_semantic_scan.json")
    parser.add_argument("--fail-on-parse-error", action="store_true", default=False)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    payload = scan(root)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": payload["schema"], **payload["summary"], "inventory_sha256": payload["inventory_sha256"]}, sort_keys=True))
    for key in ("learner_files", "launcher_files", "workflow_files"):
        print(f"\n[{key}]")
        for path in payload[key]:
            print(path)
    if payload["parse_errors"]:
        print("\n[parse_errors]")
        for err in payload["parse_errors"]:
            print(err)
        if args.fail_on_parse_error:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
