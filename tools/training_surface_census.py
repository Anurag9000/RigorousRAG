#!/usr/bin/env python3
"""Deterministic closed-world census for learner/training-adjacent source surfaces.

This scanner intentionally does not rely on GitHub code search.  It walks the
checked-out repository, parses Python with :mod:`ast`, and records source files
that contain concrete learning/fitting/optimization/index-training signals.
The first use is diagnostic (``--report``); a later manifest-backed strict mode
can turn the same inventory into a hard training-control acceptance gate.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SCHEMA = "rigorousrag.training_surface_census.v1"

EXCLUDED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        ".training_control",
    }
)
EXCLUDED_PREFIXES = (
    "tests/",
    "test/",
)

# Calls that directly mutate learned parameters or fitted estimator state.
DIRECT_METHODS = frozenset(
    {
        "fit",
        "partial_fit",
        "fit_predict",
        "backward",
        "minimize",
        "minimize_loss",
        "apply_gradients",
        "train_on_batch",
        "fit_generator",
    }
)
# These are strong only when their receiver/name gives training context.
CONTEXT_METHODS = frozenset({"step", "train", "compile"})
TRAINING_RECEIVER_TOKENS = (
    "optim",
    "optimizer",
    "trainer",
    "learner",
    "estimator",
    "model",
    "network",
    "net",
    "index",
)

TRAINING_CONSTRUCTOR_TOKENS = frozenset(
    {
        "Trainer",
        "Seq2SeqTrainer",
        "SFTTrainer",
        "DPOTrainer",
        "PPOTrainer",
        "TrainingArguments",
        "KMeans",
        "MiniBatchKMeans",
        "LogisticRegression",
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SGDClassifier",
        "SGDRegressor",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "CalibratedClassifierCV",
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "CatBoostClassifier",
        "CatBoostRegressor",
    }
)
ML_IMPORT_PREFIXES = (
    "torch.optim",
    "torch.nn",
    "transformers",
    "trl",
    "peft",
    "sklearn",
    "xgboost",
    "lightgbm",
    "catboost",
    "tensorflow",
    "keras",
    "jax",
    "optax",
    "flax",
    "sentence_transformers",
    "faiss",
)

PATH_HINTS = (
    "train",
    "training",
    "fit",
    "calibr",
    "rank",
    "cluster",
    "learn",
    "finetun",
    "fine_tun",
    "fine-tun",
    "optim",
    "adapter",
    "checkpoint",
    "retriev",
    "embed",
    "index",
)
SOURCE_SUFFIXES = frozenset({".py", ".ipynb", ".sh", ".bash", ".zsh", ".ps1", ".r", ".jl"})


@dataclass(frozen=True)
class Signal:
    kind: str
    line: int
    detail: str
    strength: str


@dataclass(frozen=True)
class Finding:
    path: str
    git_blob_sha1: str
    sha256: str
    executable: bool
    direct_signals: tuple[Signal, ...]
    contextual_signals: tuple[Signal, ...]
    import_signals: tuple[Signal, ...]
    path_hints: tuple[str, ...]

    @property
    def strongest(self) -> str:
        if self.direct_signals:
            return "direct"
        if self.contextual_signals:
            return "contextual"
        if self.import_signals:
            return "import-only"
        return "path-only"


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _dotted_name(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _dotted_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    if isinstance(node, ast.Subscript):
        return _dotted_name(node.value)
    return ""


def _is_executable_module(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        left = _dotted_name(test.left)
        comparators = test.comparators
        if left != "__name__" or len(comparators) != 1:
            continue
        rhs = comparators[0]
        if isinstance(rhs, ast.Constant) and rhs.value == "__main__":
            return True
    return False


def _constructor_name(node: ast.Call) -> str:
    dotted = _dotted_name(node.func)
    return dotted.rsplit(".", 1)[-1]


def _method_signal(node: ast.Call) -> Signal | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    method = func.attr
    receiver = _dotted_name(func.value)
    detail = f"{receiver}.{method}" if receiver else method
    if method in DIRECT_METHODS:
        return Signal("call", int(getattr(node, "lineno", 0)), detail, "direct")
    if method in CONTEXT_METHODS:
        receiver_lower = receiver.lower()
        if any(token in receiver_lower for token in TRAINING_RECEIVER_TOKENS):
            return Signal("call", int(getattr(node, "lineno", 0)), detail, "contextual")
    return None


def _call_name_signal(node: ast.Call) -> Signal | None:
    name = _constructor_name(node)
    line = int(getattr(node, "lineno", 0))
    if name in TRAINING_CONSTRUCTOR_TOKENS:
        return Signal("constructor", line, _dotted_name(node.func), "contextual")
    dotted = _dotted_name(node.func)
    lowered = dotted.lower()
    if "torch.optim." in lowered or lowered.startswith("optim."):
        return Signal("optimizer-constructor", line, dotted, "direct")
    if any(token in lowered for token in ("gradienttape", "value_and_grad", "grad_and_value", "train_step")):
        return Signal("training-api", line, dotted, "contextual")
    if any(token in lowered for token in ("prepare_model_for_kbit_training", "get_peft_model")):
        return Signal("fine-tuning-api", line, dotted, "contextual")
    return None


def _import_signals(tree: ast.AST) -> list[Signal]:
    out: list[Signal] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
        else:
            continue
        for name in names:
            if name.startswith(ML_IMPORT_PREFIXES):
                out.append(Signal("import", int(getattr(node, "lineno", 0)), name, "import-only"))
    return out


def _scan_python(path: Path, rel: str) -> tuple[Finding | None, str | None]:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, f"{rel}: utf-8 decode error: {exc}"
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as exc:
        return None, f"{rel}:{exc.lineno or 0}: syntax error: {exc.msg}"

    direct: list[Signal] = []
    contextual: list[Signal] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for signal in (_method_signal(node), _call_name_signal(node)):
            if signal is None:
                continue
            if signal.strength == "direct":
                direct.append(signal)
            else:
                contextual.append(signal)

    imports = _import_signals(tree)
    lower_rel = rel.lower()
    hints = tuple(sorted({hint for hint in PATH_HINTS if hint in lower_rel}))
    if not direct and not contextual and not imports and not hints:
        return None, None

    key = lambda s: (s.line, s.kind, s.detail, s.strength)
    finding = Finding(
        path=rel,
        git_blob_sha1=_git_blob_sha1(data),
        sha256=hashlib.sha256(data).hexdigest(),
        executable=_is_executable_module(tree),
        direct_signals=tuple(sorted(set(direct), key=key)),
        contextual_signals=tuple(sorted(set(contextual), key=key)),
        import_signals=tuple(sorted(set(imports), key=key)),
        path_hints=hints,
    )
    return finding, None


def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in EXCLUDED_DIR_NAMES)
        base = Path(dirpath)
        for filename in sorted(filenames):
            path = base / filename
            rel = path.relative_to(root).as_posix()
            if rel.startswith(EXCLUDED_PREFIXES):
                continue
            yield path


def build_report(root: Path) -> dict[str, object]:
    findings: list[Finding] = []
    parse_errors: list[str] = []
    hinted_non_python: list[dict[str, object]] = []
    python_scanned = 0
    files_scanned = 0

    for path in _iter_files(root):
        files_scanned += 1
        rel = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix == ".py":
            python_scanned += 1
            finding, error = _scan_python(path, rel)
            if error:
                parse_errors.append(error)
            elif finding:
                findings.append(finding)
            continue
        if suffix in SOURCE_SUFFIXES:
            lower_rel = rel.lower()
            hints = sorted({hint for hint in PATH_HINTS if hint in lower_rel})
            if hints:
                data = path.read_bytes()
                hinted_non_python.append(
                    {
                        "path": rel,
                        "git_blob_sha1": _git_blob_sha1(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "path_hints": hints,
                    }
                )

    findings.sort(key=lambda item: item.path)
    hinted_non_python.sort(key=lambda item: str(item["path"]))
    direct_files = [item.path for item in findings if item.direct_signals]
    contextual_files = [item.path for item in findings if item.contextual_signals and not item.direct_signals]
    import_only_files = [
        item.path
        for item in findings
        if item.import_signals and not item.direct_signals and not item.contextual_signals
    ]
    path_only_files = [
        item.path
        for item in findings
        if item.path_hints and not item.direct_signals and not item.contextual_signals and not item.import_signals
    ]
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "root": ".",
        "files_scanned": files_scanned,
        "python_files_scanned": python_scanned,
        "parse_errors": sorted(parse_errors),
        "summary": {
            "finding_files": len(findings),
            "direct_files": len(direct_files),
            "contextual_files": len(contextual_files),
            "import_only_files": len(import_only_files),
            "path_only_files": len(path_only_files),
            "hinted_non_python_files": len(hinted_non_python),
        },
        "direct_files": direct_files,
        "contextual_files": contextual_files,
        "import_only_files": import_only_files,
        "path_only_files": path_only_files,
        "hinted_non_python_files": hinted_non_python,
        "findings": [
            {
                **{k: v for k, v in asdict(item).items() if k not in {"direct_signals", "contextual_signals", "import_signals"}},
                "strongest": item.strongest,
                "direct_signals": [asdict(s) for s in item.direct_signals],
                "contextual_signals": [asdict(s) for s in item.contextual_signals],
                "import_signals": [asdict(s) for s in item.import_signals],
            }
            for item in findings
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["inventory_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=".training_control/training_surface_census.json")
    parser.add_argument("--fail-on-parse-error", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    report = build_report(root)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = report["summary"]
    print(json.dumps({"schema": report["schema"], "inventory_sha256": report["inventory_sha256"], **summary}, sort_keys=True))
    for key in ("direct_files", "contextual_files", "import_only_files", "path_only_files"):
        print(f"\n[{key}]")
        for rel in report[key]:
            print(rel)
    if report["hinted_non_python_files"]:
        print("\n[hinted_non_python_files]")
        for item in report["hinted_non_python_files"]:
            print(item["path"])
    if report["parse_errors"]:
        print("\n[parse_errors]")
        for error in report["parse_errors"]:
            print(error)
        if args.fail_on_parse_error:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
