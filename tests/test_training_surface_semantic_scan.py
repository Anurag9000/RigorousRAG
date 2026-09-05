from __future__ import annotations

import json
import runpy
from pathlib import Path

_MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools" / "training_surface_semantic_scan.py")
)
scan = _MODULE["scan"]


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_multilanguage_learner_and_launcher_categories(tmp_path: Path) -> None:
    _write(tmp_path, "models/train.py", "def fit_model(x):\n    return x\n")
    _write(tmp_path, "web/train.ts", "await model.fit(xs, ys);\n")
    _write(tmp_path, "analysis/model.R", "result <- caret::train(x, y)\n")
    _write(tmp_path, "julia/fit.jl", "Flux.train!(loss, model, data, opt)\n")
    _write(tmp_path, "scripts/run.sh", "torchrun --nproc_per_node=2 training/train.py\n")
    _write(
        tmp_path,
        ".github/workflows/train.yml",
        "name: train\njobs:\n  x:\n    steps:\n      - run: python run_all_training.py\n",
    )

    report = scan(tmp_path)

    assert "models/train.py" in report["learner_files"]
    assert "web/train.ts" in report["learner_files"]
    assert "analysis/model.R" in report["learner_files"]
    assert "julia/fit.jl" in report["learner_files"]
    assert "scripts/run.sh" in report["launcher_files"]
    assert ".github/workflows/train.yml" in report["workflow_files"]


def test_generic_application_fit_and_comments_are_not_javascript_learners(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "ui/layout.ts",
        "// model.fit(xs, ys) is documentation only\nbox.fit(container);\n",
    )
    report = scan(tmp_path)
    assert "ui/layout.ts" not in report["learner_files"]


def test_notebook_code_cells_are_semantically_scanned(tmp_path: Path) -> None:
    notebook = {
        "cells": [
            {"cell_type": "markdown", "source": ["fit_model() is prose"]},
            {"cell_type": "code", "source": ["def train_epoch():\n", "    return 1\n"]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    _write(tmp_path, "research/experiment.ipynb", json.dumps(notebook))
    report = scan(tmp_path)
    assert "research/experiment.ipynb" in report["learner_files"]


def test_parse_errors_are_reported_deterministically(tmp_path: Path) -> None:
    _write(tmp_path, "broken.py", "def train_model(:\n    pass\n")
    report = scan(tmp_path)
    assert report["parse_errors"]
    assert any("broken.py" in item for item in report["parse_errors"])
