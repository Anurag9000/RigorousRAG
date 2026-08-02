from __future__ import annotations

import os

import pytest

from tools.evidence_graph_relation_actor import (
    ReviewActorBinding,
    load_relation_review_actor,
    require_relation_review_actor,
)


def test_environment_actor_binding_is_deterministic_and_required(monkeypatch):
    monkeypatch.setenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID", "reviewer-1")
    monkeypatch.delenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH", raising=False)

    first = load_relation_review_actor(loaded_at=1.0)
    second = load_relation_review_actor(loaded_at=2.0)

    assert first.actor_id == "reviewer-1"
    assert first.binding_method == "process_environment"
    assert first.binding_digest == second.binding_digest
    assert first.loaded_at == 1.0
    assert second.loaded_at == 2.0
    assert require_relation_review_actor("reviewer-1", binding=first) == first
    with pytest.raises(PermissionError, match="process-owned"):
        require_relation_review_actor("reviewer-2", binding=first)


def test_descriptor_file_actor_binding_rejects_redirects(tmp_path, monkeypatch):
    path = tmp_path / "reviewer.txt"
    path.write_text("reviewer-file", encoding="utf-8")
    monkeypatch.delenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID", raising=False)
    monkeypatch.setenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH", str(path))

    value = load_relation_review_actor(loaded_at=3.0)

    assert value.actor_id == "reviewer-file"
    assert value.binding_method == "descriptor_file"
    assert len(value.binding_digest) == 64

    link = tmp_path / "reviewer-link.txt"
    try:
        link.symlink_to(path)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable on this platform")
    with pytest.raises(ValueError, match="redirects"):
        load_relation_review_actor(path=link)


def test_actor_source_selection_and_input_bounds_fail_closed(
    tmp_path, monkeypatch
):
    path = tmp_path / "reviewer.txt"
    path.write_text("reviewer-file", encoding="utf-8")
    monkeypatch.setenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID", "reviewer-env")
    monkeypatch.setenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH", str(path))
    with pytest.raises(RuntimeError, match="multiple"):
        load_relation_review_actor()

    monkeypatch.delenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID", raising=False)
    monkeypatch.delenv("EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH", raising=False)
    with pytest.raises(RuntimeError, match="not configured"):
        load_relation_review_actor()

    with pytest.raises(ValueError, match="invalid"):
        load_relation_review_actor(actor_id="bad\nactor")

    path.write_bytes(b"x" * 4097)
    with pytest.raises(ValueError, match="too large"):
        load_relation_review_actor(path=path)


def test_binding_reconstruction_detects_tampering():
    value = ReviewActorBinding.create(
        actor_id="reviewer-1",
        binding_method="process_environment",
        loaded_at=1.0,
    )
    with pytest.raises(ValueError, match="differs"):
        ReviewActorBinding(
            actor_id="reviewer-2",
            binding_method=value.binding_method,
            binding_digest=value.binding_digest,
            loaded_at=value.loaded_at,
        )
    with pytest.raises(ValueError, match="unsupported"):
        ReviewActorBinding.create(
            actor_id="reviewer-1",
            binding_method="command_line",
            loaded_at=1.0,
        )
