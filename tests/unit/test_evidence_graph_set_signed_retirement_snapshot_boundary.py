from __future__ import annotations

import os

import pytest

from tools import evidence_graph_set_signed_retirement_snapshot_boundary as boundary
from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    export_signed_retirement_snapshot,
)


class Journal:
    def list(self, **kwargs):
        return (
            SignedPublicationRetirementAttempt.create(
                owner_id="alice",
                publication_operation_id="1" * 64,
                graph_set_key="review",
                signed_candidate_set_id="2" * 64,
                signed_candidate_set_digest="3" * 64,
                authorization_candidate_set_id="4" * 64,
                signed_authority_digest="5" * 64,
                now=1.0,
            ),
        )


def snapshot(tmp_path):
    path = tmp_path / "retirements.json"
    export_signed_retirement_snapshot(
        owner_id="alice",
        journal=Journal(),
        output_path=path,
        now=10.0,
        limit=100,
    )
    return path


def test_descriptor_boundary_verifies_valid_snapshot(tmp_path):
    value = boundary.verify_signed_retirement_snapshot(snapshot(tmp_path))
    assert value.owner_id == "alice"
    assert value.record_count == 1


def test_descriptor_boundary_detects_file_growth_during_read(
    tmp_path, monkeypatch
):
    path = snapshot(tmp_path)
    original_read = boundary.os.read
    calls = 0

    def growing_read(descriptor, count):
        nonlocal calls
        calls += 1
        data = original_read(descriptor, count)
        if calls == 1:
            with path.open("ab") as stream:
                stream.write(b"x")
                stream.flush()
                os.fsync(stream.fileno())
        return data

    monkeypatch.setattr(boundary.os, "read", growing_read)
    with pytest.raises(RuntimeError, match="grew|identity changed"):
        boundary.verify_signed_retirement_snapshot(path)


def test_descriptor_boundary_reads_opened_inode_when_path_is_replaced(
    tmp_path, monkeypatch
):
    path = snapshot(tmp_path)
    original_read = boundary.os.read
    calls = 0

    def replacing_read(descriptor, count):
        nonlocal calls
        calls += 1
        data = original_read(descriptor, count)
        if calls == 1:
            replacement = tmp_path / "replacement.json"
            replacement.write_text("{}", encoding="utf-8")
            os.replace(replacement, path)
        return data

    monkeypatch.setattr(boundary.os, "read", replacing_read)
    value = boundary.verify_signed_retirement_snapshot(path)
    assert value.owner_id == "alice"
    assert value.record_count == 1
