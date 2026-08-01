from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.embedding_registry import resolve_embedding_profile
from tools.migration_shadow_builder import MigrationShadowBuilder
from tools.migration_types import MigrationTask

SOURCE = "a" * 64


def task(profile_name="minilm-l6-v2", profile_fingerprint=None):
    profile = resolve_embedding_profile(profile_name, allow_compatibility=False)
    return MigrationTask(
        task_id="e" * 64,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=SOURCE,
        target_profile_name=profile.alias,
        target_profile_fingerprint=profile_fingerprint or profile.fingerprint,
        state="running",
        attempt=1,
        created_at=1.0,
        updated_at=2.0,
        lease_owner="worker",
        lease_expires_at=100.0,
    )


class Registry:
    def __init__(self, root: Path, source: Path | None):
        self.upload_root = root
        self.source = source

    def get(self, *, owner_id, doc_id):
        if self.source is None:
            return None
        return {
            "owner_id": owner_id,
            "doc_id": doc_id,
            "source_retained": True,
            "source_path": str(self.source),
        }


class Encoder:
    def __init__(self, profile, *, rows=None):
        self.profile = profile
        self.rows = rows
        self.passages = None

    def encode_passages(self, passages):
        self.passages = tuple(passages)
        if self.rows is not None:
            return self.rows
        dimensions = self.profile.dimensions or 2
        return tuple((0.1,) * dimensions for _ in passages)


def source(tmp_path):
    root = tmp_path / "uploads"
    directory = root / "alice"
    directory.mkdir(parents=True)
    path = directory / "paper.pdf"
    path.write_bytes(b"retained bytes")
    return root, path


def document(identifier="doc-1"):
    return SimpleNamespace(
        id=identifier,
        text="Privacy finalized body text.",
        title="Paper title",
        filename="paper.pdf",
        metadata={
            "parser": "pdf",
            "parser_version": "1",
            "redaction": "best_effort_regex_masking",
            "document_identity": "owner_and_source_sha256",
        },
        sections=[
            SimpleNamespace(
                title="Abstract",
                content="Abstract evidence.",
                page_number=1,
                metadata={"field_type": "abstract"},
            )
        ],
    )


def test_builder_creates_one_to_one_vector_sparse_provenance(tmp_path):
    root, path = source(tmp_path)
    parsed = document()
    encoders = []

    def encoder_factory(profile):
        value = Encoder(profile)
        encoders.append(value)
        return value

    builder = MigrationShadowBuilder(
        registry=Registry(root, path),
        parser=lambda source_path, owner_id: parsed,
        encoder_factory=encoder_factory,
    )
    result = builder(task())
    assert len(result.vector_rows) == len(result.sparse_rows) == 3
    assert len(result.content_sha256) == 64
    assert len(result.parser_fingerprint) == 64
    assert encoders[0].passages == tuple(row["text"] for row in result.sparse_rows)
    for vector, sparse in zip(result.vector_rows, result.sparse_rows, strict=True):
        assert vector["row_id"] == sparse["field_id"]
        assert vector["metadata"]["field_id"] == sparse["field_id"]
        assert vector["metadata"]["owner_id"] == "alice"
        assert vector["metadata"]["doc_id"] == "doc-1"
        assert vector["metadata"]["source_sequence"] == 4
        assert vector["metadata"]["target_profile_name"] == "minilm-l6-v2"
        assert len(vector["embedding"]) == 384
    serialized = json.dumps(
        {
            "vectors": result.vector_rows,
            "sparse": result.sparse_rows,
        }
    )
    assert str(path) not in serialized
    assert "source_path" not in serialized


def test_profile_fingerprint_source_and_document_identity_fail_closed(tmp_path):
    root, path = source(tmp_path)
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        MigrationShadowBuilder(
            registry=Registry(root, path),
            parser=lambda source_path, owner_id: document(),
            encoder_factory=lambda profile: Encoder(profile),
        )(task(profile_fingerprint="f" * 64))
    with pytest.raises(RuntimeError, match="source is unavailable"):
        MigrationShadowBuilder(
            registry=Registry(root, None),
            parser=lambda source_path, owner_id: document(),
            encoder_factory=lambda profile: Encoder(profile),
        )(task())
    with pytest.raises(RuntimeError, match="identity changed"):
        MigrationShadowBuilder(
            registry=Registry(root, path),
            parser=lambda source_path, owner_id: document("different"),
            encoder_factory=lambda profile: Encoder(profile),
        )(task())


def test_custom_encoder_wrong_rows_dimensions_and_nonfinite_fail_closed(tmp_path):
    root, path = source(tmp_path)
    profile = resolve_embedding_profile("minilm-l6-v2", allow_compatibility=False)
    with pytest.raises(RuntimeError, match="wrong row count"):
        MigrationShadowBuilder(
            registry=Registry(root, path),
            parser=lambda source_path, owner_id: document(),
            encoder_factory=lambda selected: Encoder(selected, rows=()),
        )(task())
    with pytest.raises(RuntimeError, match="dimensions"):
        MigrationShadowBuilder(
            registry=Registry(root, path),
            parser=lambda source_path, owner_id: document(),
            encoder_factory=lambda selected: Encoder(
                selected,
                rows=((0.0, 1.0),) * 3,
            ),
        )(task())
    with pytest.raises(RuntimeError, match="non-finite"):
        MigrationShadowBuilder(
            registry=Registry(root, path),
            parser=lambda source_path, owner_id: document(),
            encoder_factory=lambda selected: Encoder(
                selected,
                rows=((float("nan"),) * profile.dimensions,) * 3,
            ),
        )(task())


def test_inferred_dimensions_must_remain_consistent(tmp_path, monkeypatch):
    root, path = source(tmp_path)
    compatibility = SimpleNamespace(
        alias="custom",
        fingerprint="b" * 64,
        dimensions=None,
    )
    monkeypatch.setattr(
        "tools.migration_shadow_builder.resolve_embedding_profile",
        lambda *args, **kwargs: compatibility,
    )
    selected_task = MigrationTask(
        task_id="e" * 64,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=SOURCE,
        target_profile_name="custom",
        target_profile_fingerprint="b" * 64,
        state="running",
        attempt=1,
        created_at=1.0,
        updated_at=2.0,
        lease_owner="worker",
        lease_expires_at=100.0,
    )
    encoder = SimpleNamespace(
        profile=compatibility,
        encode_passages=lambda passages: ((0.0, 1.0), (0.0,), (0.0, 1.0)),
    )
    with pytest.raises(RuntimeError, match="dimensions"):
        MigrationShadowBuilder(
            registry=Registry(root, path),
            parser=lambda source_path, owner_id: document(),
            encoder_factory=lambda profile: encoder,
        )(selected_task)
