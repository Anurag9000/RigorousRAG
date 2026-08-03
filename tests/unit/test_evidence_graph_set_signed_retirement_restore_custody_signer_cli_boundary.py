from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_signer_cli_boundary as boundary,
)


def binding(method: str):
    return SimpleNamespace(
        actor_id="operator",
        binding_method=method,
        binding_digest="1" * 64,
    )


def test_direct_process_owned_actor_methods_are_allowed(monkeypatch):
    monkeypatch.setattr(
        boundary,
        "_ORIGINAL_REQUIRE",
        lambda requested, *, binding: binding,
    )
    for method in ("process_environment", "descriptor_file"):
        value = binding(method)
        assert boundary._require_direct_signer_admin_actor(
            "operator",
            binding=value,
        ) is value


def test_every_non_direct_actor_method_fails_closed(monkeypatch):
    monkeypatch.setattr(
        boundary,
        "_ORIGINAL_REQUIRE",
        lambda requested, *, binding: binding,
    )
    for method in (
        "signed_assertion",
        "hmac_assertion",
        "oidc_assertion",
        "command_line",
        "future_binding_method",
    ):
        with pytest.raises(PermissionError, match="direct process-owned"):
            boundary._require_direct_signer_admin_actor(
                "operator",
                binding=binding(method),
            )


def test_boundary_main_delegates_after_installing_restriction(monkeypatch):
    observed = {}

    def run(argv):
        observed["argv"] = argv
        assert (
            boundary._base.require_relation_review_actor
            is boundary._require_direct_signer_admin_actor
        )
        return 7

    monkeypatch.setattr(boundary._base, "main", run)
    assert boundary.main(["status"]) == 7
    assert observed["argv"] == ["status"]
