"""Private-safe verification of stored research capsules against durable authorities.

The verifier never decrypts replay recipes and never materializes raw query/evidence bytes.
Each reference digest is re-derived from the durable object that owns that identity. This
makes verification meaningful rather than comparing a capsule hash to another copy of the
same capsule hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from tools.capsule_replay import CapsuleVerificationReceipt, verify_capsule
from tools.research_capsule import CapsuleReference
from tools.research_capsule_store import StoredResearchCapsule
from tools.research_result_provenance import session_binding, sha256_text
from tools.research_result_store import ResearchResultStore, StoredResearchResult
from tools.research_workspace import ResearchProject, ResearchSession, ResearchTurn

_RUNTIME_KEY = "_rigorousrag_runtime"
_CITATION_DIGEST_KEYS = (
    "content_sha256",
    "evidence_sha256",
    "chunk_sha256",
    "document_sha256",
    "source_sha256",
)


class WorkspaceStore(Protocol):
    def get_project(self, owner_id: str, project_id: str) -> ResearchProject: ...
    def get_session(self, owner_id: str, session_id: str) -> ResearchSession: ...


def _sha(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        return ""
    return digest


def _revision(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    revision = value.strip().lower()
    if len(revision) not in {40, 64}:
        return ""
    if any(ch not in "0123456789abcdef" for ch in revision):
        return ""
    return revision


def _citation_digest(citation: Any) -> str:
    metadata = getattr(citation, "metadata", None)
    if not isinstance(metadata, Mapping):
        return ""
    for key in _CITATION_DIGEST_KEYS:
        digest = _sha(metadata.get(key))
        if digest:
            return digest
    return ""


def _turn_and_prefix(
    session: ResearchSession,
    result: StoredResearchResult,
) -> tuple[ResearchTurn, ResearchSession] | None:
    matches = [
        (index, turn)
        for index, turn in enumerate(session.turns)
        if turn.result_sha256 == result.result_id and turn.query_sha256 == result.query_sha256
    ]
    if len(matches) != 1:
        return None
    index, turn = matches[0]
    prefix = ResearchSession(
        owner_id=session.owner_id,
        project_id=session.project_id,
        session_id=session.session_id,
        turns=session.turns[:index],
        created_at=session.created_at,
        closed_at=None,
    )
    return turn, prefix


class ResearchCapsuleDigestAuthority:
    """Re-derive capsule reference digests from Workspace and immutable result state."""

    def __init__(
        self,
        stored: StoredResearchCapsule,
        *,
        workspace_store: WorkspaceStore,
        result_store: ResearchResultStore,
    ) -> None:
        if not isinstance(stored, StoredResearchCapsule):
            raise TypeError("stored must be StoredResearchCapsule")
        if not isinstance(result_store, ResearchResultStore):
            raise TypeError("result_store must be ResearchResultStore")
        self.stored = stored
        self.project = workspace_store.get_project(stored.owner_id, stored.project_id)
        self.session = workspace_store.get_session(stored.owner_id, stored.session_id)
        self.result = result_store.get(stored.owner_id, stored.result_id)
        if self.project.project_id != stored.project_id:
            raise RuntimeError("capsule project authority mismatch")
        if self.session.project_id != stored.project_id:
            raise RuntimeError("capsule session authority mismatch")
        if self.result.result_id != stored.result_id:
            raise RuntimeError("capsule result authority mismatch")
        self._turn_prefix = _turn_and_prefix(self.session, self.result)
        self._runtime = self.result.metadata.get(_RUNTIME_KEY)
        if not isinstance(self._runtime, Mapping):
            self._runtime = {}
        self._corpora = {item.corpus_id: item for item in self.project.corpora}

    def _session_digest(self, reference: CapsuleReference) -> str | None:
        if self._turn_prefix is None:
            return None
        _, prefix = self._turn_prefix
        binding = session_binding(self.result.metadata)
        if binding is None or binding["session_fingerprint_before"] != prefix.fingerprint:
            return None
        if reference.ref_id == "session-execution-snapshot":
            return prefix.fingerprint
        # New manifests identify the session with the immutable pre-execution snapshot.
        if reference.metadata.get("snapshot") == "pre_execution":
            return prefix.fingerprint
        if reference.content_sha256 == prefix.fingerprint:
            return prefix.fingerprint
        # Compatibility for older manifests that captured the then-current full session.
        # This can only be proven while that exact full session version is still current.
        if reference.content_sha256 == self.session.fingerprint:
            return self.session.fingerprint
        return None

    def _corpus_digest(self, ref_id: str) -> str | None:
        prefix = "corpus:"
        generation_suffix = ":generation"
        profile_suffix = ":retrieval-profile"
        if not ref_id.startswith(prefix):
            return None
        if ref_id.endswith(generation_suffix):
            corpus_id = ref_id[len(prefix) : -len(generation_suffix)]
            corpus = self._corpora.get(corpus_id)
            return corpus.generation_sha256 if corpus is not None else None
        if ref_id.endswith(profile_suffix):
            corpus_id = ref_id[len(prefix) : -len(profile_suffix)]
            corpus = self._corpora.get(corpus_id)
            return corpus.retrieval_profile_sha256 if corpus is not None else None
        return None

    def _capability_digest(self, reference: CapsuleReference) -> str | None:
        role = reference.ref_id[len("capability:") :]
        selected = self._runtime.get("selected_capabilities")
        if not isinstance(selected, Mapping):
            return None
        raw = selected.get(role)
        if not isinstance(raw, Mapping):
            return None
        capability_id = str(raw.get("capability_id") or "")
        expected_id = reference.metadata.get("capability_id", "")
        if expected_id and expected_id != capability_id:
            return None
        version = str(raw.get("version") or "")
        if reference.version and reference.version != version:
            return None
        digest = _sha(raw.get("fingerprint"))
        return digest or None

    def _citation_digest(self, reference: CapsuleReference) -> str | None:
        try:
            index = int(reference.ref_id.split(":", 1)[1])
        except (ValueError, IndexError):
            return None
        if index < 0 or index >= len(self.result.citations):
            return None
        citation_id = self.result.citation_ids[index]
        expected_id = reference.metadata.get("citation_id", "")
        if expected_id and expected_id != citation_id:
            return None
        digest = _citation_digest(self.result.citations[index])
        return digest or None

    def digest(self, reference: CapsuleReference) -> str | None:
        if not isinstance(reference, CapsuleReference):
            raise TypeError("reference must be CapsuleReference")
        ref_id = reference.ref_id
        if ref_id == "project":
            return self.project.fingerprint
        if ref_id in {"session", "session-execution-snapshot"}:
            return self._session_digest(reference)
        if ref_id == "query":
            return self.result.query_sha256
        if ref_id == "result":
            return self.result.result_id
        if ref_id == "runtime-config":
            return _sha(self._runtime.get("runtime_config_sha256")) or None
        if ref_id == "capability-registry":
            return _sha(self._runtime.get("capability_registry_sha256")) or None
        if ref_id == "model-identifier":
            if not self.result.model:
                return None
            digest = sha256_text(self.result.model)
            bound = _sha(self._runtime.get("model_identifier_sha256"))
            return digest if bound == digest else None
        if ref_id == "policy":
            if self._turn_prefix is None:
                return None
            turn, _ = self._turn_prefix
            digest = _sha(turn.policy_sha256)
            return digest or None
        if ref_id.startswith("corpus:"):
            return self._corpus_digest(ref_id)
        if ref_id.startswith("capability:"):
            return self._capability_digest(reference)
        if ref_id.startswith("citation:"):
            return self._citation_digest(reference)
        return None


@dataclass(frozen=True)
class StoredCapsuleVerification:
    receipt: CapsuleVerificationReceipt
    code_revision_status: str
    deployment_code_revision: str
    manifest_verified: bool
    deployment_compatible: bool


def verify_stored_capsule(
    stored: StoredResearchCapsule,
    *,
    workspace_store: WorkspaceStore,
    result_store: ResearchResultStore,
    deployment_code_revision: str = "",
) -> StoredCapsuleVerification:
    authority = ResearchCapsuleDigestAuthority(
        stored,
        workspace_store=workspace_store,
        result_store=result_store,
    )
    receipt = verify_capsule(stored.capsule, authority=authority)
    deployment_revision = _revision(deployment_code_revision)
    capsule_revision = _revision(stored.capsule.code_revision)
    if not deployment_revision:
        code_status = "unavailable"
    elif deployment_revision == capsule_revision:
        code_status = "matched"
    else:
        code_status = "mismatch"
    return StoredCapsuleVerification(
        receipt=receipt,
        code_revision_status=code_status,
        deployment_code_revision=deployment_revision,
        manifest_verified=receipt.verified,
        deployment_compatible=receipt.verified and code_status == "matched",
    )


__all__ = [
    "ResearchCapsuleDigestAuthority",
    "StoredCapsuleVerification",
    "WorkspaceStore",
    "verify_stored_capsule",
]
