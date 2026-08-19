"""Advisory inference over trained grounded-generation auxiliary heads.

This scorer makes citation/support/contradiction/abstention/reflection heads usable after
artifact export. Its outputs are *signals*, not authority: citation IDs still must be drawn
from the server-owned released evidence universe and final output must pass the repository's
existing citation refinement, abstention, closed-schema and publication boundaries.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from training.advanced_rag_data import GroundedClaimAnnotation, GroundedCollatorConfig, GroundedEvidenceRecord, GroundedGenerationExample, TextSpan
from training.advanced_rag_final_collation import FinalCausalGroundedCollator, FinalSeq2SeqGroundedCollator
from training.advanced_rag_runtime_loading import LoadedGroundedArtifact
from training.grounded_generation import ReflectionAction

_MAX_CLAIMS = 4096
_MAX_EVIDENCE = 4096


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("grounded artifact scoring requires optional PyTorch")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


@dataclass(frozen=True)
class AdvisoryClaimGroundingScore:
    claim_start: int
    claim_end: int
    evidence_probabilities: Mapping[str, float]
    support_probability: float
    contradiction_probability: float

    def __post_init__(self) -> None:
        if isinstance(self.claim_start, bool) or isinstance(self.claim_end, bool) or not isinstance(self.claim_start, int) or not isinstance(self.claim_end, int) or self.claim_start < 0 or self.claim_end <= self.claim_start:
            raise ValueError("claim span is invalid")
        probabilities = {str(key): _finite(value, f"evidence probability {key}") for key, value in self.evidence_probabilities.items()}
        if not probabilities or any(value < 0.0 or value > 1.0 for value in probabilities.values()):
            raise ValueError("evidence probabilities must be non-empty and lie in [0,1]")
        if abs(sum(probabilities.values()) - 1.0) > 1e-4:
            raise ValueError("evidence probabilities must sum to one")
        object.__setattr__(self, "evidence_probabilities", probabilities)
        for name in ("support_probability", "contradiction_probability"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GroundedAdvisoryScoringResult:
    artifact_sha256: str
    request_sha256: str
    answer_sha256: str
    evidence_universe_sha256: str
    claims: tuple[AdvisoryClaimGroundingScore, ...]
    abstention_probability: float
    reflection_probabilities: Mapping[str, float]
    result_sha256: str

    def __post_init__(self) -> None:
        for name in ("artifact_sha256", "request_sha256", "answer_sha256", "evidence_universe_sha256", "result_sha256"):
            value = str(getattr(self, name)).strip().lower()
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"{name} must be SHA-256")
            object.__setattr__(self, name, value)
        claims = tuple(self.claims)
        if not claims or len(claims) > _MAX_CLAIMS or any(not isinstance(item, AdvisoryClaimGroundingScore) for item in claims):
            raise ValueError("claims must be a bounded non-empty advisory score sequence")
        object.__setattr__(self, "claims", claims)
        abstain = _finite(self.abstention_probability, "abstention_probability")
        if not 0.0 <= abstain <= 1.0:
            raise ValueError("abstention_probability must lie in [0,1]")
        object.__setattr__(self, "abstention_probability", abstain)
        reflections = {str(key): _finite(value, f"reflection probability {key}") for key, value in self.reflection_probabilities.items()}
        if not reflections or any(value < 0.0 or value > 1.0 for value in reflections.values()) or abs(sum(reflections.values()) - 1.0) > 1e-4:
            raise ValueError("reflection probabilities must form a distribution")
        object.__setattr__(self, "reflection_probabilities", reflections)
        if _digest(self._payload()) != self.result_sha256:
            raise ValueError("grounded advisory scoring result digest mismatch")

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-grounded-advisory-scoring-result/v1",
            "artifact_sha256": self.artifact_sha256,
            "request_sha256": self.request_sha256,
            "answer_sha256": self.answer_sha256,
            "evidence_universe_sha256": self.evidence_universe_sha256,
            "claims": [asdict(item) for item in self.claims],
            "abstention_probability": self.abstention_probability,
            "reflection_probabilities": dict(self.reflection_probabilities),
        }


@dataclass(frozen=True)
class GroundedAdvisoryScoringConfig:
    sequence_max_length: int = 4096
    evidence_max_length: int = 768
    evidence_limit: int = 64
    claim_limit: int = 128

    def __post_init__(self) -> None:
        for name, maximum in (("sequence_max_length", 1_000_000), ("evidence_max_length", 1_000_000), ("evidence_limit", _MAX_EVIDENCE), ("claim_limit", _MAX_CLAIMS)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} is out of bounds")

    @property
    def config_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-grounded-advisory-scoring-config/v1", **asdict(self)})


class GroundedArtifactAdvisoryScorer:
    def __init__(self, loaded: LoadedGroundedArtifact, config: GroundedAdvisoryScoringConfig = GroundedAdvisoryScoringConfig()) -> None:
        if not isinstance(loaded, LoadedGroundedArtifact):
            raise ValueError("loaded must be LoadedGroundedArtifact")
        if not isinstance(config, GroundedAdvisoryScoringConfig):
            raise ValueError("config must be GroundedAdvisoryScoringConfig")
        self.loaded, self.config = loaded, config

    @property
    def contract_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-grounded-advisory-scorer/v1", "artifact_sha256": self.loaded.manifest.artifact_sha256, "config_sha256": self.config.config_sha256})

    def score(
        self,
        *,
        request_text: str,
        prompt: str,
        answer: str,
        claim_spans: Sequence[TextSpan],
        evidence: Sequence[GroundedEvidenceRecord],
    ) -> GroundedAdvisoryScoringResult:
        _require_torch()
        if not isinstance(request_text, str) or not request_text or "\x00" in request_text:
            raise ValueError("request_text must be non-empty text")
        if not isinstance(prompt, str) or not prompt or not isinstance(answer, str) or not answer:
            raise ValueError("prompt and answer must be non-empty text")
        spans = tuple(claim_spans)
        records = tuple(evidence)
        if not spans or len(spans) > self.config.claim_limit or any(not isinstance(span, TextSpan) or span.end > len(answer) for span in spans):
            raise ValueError("claim_spans are invalid, empty or exceed scoring limit")
        if not records or len(records) > self.config.evidence_limit or any(not isinstance(item, GroundedEvidenceRecord) for item in records):
            raise ValueError("evidence is invalid, empty or exceeds scoring limit")
        if len({item.evidence_id for item in records}) != len(records):
            raise ValueError("evidence ids must be unique")
        example = GroundedGenerationExample(
            example_id="advisory-inference",
            prompt=prompt,
            answer=answer,
            evidence=records,
            claims=tuple(GroundedClaimAnnotation(span=span) for span in spans),
            reflection_action=ReflectionAction.STOP,
        )
        collator_config = GroundedCollatorConfig(
            sequence_max_length=self.config.sequence_max_length,
            evidence_max_length=self.config.evidence_max_length,
            evidence_limit=self.config.evidence_limit,
            claim_limit=self.config.claim_limit,
        )
        collator_cls = FinalSeq2SeqGroundedCollator if self.loaded.manifest.generator_family == "seq2seq_lm" else FinalCausalGroundedCollator
        batch = collator_cls(self.loaded.tokenizer, collator_config)([example])
        try:
            device = next(self.loaded.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        model_inputs = {
            key: ({nested: tensor.to(device) if torch.is_tensor(tensor) else tensor for nested, tensor in value.items()} if isinstance(value, Mapping) else value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch["model_inputs"].items()
        }
        self.loaded.model.eval()
        with torch.no_grad():
            output = self.loaded.model(**model_inputs)
        citation = output["citation_logits"][0, :len(spans), :len(records)].detach().float().cpu()
        support = torch.sigmoid(output["support_logits"][0, :len(spans)].detach().float().cpu())
        contradiction = torch.sigmoid(output["contradiction_logits"][0, :len(spans)].detach().float().cpu())
        abstention = float(torch.sigmoid(output["abstention_logits"][0].detach().float().cpu()))
        reflection = torch.softmax(output["reflection_logits"][0].detach().float().cpu(), dim=-1)
        claim_results = []
        for index, span in enumerate(spans):
            probabilities = torch.softmax(citation[index], dim=-1).tolist()
            claim_results.append(AdvisoryClaimGroundingScore(
                claim_start=span.start,
                claim_end=span.end,
                evidence_probabilities={record.evidence_id: float(probabilities[position]) for position, record in enumerate(records)},
                support_probability=float(support[index]),
                contradiction_probability=float(contradiction[index]),
            ))
        actions = self.loaded.model.config.reflection_actions
        reflection_map = {action.value: float(reflection[index]) for index, action in enumerate(actions)}
        request_sha = hashlib.sha256(request_text.encode("utf-8")).hexdigest()
        answer_sha = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        evidence_sha = _digest({"evidence": [{"id": item.evidence_id, "text_sha256": hashlib.sha256(item.text.encode("utf-8")).hexdigest(), "source_id": item.source_id} for item in records]})
        unsigned = {
            "schema": "rigorousrag-grounded-advisory-scoring-result/v1",
            "artifact_sha256": self.loaded.manifest.artifact_sha256,
            "request_sha256": request_sha,
            "answer_sha256": answer_sha,
            "evidence_universe_sha256": evidence_sha,
            "claims": [asdict(item) for item in claim_results],
            "abstention_probability": abstention,
            "reflection_probabilities": reflection_map,
        }
        return GroundedAdvisoryScoringResult(
            artifact_sha256=self.loaded.manifest.artifact_sha256,
            request_sha256=request_sha,
            answer_sha256=answer_sha,
            evidence_universe_sha256=evidence_sha,
            claims=tuple(claim_results),
            abstention_probability=abstention,
            reflection_probabilities=reflection_map,
            result_sha256=_digest(unsigned),
        )


__all__ = [
    "AdvisoryClaimGroundingScore", "GroundedAdvisoryScoringConfig",
    "GroundedAdvisoryScoringResult", "GroundedArtifactAdvisoryScorer",
]
