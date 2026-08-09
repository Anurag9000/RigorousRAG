from __future__ import annotations

from tools.agent_stopping import AgentStoppingSignals, decide_agent_stopping


def signals(**overrides):
    values = {
        "evidence_sufficiency": 0.9,
        "answer_confidence": 0.85,
        "agent_disagreement": 0.1,
        "contradiction_risk": 0.1,
        "uncertainty": 0.2,
        "marginal_improvement": 0.02,
        "remaining_budget_fraction": 0.5,
        "external_currentness_required": False,
        "current_evidence_fraction": 1.0,
    }
    values.update(overrides)
    return AgentStoppingSignals(**values)


def test_stops_when_evidence_confidence_currentness_and_consensus_are_sufficient():
    decision = decide_agent_stopping(signals())
    assert decision.action == "stop"
    assert decision.reasons == ("evidence_and_confidence_sufficient",)
    assert decision.confidence_to_stop > 0.0
    assert len(decision.decision_digest) == 64
    assert decision == decide_agent_stopping(signals())


def test_disagreement_contradiction_or_stale_currentness_escalates():
    disagreement = decide_agent_stopping(signals(agent_disagreement=0.8))
    contradiction = decide_agent_stopping(signals(contradiction_risk=0.8))
    stale = decide_agent_stopping(
        signals(
            external_currentness_required=True,
            current_evidence_fraction=0.2,
        )
    )
    assert disagreement.action == "escalate"
    assert contradiction.action == "escalate"
    assert stale.action == "escalate"
    assert "current_evidence_insufficient" in stale.reasons


def test_low_budget_abstains_when_sufficiency_or_risk_is_unresolved():
    insufficient = decide_agent_stopping(
        signals(
            evidence_sufficiency=0.2,
            answer_confidence=0.2,
            remaining_budget_fraction=0.01,
        )
    )
    conflict = decide_agent_stopping(
        signals(
            agent_disagreement=0.9,
            remaining_budget_fraction=0.01,
        )
    )
    assert insufficient.action == "abstain"
    assert conflict.action == "abstain"
    assert "budget_exhausted_with_unresolved_risk" in conflict.reasons


def test_plateau_below_sufficiency_escalates_but_improving_case_continues():
    plateau = decide_agent_stopping(
        signals(
            evidence_sufficiency=0.5,
            answer_confidence=0.6,
            marginal_improvement=0.0,
        )
    )
    improving = decide_agent_stopping(
        signals(
            evidence_sufficiency=0.5,
            answer_confidence=0.6,
            marginal_improvement=0.2,
        )
    )
    assert plateau.action == "escalate"
    assert improving.action == "continue"
