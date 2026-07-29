import itertools

import pytest

from tools.models import AgentAnswer, Citation
from tools.verification import audit_hallucination, verify_citations


def citation(label="[1]", snippet="detailed evidence about alpha beta gamma delta epsilon zeta eta theta"):
    return Citation(
        label=label,
        title="Source",
        url="https://example.test/source",
        source_type="web_page",
        snippet=snippet,
    )


def test_all_bracket_labels_are_structurally_checked():
    issues = verify_citations("Claim [doc-1] and other claim [2].", [citation("[doc-1]")])
    assert any(issue["type"] == "missing_source" and issue["label"] == "[2]" for issue in issues)


def test_missing_markers_and_unused_sources_are_reported():
    issues = verify_citations("An uncited answer.", [citation()])
    assert any(issue["type"] == "missing_markers" for issue in issues)
    assert any(issue["type"] == "unused_source" for issue in issues)


def test_duplicate_labels_are_rejected_on_direct_verification():
    issues = verify_citations("Claim [1].", [citation(), citation()])
    assert any(issue["type"] == "duplicate_labels" for issue in issues)


def test_short_evidence_is_diagnostic_not_silently_passed():
    answer = AgentAnswer(
        answer="Claim [1].",
        citations=[citation(snippet="tiny evidence")],
    )

    issues = verify_citations(answer.answer, answer.citations)
    assert any(
        issue["type"] == "weak_evidence_text"
        and issue["evidence_token_count"] < 8
        for issue in issues
    )
    message = audit_hallucination(answer)
    assert "too short" in message
    assert "manual source inspection" in message
    assert "passed" not in message


def test_no_evidence_is_not_reported_as_a_pass():
    answer = AgentAnswer(answer="An unsupported response.")

    issues = verify_citations(answer.answer, answer.citations)
    message = audit_hallucination(answer)

    assert issues == [
        {
            "type": "no_evidence",
            "label": "*",
            "error": (
                "No citations or citation markers were supplied; citation structure "
                "could not be evaluated."
            ),
        }
    ]
    assert "no evidence" in message.lower()
    assert "passed" not in message.lower()


def test_marker_scan_and_issue_count_are_hard_bounded():
    overflow_answer = " ".join("[1]" for _ in range(1001))
    overflow_issues = verify_citations(overflow_answer, [citation()])
    assert any(issue["type"] == "too_many_markers" for issue in overflow_issues)

    unique_markers = " ".join(f"[m{index}]" for index in range(1000))
    issues = verify_citations(unique_markers, [])
    assert len(issues) == 500
    assert issues[-1]["type"] == "issue_limit_reached"


def test_direct_answer_and_citation_inputs_are_bounded_before_iteration():
    with pytest.raises(ValueError, match="answer must be a string"):
        verify_citations(object(), [])
    with pytest.raises(ValueError, match="100,000"):
        verify_citations("a" * 100_001, [])
    with pytest.raises(ValueError, match="list of Citation"):
        verify_citations("answer", "not-a-list")
    with pytest.raises(ValueError, match="only Citation"):
        verify_citations("answer", [object()])
    with pytest.raises(ValueError, match="at most 100"):
        verify_citations("answer", itertools.repeat(citation()))


def test_audit_rejects_non_agent_answer_direct_calls():
    with pytest.raises(ValueError, match="AgentAnswer"):
        audit_hallucination(object())


def test_audit_names_check_as_structural_not_factual_proof():
    answer = AgentAnswer(answer="Alpha beta gamma [1].", citations=[citation()])
    message = audit_hallucination(answer)
    assert "Citation-structure" in message or "Citation diagnostic" in message
    assert "semantically grounded" not in message
