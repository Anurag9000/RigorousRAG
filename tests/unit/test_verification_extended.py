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


def test_duplicate_labels_are_rejected():
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


def test_audit_names_check_as_structural_not_factual_proof():
    answer = AgentAnswer(answer="Alpha beta gamma [1].", citations=[citation()])
    message = audit_hallucination(answer)
    assert "Citation-structure" in message or "Citation diagnostic" in message
    assert "semantically grounded" not in message
