from tools.verification import verify_citations, audit_hallucination
from tools.models import Citation, AgentAnswer

class TestVerificationExtended:
    def test_verify_citations_ok(self):
        answer = "Claim 1 [1]. Claim 2 [2]."
        citations = [
            Citation(label="[1]", title="Title A", url="http://a.com", snippet="content 1", source_type="web_page"),
            Citation(label="[2]", title="Title B", url="http://b.com", snippet="content 2", source_type="web_page")
        ]
        issues = verify_citations(answer, citations)
        assert len(issues) == 0

    def test_verify_citations_missing(self):
        answer = "Claim 1 [1]. Claim 3 [3]."
        citations = [
            Citation(label="[1]", title="Title A", url="http://a.com", snippet="content 1", source_type="web_page")
        ]
        issues = verify_citations(answer, citations)
        assert len(issues) == 1
        assert issues[0]["label"] == "[3]"
        assert "no corresponding source" in issues[0]["error"]

    def test_audit_hallucination_verified(self):
        ans = AgentAnswer(
            answer="Correct [1]",
            citations=[Citation(label="[1]", title="A", url="a", snippet="s", source_type="web_page")],
            metadata={}
        )
        msg = audit_hallucination(ans)
        assert "Verified" in msg

    def test_audit_hallucination_warning(self):
        ans = AgentAnswer(
            answer="Hallucination [2]",
            citations=[Citation(label="[1]", title="A", url="a", snippet="s", source_type="web_page")],
            metadata={}
        )
        msg = audit_hallucination(ans)
        assert "Warning" in msg
        assert "[2]" in msg
