import json
from tools.integrity import (
    check_visual_entailment, extract_protocol, run_scientific_debate,
    compare_papers, generate_comparison_matrix, detect_conflicts,
    extract_limitations, EntailmentVerdict
)

class TestIntegrityExtended:
    def test_check_visual_entailment(self):
        result_json = check_visual_entailment("The sky is blue", "Fig 1", "doc1")
        result = json.loads(result_json)
        assert result["claim_text"] == "The sky is blue"
        assert result["verdict"] == EntailmentVerdict.UNCERTAIN

    def test_extract_protocol(self):
        result_json = extract_protocol("Heat at 90C for 5 mins.", "doc2")
        result = json.loads(result_json)
        assert len(result["steps"]) > 0
        assert result["metadata"]["source_doc"] == "doc2"

    def test_run_scientific_debate(self):
        result_json = run_scientific_debate("Claim X is true", "Context Y")
        result = json.loads(result_json)
        assert "verdict" in result
        assert len(result["key_issues"]) > 0

    def test_compare_papers(self):
        result_json = compare_papers(["doc1", "doc2"], "Methodology")
        result = json.loads(result_json)
        assert len(result["consistencies"]) > 0
        assert "summary" in result

    def test_generate_comparison_matrix(self):
        matrix = generate_comparison_matrix(["doc1", "doc2"], ["Accuracy", "Recall"])
        assert "Accuracy" in matrix
        assert "doc1" in matrix
        assert "| --- |" in matrix

    def test_detect_conflicts(self):
        result_json = detect_conflicts("Topic T", "Context C")
        result = json.loads(result_json)
        assert result["topic"] == "Topic T"
        assert len(result["conflicts"]) > 0

    def test_extract_limitations(self):
        # Test with 'limitations' text
        result_json = extract_limitations("doc3", "Our study has several limitations...")
        result = json.loads(result_json)
        assert len(result["limitations"]) > 0
        
        # Test without 'limitations' text
        result_json_2 = extract_limitations("doc3", "Perfect study.")
        result_2 = json.loads(result_json_2)
        assert len(result_2["limitations"]) == 0
