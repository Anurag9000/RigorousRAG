import json
from unittest.mock import patch

import fitz

from tools.integrity import (
    _extract_figure_region,
    compare_papers,
    extract_protocol,
    generate_comparison_matrix,
    run_scientific_debate,
)


def test_protocol_fallback_does_not_convert_nonprocedural_prose_into_steps():
    result = json.loads(extract_protocol("The study discusses uncertainty and prior work.", client=None))
    assert result["steps"] == []
    assert result["warnings"]


def test_protocol_fallback_extracts_only_explicit_actions():
    text = "Add buffer and incubate at 37 C for 10 minutes. The study was important."
    result = json.loads(extract_protocol(text, "doc-1", client=None))
    assert len(result["steps"]) == 1
    assert "Add buffer" in result["steps"][0]["description"]
    assert result["metadata"]["source_doc"] == "doc-1"


def test_debate_fails_closed_without_evidence():
    result = json.loads(run_scientific_debate("Claim", "", client=None))
    assert result["verdict"] == "insufficient evidence"
    assert result["supporting_evidence"] == []


def test_judge_receives_original_evidence_not_only_generated_arguments():
    prompts = []

    def fake_completion(_client, **kwargs):
        prompts.append(kwargs["user"])
        if len(prompts) == 1:
            return "advocate"
        if len(prompts) == 2:
            return "skeptic"
        return json.dumps({
            "verdict": "uncertain",
            "key_issues": [],
            "supporting_evidence": ["measured evidence"],
            "recommended_followups": [],
            "uncertainty": "limited sample",
        })

    with patch("tools.integrity._completion", side_effect=fake_completion):
        run_scientific_debate("Claim", "ORIGINAL EVIDENCE", client=object())
    assert len(prompts) == 3
    assert "ORIGINAL EVIDENCE" in prompts[2]


def test_comparison_stops_when_any_document_has_no_evidence():
    def fake_retrieve(doc_id, *_args, **_kwargs):
        return ([], [])

    with patch("tools.integrity._retrieve_document_evidence", side_effect=fake_retrieve):
        result = json.loads(compare_papers(["a", "b"], "accuracy", owner_id="alice"))
    assert sorted(result["evidence_gaps"]) == ["a", "b"]
    assert "not generated" in result["summary"].lower()


def test_matrix_stops_on_missing_evidence():
    with patch("tools.integrity._retrieve_document_evidence", return_value=([], [])):
        result = json.loads(generate_comparison_matrix(["a"], ["accuracy"], owner_id="alice"))
    assert result["markdown"] == ""
    assert result["evidence_gaps"] == ["a"]


def test_figure_region_is_tied_to_exact_caption_page(tmp_path):
    path = tmp_path / "figure.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 120), "UNRELATED HEADER")
    page.insert_text((72, 600), "Figure 2B. Accuracy across conditions")
    page.draw_rect(fitz.Rect(100, 250, 400, 520), color=(0, 0, 0))
    document.save(path)
    document.close()
    encoded, page_number, caption = _extract_figure_region(str(path), "Figure 2B")
    assert encoded
    assert page_number == 1
    assert "Figure 2B" in caption
