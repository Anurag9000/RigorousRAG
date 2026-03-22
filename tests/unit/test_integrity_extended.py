import json
from unittest.mock import patch, MagicMock
from tools.integrity import (
    check_visual_entailment, extract_protocol, run_scientific_debate,
    compare_papers, generate_comparison_matrix, detect_conflicts,
    extract_limitations, EntailmentVerdict
)

class TestIntegrityExtended:
    @patch('tools.integrity._get_file_path_for_doc')
    @patch('tools.integrity._extract_figure_image_b64')
    @patch('tools.integrity._client')
    def test_check_visual_entailment(self, mock_client, mock_extract_img, mock_get_path):
        # Simulate that we successfully extract a figure image (b64-encoded JPEG)
        mock_get_path.return_value = '/fake/path/doc.pdf'
        mock_extract_img.return_value = 'FAKEB64IMAGEDATA'

        # Mock LLM Response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "claim_text": "The sky is blue",
            "figure_id": "Fig 1",
            "verdict": "uncertain",
            "rationale": "Test rationale",
            "confidence": 0.5
        })
        mock_client.chat.completions.create.return_value = mock_response

        result_json = check_visual_entailment("The sky is blue", "Fig 1", "doc1")
        result = json.loads(result_json)
        assert result["claim_text"] == "The sky is blue"
        assert result["verdict"] == "uncertain"

    @patch('tools.integrity._client')
    def test_extract_protocol(self, mock_client):
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "steps": [{"description": "Heat at 90C for 5 mins.", "temperature": "90C", "time": "5 mins", "reagent": None, "notes": None}],
            "metadata": {"source_doc": "doc2", "step_count": 1, "extraction_method": "llm"}
        })
        mock_client.chat.completions.create.return_value = mock_response

        result_json = extract_protocol("Heat at 90C for 5 mins.", "doc2")
        result = json.loads(result_json)
        assert len(result["steps"]) > 0
        assert result["metadata"]["source_doc"] == "doc2"


    @patch('tools.integrity._client')
    def test_run_scientific_debate(self, mock_client):
        # Mock 3 calls (Advocate, Skeptic, Judge)
        mock_adv = MagicMock()
        mock_adv.choices[0].message.content = "Advocate argument"
        
        mock_skep = MagicMock()
        mock_skep.choices[0].message.content = "Skeptic argument"
        
        mock_judge = MagicMock()
        mock_judge.choices[0].message.content = json.dumps({
            "verdict": "Caution",
            "key_issues": ["Issue 1"],
            "supporting_evidence": ["Evidence 1"],
            "recommended_followups": ["Followup 1"]
        })
        
        mock_client.chat.completions.create.side_effect = [mock_adv, mock_skep, mock_judge]

        result_json = run_scientific_debate("Claim X is true", "Context Y")
        result = json.loads(result_json)
        assert result["verdict"] == "Caution"
        assert len(result["key_issues"]) > 0

    @patch('tools.integrity._client')
    def test_compare_papers(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({
            "consistencies": ["Consistency 1"],
            "conflicts": ["Conflict 1"],
            "trends": ["Trend 1"],
            "summary": "Test summary"
        })
        mock_client.chat.completions.create.return_value = mock_resp

        # Mock RAG inside tools.integrity
        with patch('tools.integrity.get_rag_layer') as mock_rag_init:
            mock_rag = mock_rag_init.return_value
            mock_rag.query.return_value = []
            
            result_json = compare_papers(["doc1", "doc2"], "Methodology")
            result = json.loads(result_json)
            assert len(result["consistencies"]) > 0
            assert "summary" in result

    def test_generate_comparison_matrix(self):
        # Mock RAG and Client
        with patch('tools.integrity.get_rag_layer') as mock_rag_init, \
             patch('tools.integrity._client') as mock_client:
            mock_rag = mock_rag_init.return_value
            mock_rag.query.return_value = [MagicMock(text="extracted text")]
            
            mock_ext = MagicMock()
            mock_ext.choices[0].message.content = "100 samples"
            mock_client.chat.completions.create.return_value = mock_ext

            matrix = generate_comparison_matrix(["doc1", "doc2"], ["Accuracy", "Recall"])
            assert "Accuracy" in matrix
            assert "doc1" in matrix
            assert "100 samples" in matrix

    @patch('tools.integrity._client')
    def test_detect_conflicts(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({
            "topic": "Topic T",
            "conflicts": [{"claim_a": "A", "claim_b": "B", "source_a": "S1", "source_b": "S2"}],
            "synthesis": "Test synthesis"
        })
        mock_client.chat.completions.create.return_value = mock_resp

        result_json = detect_conflicts("Topic T", "Context C")
        result = json.loads(result_json)
        assert result["topic"] == "Topic T"
        assert len(result["conflicts"]) > 0

    @patch('tools.integrity._client')
    def test_extract_limitations(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = json.dumps({
            "doc_id": "doc3",
            "limitations": ["Limit 1"],
            "recommendation": "Rec 1"
        })
        mock_client.chat.completions.create.return_value = mock_resp

        result_json = extract_limitations("doc3", "Our study has several limitations...")
        result = json.loads(result_json)
        assert len(result["limitations"]) > 0
