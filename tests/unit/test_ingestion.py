import pytest
from tools.ingestion import extract_academic_metadata, _chunk_text_semantically

class TestIngestion:

    def test_metadata_extraction(self):
        text = """
        Title: Quantum Computing Advances
        DOI: 10.1038/s41586-023-0000
        Published: 2024
        
        Abstract...
        """
        meta = extract_academic_metadata(text)
        assert meta["doi"] == "10.1038/s41586-023-0000"
        assert meta["year"] == "2024"
        assert "Quantum Computing" in meta["extracted_title"]

    def test_semantic_chunking(self):
        # Create text with clear paragraph breaks
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        chunks = _chunk_text_semantically(text, max_chars=50) # Small limit
        
        # If max_chars allows it, chunks should respect newlines
        # Logic says: if current + next < max, merge. else split.
        # "Para 1.\n\n" is ~9 chars. 
        # If we pass 50, all might fit in one if logic merges eagerly.
        # Let's use a very small limit to force split.
        
        chunks_small = _chunk_text_semantically(text, max_chars=10)
        assert len(chunks_small) >= 2
