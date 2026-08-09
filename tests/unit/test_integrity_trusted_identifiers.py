from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from tools.integrity import check_visual_entailment


def test_arbitrary_registry_identifier_does_not_bypass_privacy_mask():
    arbitrary_identifier = "555-123-4567"
    source_bytes = b"immutable-retained-source"
    store = SimpleNamespace(source_bytes=lambda **_kwargs: source_bytes)

    with patch(
        "tools.integrity._document_metadata",
        return_value={"filename": "figure.pdf"},
    ), patch("tools.integrity.get_document_store", return_value=store), patch(
        "tools.integrity._extract_figure_region",
        return_value=("encoded", 1, "Figure 1. Accuracy"),
    ):
        result = json.loads(
            check_visual_entailment(
                "Accuracy increased.",
                "Figure 1",
                arbitrary_identifier,
                owner_id="alice",
                client=None,
            )
        )

    citation = result["citations"][0]
    assert citation["doc_id"] == "[REDACTED_PHONE]"
    assert arbitrary_identifier not in citation["url"]
