import pytest

from tools.privacy import mask_metadata_text, sanitize_metadata


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Contact alice@example.com.", "Contact [REDACTED_EMAIL]."),
        ("Contact alice@example.com, then continue.", "Contact [REDACTED_EMAIL], then continue."),
        ("(alice@example.com)", "([REDACTED_EMAIL])"),
        ("alice@example.com; bob@example.org:", "[REDACTED_EMAIL]; [REDACTED_EMAIL]:"),
        (
            "Researcher alice.smith+trial@sub.example.co.uk\nMethods follow.",
            "Researcher [REDACTED_EMAIL]\nMethods follow.",
        ),
    ],
)
def test_email_masking_preserves_ordinary_trailing_punctuation(source, expected):
    assert mask_metadata_text(source) == expected


def test_nested_metadata_uses_the_same_sentence_final_email_boundary():
    sanitized = sanitize_metadata({
        "ocr": "Scanned methods. Contact alice@example.com.",
        "sections": ["Correspondence: bob@example.org,"],
    })

    assert sanitized == {
        "ocr": "Scanned methods. Contact [REDACTED_EMAIL].",
        "sections": ["Correspondence: [REDACTED_EMAIL],"],
    }


def test_incomplete_email_like_values_are_not_overmasked():
    source = "Identifiers alice@example and @example.com are incomplete."

    assert mask_metadata_text(source) == source
