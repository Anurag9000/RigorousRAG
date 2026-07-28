from tools.privacy import mask_metadata_text, sanitize_metadata_dict


def test_local_paths_are_redacted_from_public_metadata():
    value = (
        "failed at /private/alice/state.sqlite3 and C:\\Users\\Alice\\secret.txt "
        "or ~/workspace/config.json"
    )

    masked = mask_metadata_text(value)

    assert "/private" not in masked
    assert "C:\\Users" not in masked
    assert "~/workspace" not in masked
    assert masked.count("[REDACTED_PATH]") == 3


def test_uri_credentials_and_secret_parameters_are_redacted():
    value = (
        "https://alice:password@example.com/query?api_key=top-secret"
        "&access_token=second-secret"
    )

    masked = mask_metadata_text(value)

    assert "alice" not in masked
    assert "password" not in masked
    assert "top-secret" not in masked
    assert "second-secret" not in masked
    assert "example.com" in masked
    assert masked.count("[REDACTED_SECRET]") == 2


def test_file_uri_and_nested_metadata_are_redacted():
    sanitized = sanitize_metadata_dict({
        "error": "file:///var/lib/rigorousrag/jobs.sqlite3",
        "nested": ["contact admin@example.com", "/srv/app/private/key.pem"],
    })

    assert sanitized["error"] == "[REDACTED_PATH]"
    assert sanitized["nested"][0] == "contact [REDACTED_EMAIL]"
    assert sanitized["nested"][1] == "[REDACTED_PATH]"
