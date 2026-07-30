import json
import math

from tools.privacy import mask_metadata_text, sanitize_metadata, sanitize_metadata_dict


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


def test_iso_dates_are_not_misclassified_as_phone_numbers():
    value = "created=2026-02-01 phone=+1 202-555-0114"

    masked = mask_metadata_text(value)

    assert "2026-02-01" in masked
    assert "202-555-0114" not in masked
    assert "[REDACTED_PHONE]" in masked


def test_file_uri_and_nested_metadata_are_redacted():
    sanitized = sanitize_metadata_dict({
        "error": "file:///var/lib/rigorousrag/jobs.sqlite3",
        "nested": ["contact admin@example.com", "/srv/app/private/key.pem"],
    })

    assert sanitized["error"] == "[REDACTED_PATH]"
    assert sanitized["nested"][0] == "contact [REDACTED_EMAIL]"
    assert sanitized["nested"][1] == "[REDACTED_PATH]"


def test_mapping_keys_are_redacted_bounded_and_collision_preserving():
    sanitized = sanitize_metadata_dict({
        "/private/alice/one.txt": "first",
        "/private/bob/two.txt": "second",
        "https://alice:password@example.com": "third",
        "x" * 700: "bounded",
    })

    assert sanitized["[REDACTED_PATH]"] == "first"
    assert sanitized["[REDACTED_PATH]#2"] == "second"
    assert "https://[REDACTED_CREDENTIALS]@example.com" in sanitized
    assert max(len(key) for key in sanitized) <= 500
    assert len(sanitized) == 4


def test_nonfinite_numbers_become_json_null_equivalents():
    sanitized = sanitize_metadata({
        "nan": float("nan"),
        "positive": float("inf"),
        "negative": float("-inf"),
        "finite": 1.25,
    })

    assert sanitized == {
        "nan": None,
        "positive": None,
        "negative": None,
        "finite": 1.25,
    }
    assert math.isfinite(sanitized["finite"])


def test_huge_integers_are_replaced_with_json_safe_sentinel():
    sanitized = sanitize_metadata({
        "normal": 123,
        "huge": 1 << 5000,
    })

    assert sanitized["normal"] == 123
    assert sanitized["huge"] == "[INTEGER_OUT_OF_RANGE]"
    json.dumps(sanitized, allow_nan=False)


def test_self_referential_containers_are_replaced_with_sentinel():
    mapping = {}
    mapping["self"] = mapping
    sequence = []
    sequence.append(sequence)

    sanitized_mapping = sanitize_metadata(mapping)
    sanitized_sequence = sanitize_metadata(sequence)

    assert sanitized_mapping["self"] == "[CIRCULAR_REFERENCE]"
    assert sanitized_sequence == ["[CIRCULAR_REFERENCE]"]


def test_excessive_depth_is_truncated_without_recursion_error():
    nested = value = {}
    for index in range(20):
        value["next"] = {}
        value = value["next"]
        value["index"] = index

    sanitized = sanitize_metadata(nested)
    cursor = sanitized
    found = False
    for _ in range(20):
        if cursor == "[TRUNCATED_DEPTH]":
            found = True
            break
        if not isinstance(cursor, dict):
            break
        cursor = cursor.get("next")

    assert found is True


def test_mapping_and_sequence_item_counts_are_bounded():
    mapping = {f"key-{index}": index for index in range(1500)}
    sequence = list(range(1500))

    sanitized_mapping = sanitize_metadata(mapping)
    sanitized_sequence = sanitize_metadata(sequence)

    assert len(sanitized_mapping) == 1001
    assert sanitized_mapping["__truncated_items__"] is True
    assert len(sanitized_sequence) == 1001
    assert sanitized_sequence[-1] == {"__truncated_items__": True}


def test_masked_key_collision_storm_remains_bounded_and_unique():
    mapping = {
        f"/private/tenant-{index}/secret.txt": index
        for index in range(1200)
    }

    sanitized = sanitize_metadata(mapping)

    assert len(sanitized) == 1001
    assert len(set(sanitized)) == len(sanitized)
    assert "[REDACTED_PATH]" in sanitized
    assert "[REDACTED_PATH]#1000" in sanitized
    assert sanitized["__truncated_items__"] is True


def test_strings_and_hostile_custom_objects_are_bounded_without_bool_calls():
    class BrokenString:
        def __bool__(self):
            raise RuntimeError("do not call bool")

        def __str__(self):
            raise RuntimeError("private /secret/path")

    broken = BrokenString()
    sanitized = sanitize_metadata({
        "long": "x" * 200_000,
        "broken": broken,
    })

    assert len(sanitized["long"]) == 100_000
    assert sanitized["broken"] == "[UNPRINTABLE_BrokenString]"
    assert mask_metadata_text(broken) == "[UNPRINTABLE_BrokenString]"


def test_hostile_container_subclasses_fail_closed_without_escaping():
    class BrokenDict(dict):
        def items(self):
            raise RuntimeError("private mapping")

    class BrokenList(list):
        def __iter__(self):
            raise RuntimeError("private sequence")

    assert sanitize_metadata(BrokenDict(secret="value")) == "[UNREADABLE_CONTAINER]"
    assert sanitize_metadata_dict(BrokenDict(secret="value")) == {}
    assert sanitize_metadata(BrokenList(["secret"])) == ["[UNREADABLE_CONTAINER]"]
