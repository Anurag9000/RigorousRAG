from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one repair target, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, sentinel: str, addition: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if sentinel in text:
        raise RuntimeError(f"{path}: repair sentinel already present: {sentinel}")
    if not text.endswith("\n"):
        text += "\n"
    target.write_text(text + addition, encoding="utf-8")


def main() -> int:
    # The canonical artifact digest includes schema, but schema is metadata rather than a
    # dataclass constructor field. Keep schema inside the hash payload while passing only
    # actual dataclass fields to the constructor.
    replace_once(
        "training/cross_profile_fusion_fitting.py",
        "        return cls(**payload, artifact_sha256=_canonical_digest(payload))\n",
        "        constructor = dict(payload)\n"
        "        constructor.pop(\"schema\")\n"
        "        return cls(**constructor, artifact_sha256=_canonical_digest(payload))\n",
    )
    replace_once(
        "training/cross_profile_listwise_fusion.py",
        "        return cls(**payload, artifact_sha256=_digest(payload))\n",
        "        constructor = dict(payload)\n"
        "        constructor.pop(\"schema\")\n"
        "        return cls(**constructor, artifact_sha256=_digest(payload))\n",
    )

    replace_once(
        "training/authoritative_classical_training_cli_v3.py",
        "    profiles = tuple(v1._identifier(item, \"profile_id\", 200) for item in value)\n"
        "    if not profiles:\n"
        "        raise ValueError(\"profile_ids must contain at least one profile\")\n",
        "    profiles_list: list[str] = []\n"
        "    for raw_item in value:\n"
        "        if not isinstance(raw_item, str):\n"
        "            raise ValueError(\"profile_ids entries must be strings\")\n"
        "        profile = v1._identifier(raw_item, \"profile_id\", 200)\n"
        "        if profile != raw_item:\n"
        "            raise ValueError(\"profile_ids entries must use canonical identifiers without surrounding whitespace\")\n"
        "        profiles_list.append(profile)\n"
        "    profiles = tuple(profiles_list)\n"
        "    if not profiles:\n"
        "        raise ValueError(\"profile_ids must contain at least one profile\")\n",
    )
    replace_once(
        "training/authoritative_classical_training_cli_v3.py",
        "        key = v1._identifier(raw_key, f\"{label} key\", 200)\n"
        "        if key in normalized:\n",
        "        key = v1._identifier(raw_key, f\"{label} key\", 200)\n"
        "        if key != raw_key:\n"
        "            raise ValueError(f\"{label} keys must use canonical identifiers without surrounding whitespace\")\n"
        "        if key in normalized:\n",
    )

    replace_once(
        "training/authoritative_retrieval_training_cli_v2.py",
        "def _preflight(config_path: str | Path) -> Mapping[str, Any]:\n"
        "    selected = Path(config_path).expanduser().resolve(strict=True)\n"
        "    raw = json.loads(selected.read_text(encoding=\"utf-8\"))\n",
        "def _preflight(config_path: str | Path) -> Mapping[str, Any]:\n"
        "    candidate = Path(config_path).expanduser()\n"
        "    _reject_symlink_components(candidate, \"config\")\n"
        "    selected = candidate.resolve(strict=True)\n"
        "    if not selected.is_file():\n"
        "        raise ValueError(\"config must resolve to a regular file\")\n"
        "    raw = json.loads(selected.read_text(encoding=\"utf-8\"))\n",
    )

    # Installed console scripts must not bypass the hardened authority versions used by
    # run_all_training.py.
    replace_once(
        "pyproject.toml",
        'rigorousrag-classical-training = "training.authoritative_classical_training_cli_v2:main"\n',
        'rigorousrag-classical-training = "training.authoritative_classical_training_cli_v3:main"\n',
    )
    replace_once(
        "pyproject.toml",
        'rigorousrag-retrieval-training = "training.authoritative_retrieval_training_cli:main"\n',
        'rigorousrag-retrieval-training = "training.authoritative_retrieval_training_cli_v2:main"\n',
    )

    replace_once(
        "tests/test_authoritative_classical_training_cli_v3.py",
        "    with pytest.raises(ValueError, match=\"cover profile_ids exactly\"):\n"
        "        run_config(config)\n\n\n"
        "def test_zero_source_revision_is_not_an_alias_for_auto(tmp_path: Path) -> None:\n",
        "    with pytest.raises(ValueError, match=\"canonical identifiers\"):\n"
        "        run_config(config)\n\n\n"
        "def test_profile_ids_may_not_have_surrounding_whitespace(tmp_path: Path) -> None:\n"
        "    config = _config(tmp_path)\n"
        "    payload = json.loads(config.read_text(encoding=\"utf-8\"))\n"
        "    payload[\"profile_ids\"] = [\" dense \", \"sparse\"]\n"
        "    _write_json(config, payload)\n\n"
        "    with pytest.raises(ValueError, match=\"canonical identifiers\"):\n"
        "        run_config(config)\n\n\n"
        "def test_zero_source_revision_is_not_an_alias_for_auto(tmp_path: Path) -> None:\n",
    )

    append_once(
        "tests/test_authoritative_retrieval_training_cli_v2.py",
        "test_symlinked_config_file_fails_closed",
        "\n\ndef test_symlinked_config_file_fails_closed(tmp_path: Path) -> None:\n"
        "    config = _fixture(tmp_path)\n"
        "    linked = tmp_path / \"linked-config.json\"\n"
        "    try:\n"
        "        os.symlink(config, linked)\n"
        "    except (OSError, NotImplementedError):\n"
        "        pytest.skip(\"symlinks are unavailable on this platform\")\n"
        "    with pytest.raises(ValueError, match=\"path contains a symlink component\"):\n"
        "        _preflight(linked)\n\n\n"
        "def test_symlinked_parent_of_config_fails_closed(tmp_path: Path) -> None:\n"
        "    real_parent = tmp_path / \"real-config-parent\"\n"
        "    config = _fixture(real_parent)\n"
        "    linked_parent = tmp_path / \"linked-config-parent\"\n"
        "    try:\n"
        "        os.symlink(real_parent, linked_parent, target_is_directory=True)\n"
        "    except (OSError, NotImplementedError):\n"
        "        pytest.skip(\"symlinks are unavailable on this platform\")\n"
        "    with pytest.raises(ValueError, match=\"path contains a symlink component\"):\n"
        "        _preflight(linked_parent / config.name)\n",
    )

    # Defensive postconditions: the two exact constructor defects and old console-script
    # bypasses must be gone after this transaction.
    assertions = {
        "training/cross_profile_fusion_fitting.py": "return cls(**payload, artifact_sha256=_canonical_digest(payload))",
        "training/cross_profile_listwise_fusion.py": "return cls(**payload, artifact_sha256=_digest(payload))",
        "pyproject.toml": "authoritative_classical_training_cli_v2:main",
    }
    for path, forbidden in assertions.items():
        if forbidden in (ROOT / path).read_text(encoding="utf-8"):
            raise RuntimeError(f"{path}: forbidden stale training-authority surface remains: {forbidden}")

    print("training authority repair transaction applied successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
