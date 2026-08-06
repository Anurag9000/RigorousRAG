from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_LEDGER = Path("docs/rigorousrag_capability_ledger.json")
ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*-[0-9]{3}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

# The detailed historical audit used CORE-005 for the release-reproducibility
# sub-capability. The aggregate ledger intentionally folds that item into
# CORE-001. Keep the alias explicit and narrow rather than accepting arbitrary
# dangling dependencies.
LEGACY_DEPENDENCY_ALIASES = {"CORE-005": "CORE-001"}

REQUIRED_ROOT_KEYS = {
    "schema_version",
    "project",
    "repository",
    "audit_date",
    "audited_head",
    "authoritative_document",
    "status_model",
    "capabilities",
}
REQUIRED_CAPABILITY_KEYS = {
    "id",
    "category",
    "title",
    "origin",
    "implementation",
    "validation",
    "release",
    "priority",
    "evidence",
    "tests",
    "acceptance_criteria",
    "dependencies",
    "next_action",
}


class LedgerValidationError(ValueError):
    """Raised when a capability ledger violates its declared contract."""


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _safe_relative_path(value: Any) -> bool:
    if not _non_empty_string(value):
        return False
    candidate = Path(value)
    return not candidate.is_absolute() and ".." not in candidate.parts


def _list_of_strings(value: Any, *, non_empty: bool = False) -> bool:
    if not isinstance(value, list):
        return False
    if non_empty and not value:
        return False
    return all(_non_empty_string(item) for item in value)


def _resolve_dependency(identifier: str) -> str:
    return LEGACY_DEPENDENCY_ALIASES.get(identifier, identifier)


def validate_ledger(data: Any, repo_root: Path) -> list[str]:
    """Return every validation error rather than failing on the first one."""

    errors: list[str] = []
    if not isinstance(data, dict):
        return ["ledger root must be a JSON object"]

    missing_root = sorted(REQUIRED_ROOT_KEYS - set(data))
    if missing_root:
        errors.append(f"missing root keys: {', '.join(missing_root)}")
        return errors

    if data.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if data.get("project") != "RigorousRAG":
        errors.append("project must equal RigorousRAG")
    if not _non_empty_string(data.get("repository")):
        errors.append("repository must be a non-empty string")
    if not _non_empty_string(data.get("audit_date")):
        errors.append("audit_date must be a non-empty string")
    if not isinstance(data.get("audited_head"), str) or not SHA_PATTERN.fullmatch(
        data["audited_head"]
    ):
        errors.append("audited_head must be a lowercase 40-character Git SHA")
    authoritative_document = data.get("authoritative_document")
    if not _safe_relative_path(authoritative_document):
        errors.append("authoritative_document must be a safe relative path")
    elif not (repo_root / authoritative_document).is_file():
        errors.append(f"authoritative_document does not exist: {authoritative_document}")

    status_model = data.get("status_model")
    required_dimensions = {
        "implementation",
        "validation",
        "release",
        "origin",
        "priority",
        "category",
    }
    if not isinstance(status_model, dict):
        errors.append("status_model must be an object")
        return errors
    missing_dimensions = sorted(required_dimensions - set(status_model))
    if missing_dimensions:
        errors.append(f"status_model missing dimensions: {', '.join(missing_dimensions)}")
        return errors
    for dimension in sorted(required_dimensions):
        values = status_model.get(dimension)
        if not _list_of_strings(values, non_empty=True) or len(values) != len(set(values)):
            errors.append(f"status_model.{dimension} must contain unique non-empty strings")

    capabilities = data.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        errors.append("capabilities must be a non-empty list")
        return errors

    identifiers: list[str] = []
    records: dict[str, dict[str, Any]] = {}
    for index, capability in enumerate(capabilities):
        label = f"capabilities[{index}]"
        if not isinstance(capability, dict):
            errors.append(f"{label} must be an object")
            continue
        missing = sorted(REQUIRED_CAPABILITY_KEYS - set(capability))
        if missing:
            errors.append(f"{label} missing keys: {', '.join(missing)}")
            continue

        identifier = capability.get("id")
        if not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier):
            errors.append(f"{label}.id must match {ID_PATTERN.pattern}")
            continue
        identifiers.append(identifier)
        records.setdefault(identifier, capability)
        prefix = identifier

        for field in ("title", "next_action"):
            if not _non_empty_string(capability.get(field)):
                errors.append(f"{prefix}.{field} must be a non-empty string")

        for dimension in (
            "category",
            "origin",
            "implementation",
            "validation",
            "release",
            "priority",
        ):
            if capability.get(dimension) not in status_model.get(dimension, []):
                errors.append(f"{prefix}.{dimension} is not declared by status_model")

        for field in ("evidence", "tests", "dependencies"):
            if not _list_of_strings(capability.get(field)):
                errors.append(f"{prefix}.{field} must be a list of non-empty strings")

        criteria = capability.get("acceptance_criteria")
        if not _list_of_strings(criteria, non_empty=True):
            errors.append(f"{prefix}.acceptance_criteria must be a non-empty string list")

        for field in ("evidence", "tests"):
            values = capability.get(field)
            if not isinstance(values, list):
                continue
            if len(values) != len(set(values)):
                errors.append(f"{prefix}.{field} contains duplicates")
            for value in values:
                if not _safe_relative_path(value):
                    errors.append(f"{prefix}.{field} contains unsafe path: {value!r}")
                elif not (repo_root / value).exists():
                    errors.append(f"{prefix}.{field} path does not exist: {value}")

        implementation = capability.get("implementation")
        validation = capability.get("validation")
        release = capability.get("release")
        evidence = capability.get("evidence")
        tests = capability.get("tests")
        if implementation == "not_started" and validation != "not_validated":
            errors.append(f"{prefix}: not_started requires not_validated")
        if implementation == "implemented" and not evidence:
            errors.append(f"{prefix}: implemented capability requires evidence")
        if validation != "not_validated" and not (evidence or tests):
            errors.append(f"{prefix}: validated capability requires evidence or tests")
        if release == "release_verified" and (
            implementation != "implemented"
            or validation not in {"integration_validated", "experimentally_validated"}
        ):
            errors.append(
                f"{prefix}: release_verified requires implemented and integration/experimental validation"
            )

    duplicate_ids = sorted({identifier for identifier in identifiers if identifiers.count(identifier) > 1})
    if duplicate_ids:
        errors.append(f"duplicate capability IDs: {', '.join(duplicate_ids)}")

    identifier_set = set(identifiers)
    graph: dict[str, list[str]] = {}
    for identifier, capability in records.items():
        resolved_dependencies: list[str] = []
        for dependency in capability.get("dependencies", []):
            resolved = _resolve_dependency(dependency)
            if resolved == identifier:
                errors.append(f"{identifier}: self dependency via {dependency}")
            elif resolved not in identifier_set:
                errors.append(f"{identifier}: unknown dependency {dependency}")
            else:
                resolved_dependencies.append(resolved)
        graph[identifier] = resolved_dependencies

    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(identifier: str) -> None:
        state[identifier] = 1
        stack.append(identifier)
        for dependency in graph.get(identifier, []):
            if state.get(dependency, 0) == 0:
                visit(dependency)
            elif state.get(dependency) == 1:
                start = stack.index(dependency)
                errors.append("dependency cycle: " + " -> ".join(stack[start:] + [dependency]))
        stack.pop()
        state[identifier] = 2

    for identifier in graph:
        if state.get(identifier, 0) == 0:
            visit(identifier)

    return errors


def load_and_validate(path: Path, repo_root: Path | None = None) -> list[str]:
    resolved_path = path.resolve()
    root = repo_root.resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    try:
        data = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot load ledger {resolved_path}: {exc}"]
    return validate_ledger(data, root)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the RigorousRAG capability ledger")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_LEDGER,
        help=f"ledger path (default: {DEFAULT_LEDGER})",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repository root used to validate evidence paths",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    errors = load_and_validate(args.path, args.repo_root)
    if errors:
        print(f"Capability ledger validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    data = json.loads(args.path.read_text(encoding="utf-8"))
    print(
        "Capability ledger valid: "
        f"{len(data['capabilities'])} capabilities, audited head {data['audited_head']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
