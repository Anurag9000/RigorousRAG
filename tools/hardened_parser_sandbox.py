"""Kernel/container-isolated parser job specifications for untrusted document parsing.

This module builds a strict Kubernetes-style Job manifest and validates deployment
capabilities. It does not submit jobs itself. A deployment must inject an executor and
attest external controls such as NetworkPolicy/CNI isolation; merely setting fields in a
manifest is not treated as proof that the cluster enforces them.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

_IMAGE_DIGEST_RE = re.compile(r"^[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}$")


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if (not selected and not allow_empty) or len(selected) > maximum or "\x00" in selected:
        raise ValueError(f"{label} is invalid")
    return selected


def _positive(value: Any, label: str, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is invalid")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} is invalid") from exc
    if not math.isfinite(parsed) or parsed <= 0 or parsed > maximum:
        raise ValueError(f"{label} is invalid")
    return parsed


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


@dataclass(frozen=True)
class SandboxDeploymentCapabilities:
    provider_id: str
    network_isolation: bool
    seccomp_runtime_default: bool
    apparmor: bool
    non_root_enforcement: bool
    readonly_rootfs: bool
    capability_drop: bool
    runtime_classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id", 500))
        for name in (
            "network_isolation",
            "seccomp_runtime_default",
            "apparmor",
            "non_root_enforcement",
            "readonly_rootfs",
            "capability_drop",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        object.__setattr__(self, "runtime_classes", tuple(sorted(set(_text(item, "runtime_class", 128) for item in self.runtime_classes))))


@dataclass(frozen=True)
class HardenedParserProfile:
    profile_id: str
    image: str
    command: tuple[str, ...]
    runtime_class: str = ""
    require_apparmor: bool = False
    cpu_limit_cores: float = 1.0
    memory_limit_bytes: int = 1024 * 1024 * 1024
    ephemeral_storage_bytes: int = 1024 * 1024 * 1024
    active_deadline_seconds: int = 300
    max_output_bytes: int = 250 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _text(self.profile_id, "profile_id", 256))
        image = _text(self.image, "image", 1000).lower()
        if not _IMAGE_DIGEST_RE.fullmatch(image):
            raise ValueError("parser image must be pinned by sha256 digest")
        object.__setattr__(self, "image", image)
        if isinstance(self.command, (str, bytes)) or not 1 <= len(self.command) <= 64:
            raise ValueError("command is invalid")
        selected_command = tuple(_text(item, "command argument", 4096) for item in self.command)
        if any(item in {"sh", "bash", "/bin/sh", "/bin/bash", "cmd", "powershell", "pwsh"} for item in selected_command[:1]):
            raise ValueError("shell entrypoints are not allowed for hardened parser jobs")
        object.__setattr__(self, "command", selected_command)
        object.__setattr__(self, "runtime_class", _text(self.runtime_class, "runtime_class", 128, allow_empty=True))
        if not isinstance(self.require_apparmor, bool):
            raise ValueError("require_apparmor must be boolean")
        object.__setattr__(self, "cpu_limit_cores", _positive(self.cpu_limit_cores, "cpu_limit_cores", 64.0))
        if isinstance(self.memory_limit_bytes, bool) or not isinstance(self.memory_limit_bytes, int) or not 16 * 1024 * 1024 <= self.memory_limit_bytes <= 512 * 1024**3:
            raise ValueError("memory_limit_bytes is invalid")
        if isinstance(self.ephemeral_storage_bytes, bool) or not isinstance(self.ephemeral_storage_bytes, int) or not 16 * 1024 * 1024 <= self.ephemeral_storage_bytes <= 2 * 1024**4:
            raise ValueError("ephemeral_storage_bytes is invalid")
        if isinstance(self.active_deadline_seconds, bool) or not isinstance(self.active_deadline_seconds, int) or not 1 <= self.active_deadline_seconds <= 86_400:
            raise ValueError("active_deadline_seconds is invalid")
        if isinstance(self.max_output_bytes, bool) or not isinstance(self.max_output_bytes, int) or not 1 <= self.max_output_bytes <= 2_000_000_000:
            raise ValueError("max_output_bytes is invalid")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class HardenedParserRequest:
    request_id: str
    owner_id: str
    input_object_reference: str
    input_sha256: str
    filename: str
    media_type: str
    profile_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id", 500))
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id", 256))
        object.__setattr__(self, "input_object_reference", _text(self.input_object_reference, "input_object_reference", 2000))
        digest = _text(self.input_sha256, "input_sha256", 64).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("input_sha256 must be SHA-256")
        object.__setattr__(self, "input_sha256", digest)
        object.__setattr__(self, "filename", _text(self.filename, "filename", 500))
        object.__setattr__(self, "media_type", _text(self.media_type, "media_type", 256).lower())
        profile = _text(self.profile_fingerprint, "profile_fingerprint", 64).lower()
        if len(profile) != 64 or any(ch not in "0123456789abcdef" for ch in profile):
            raise ValueError("profile_fingerprint must be SHA-256")
        object.__setattr__(self, "profile_fingerprint", profile)


@dataclass(frozen=True)
class HardenedParserResult:
    request_id: str
    output_object_reference: str
    output_sha256: str
    output_size_bytes: int
    executor_id: str
    sandbox_receipt_fingerprint: str


class HardenedParserExecutor(Protocol):
    @property
    def executor_id(self) -> str: ...

    @property
    def capabilities(self) -> SandboxDeploymentCapabilities: ...

    def execute(self, request: HardenedParserRequest, job_manifest: Mapping[str, Any]) -> HardenedParserResult: ...


def validate_sandbox_capabilities(profile: HardenedParserProfile, capabilities: SandboxDeploymentCapabilities) -> tuple[str, ...]:
    if not isinstance(profile, HardenedParserProfile) or not isinstance(capabilities, SandboxDeploymentCapabilities):
        raise TypeError("profile/capabilities types are invalid")
    missing: list[str] = []
    if not capabilities.network_isolation:
        missing.append("network_isolation")
    if not capabilities.seccomp_runtime_default:
        missing.append("seccomp_runtime_default")
    if not capabilities.non_root_enforcement:
        missing.append("non_root_enforcement")
    if not capabilities.readonly_rootfs:
        missing.append("readonly_rootfs")
    if not capabilities.capability_drop:
        missing.append("capability_drop")
    if profile.require_apparmor and not capabilities.apparmor:
        missing.append("apparmor")
    if profile.runtime_class and profile.runtime_class not in capabilities.runtime_classes:
        missing.append(f"runtime_class:{profile.runtime_class}")
    return tuple(missing)


def _quantity_bytes(value: int) -> str:
    return f"{value}"


def build_kubernetes_parser_job(
    request: HardenedParserRequest,
    profile: HardenedParserProfile,
    capabilities: SandboxDeploymentCapabilities,
    *,
    namespace: str,
    service_account_name: str = "",
    apparmor_profile: str = "runtime/default",
) -> Mapping[str, Any]:
    if not isinstance(request, HardenedParserRequest) or not isinstance(profile, HardenedParserProfile):
        raise TypeError("request/profile types are invalid")
    if request.profile_fingerprint != profile.fingerprint:
        raise ValueError("request profile fingerprint does not match parser profile")
    missing = validate_sandbox_capabilities(profile, capabilities)
    if missing:
        raise RuntimeError("sandbox deployment lacks required controls: " + ",".join(missing))
    selected_namespace = _text(namespace, "namespace", 128)
    service_account = _text(service_account_name, "service_account_name", 128, allow_empty=True)
    annotations = {
        "rigorousrag.openai.com/profile-fingerprint": profile.fingerprint,
        "rigorousrag.openai.com/input-sha256": request.input_sha256,
        "rigorousrag.openai.com/network-isolation-attested-by": capabilities.provider_id,
    }
    pod_security: dict[str, Any] = {
        "runAsNonRoot": True,
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    container_security: dict[str, Any] = {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "runAsNonRoot": True,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }
    if profile.require_apparmor:
        selected_apparmor = _text(apparmor_profile, "apparmor_profile", 256)
        container_security["appArmorProfile"] = {"type": "RuntimeDefault" if selected_apparmor == "runtime/default" else "Localhost", "localhostProfile": None if selected_apparmor == "runtime/default" else selected_apparmor}
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "automountServiceAccountToken": False,
        "enableServiceLinks": False,
        "hostNetwork": False,
        "hostPID": False,
        "hostIPC": False,
        "securityContext": pod_security,
        "containers": [
            {
                "name": "parser",
                "image": profile.image,
                "imagePullPolicy": "IfNotPresent",
                "command": list(profile.command),
                "env": [
                    {"name": "RIGOROUSRAG_INPUT_REF", "value": request.input_object_reference},
                    {"name": "RIGOROUSRAG_INPUT_SHA256", "value": request.input_sha256},
                    {"name": "RIGOROUSRAG_FILENAME", "value": request.filename},
                    {"name": "RIGOROUSRAG_MEDIA_TYPE", "value": request.media_type},
                    {"name": "RIGOROUSRAG_MAX_OUTPUT_BYTES", "value": str(profile.max_output_bytes)},
                ],
                "securityContext": container_security,
                "resources": {
                    "requests": {
                        "cpu": str(min(profile.cpu_limit_cores, 0.25)),
                        "memory": _quantity_bytes(min(profile.memory_limit_bytes, 256 * 1024 * 1024)),
                        "ephemeral-storage": _quantity_bytes(min(profile.ephemeral_storage_bytes, 256 * 1024 * 1024)),
                    },
                    "limits": {
                        "cpu": str(profile.cpu_limit_cores),
                        "memory": _quantity_bytes(profile.memory_limit_bytes),
                        "ephemeral-storage": _quantity_bytes(profile.ephemeral_storage_bytes),
                    },
                },
                "volumeMounts": [{"name": "scratch", "mountPath": "/work"}],
                "workingDir": "/work",
            }
        ],
        "volumes": [{"name": "scratch", "emptyDir": {"medium": "Memory", "sizeLimit": _quantity_bytes(profile.ephemeral_storage_bytes)}}],
    }
    if service_account:
        pod_spec["serviceAccountName"] = service_account
    if profile.runtime_class:
        pod_spec["runtimeClassName"] = profile.runtime_class
    name_digest = hashlib.sha256(request.request_id.encode("utf-8")).hexdigest()[:20]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": f"rigorousrag-parser-{name_digest}",
            "namespace": selected_namespace,
            "labels": {
                "app.kubernetes.io/name": "rigorousrag-parser",
                "rigorousrag.openai.com/request": name_digest,
            },
            "annotations": annotations,
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": profile.active_deadline_seconds,
            "ttlSecondsAfterFinished": 600,
            "template": {"metadata": {"annotations": annotations}, "spec": pod_spec},
        },
    }


__all__ = [
    "HardenedParserExecutor",
    "HardenedParserProfile",
    "HardenedParserRequest",
    "HardenedParserResult",
    "SandboxDeploymentCapabilities",
    "build_kubernetes_parser_job",
    "validate_sandbox_capabilities",
]
