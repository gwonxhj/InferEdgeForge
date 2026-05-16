"""Agent manifest contract helpers for Reliable Edge Agent Runtime handoff."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from inferedgeforge.build import load_manifest

AGENT_MANIFEST_SCHEMA_VERSION = "inferedge-agent-manifest-v1"
DEFAULT_TELEMETRY_CONTRACT_VERSION = "inferedge-agent-telemetry-v1"

AgentType = Literal["vision", "voice", "safety", "utility"]
Severity = Literal["error", "warning"]

AGENT_TYPES = {"vision", "voice", "safety", "utility"}
FALLBACK_MODES = {
    "none",
    "drop_stale",
    "degrade_backend",
    "skip_low_priority",
    "notify_only",
}


@dataclass(frozen=True)
class AgentManifestIssue:
    severity: Severity
    path: str
    message: str


@dataclass(frozen=True)
class AgentManifestValidationResult:
    manifest_path: Path
    issues: tuple[AgentManifestIssue, ...]

    @property
    def errors(self) -> tuple[AgentManifestIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[AgentManifestIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def valid(self) -> bool:
        return not self.errors


def load_agent_manifest(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("agent manifest must be a JSON object")
    return payload


def create_agent_manifest_from_manifest(
    *,
    manifest_path: str | Path,
    agent_id: str,
    agent_type: str,
    priority: int,
    latency_budget_ms: int,
    deadline_ms: int,
    input_type: str,
    output_type: str,
    fallback_mode: str,
    required_backend: str | None = None,
    device_target: str | None = None,
    precision: str | None = None,
    runtime_artifact_path: str | None = None,
    tool_schema_path: str | None = None,
    telemetry_contract_version: str = DEFAULT_TELEMETRY_CONTRACT_VERSION,
    guard_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create a standalone agent_manifest.json without changing manifest.json."""

    source_manifest = load_manifest(manifest_path)
    runtime = _optional_object(source_manifest.get("runtime"))
    artifact = _optional_object(source_manifest.get("artifact"))
    build = _optional_object(source_manifest.get("build"))

    inferred_backend = required_backend or _string(runtime.get("engine")) or _string(build.get("backend"))
    inferred_device = device_target or _string(runtime.get("device")) or _string(build.get("target"))
    inferred_precision = precision or _string(runtime.get("precision")) or _string(artifact.get("precision"))
    inferred_artifact = (
        runtime_artifact_path
        or _string(runtime.get("artifact_path"))
        or _string(artifact.get("path"))
        or _string(artifact.get("model_path"))
    )

    payload: dict[str, object] = {
        "schema_version": AGENT_MANIFEST_SCHEMA_VERSION,
        "agent_id": agent_id,
        "agent_type": agent_type,
        "priority": priority,
        "latency_budget_ms": latency_budget_ms,
        "deadline_ms": deadline_ms,
        "input_type": input_type,
        "output_type": output_type,
        "required_backend": inferred_backend or "",
        "device_target": inferred_device or "",
        "precision": inferred_precision or "",
        "runtime_artifact_path": inferred_artifact or "",
        "fallback_policy": {
            "mode": fallback_mode,
        },
        "telemetry_contract_version": telemetry_contract_version,
        "lab_compat": {
            "source_manifest_path": str(manifest_path),
            "runtime_artifact_path": inferred_artifact or "",
            "required_backend": inferred_backend or "",
            "device_target": inferred_device or "",
            "precision": inferred_precision or "",
        },
    }

    if tool_schema_path is not None:
        payload["tool_schema_path"] = tool_schema_path
    if guard_policy is not None:
        payload["guard_policy"] = guard_policy

    return payload


def validate_agent_manifest(manifest: str | Path) -> AgentManifestValidationResult:
    manifest_path = Path(manifest)
    payload = load_agent_manifest(manifest_path)
    issues: list[AgentManifestIssue] = []

    schema_version = payload.get("schema_version")
    if schema_version != AGENT_MANIFEST_SCHEMA_VERSION:
        issues.append(
            AgentManifestIssue(
                severity="error",
                path="schema_version",
                message=f"schema_version must be {AGENT_MANIFEST_SCHEMA_VERSION}.",
            )
        )

    _required_string(payload, "agent_id", issues)
    _enum_string(payload, "agent_type", AGENT_TYPES, issues)
    _required_int(payload, "priority", issues, minimum=0, maximum=100)
    _required_int(payload, "latency_budget_ms", issues, minimum=1)
    _required_int(payload, "deadline_ms", issues, minimum=1)
    _required_string(payload, "input_type", issues)
    _required_string(payload, "output_type", issues)
    _required_string(payload, "required_backend", issues)
    _required_string(payload, "device_target", issues)
    _required_string(payload, "precision", issues)
    _required_string(payload, "runtime_artifact_path", issues)
    _required_string(payload, "telemetry_contract_version", issues)

    fallback_policy = _required_object(payload, "fallback_policy", issues)
    if fallback_policy is not None:
        _enum_string(fallback_policy, "mode", FALLBACK_MODES, issues, prefix="fallback_policy.")

    lab_compat = _required_object(payload, "lab_compat", issues)
    if lab_compat is not None:
        for key in (
            "source_manifest_path",
            "runtime_artifact_path",
            "required_backend",
            "device_target",
            "precision",
        ):
            _required_string(lab_compat, key, issues, prefix="lab_compat.")
        _matching_payload_string(
            payload,
            "runtime_artifact_path",
            lab_compat,
            "runtime_artifact_path",
            "runtime_artifact_path",
            "lab_compat.runtime_artifact_path",
            issues,
        )
        for key in ("required_backend", "device_target", "precision"):
            _matching_payload_string(
                payload,
                key,
                lab_compat,
                key,
                key,
                f"lab_compat.{key}",
                issues,
            )

    tool_schema_path = payload.get("tool_schema_path")
    if tool_schema_path is not None and not _non_empty_string(tool_schema_path):
        issues.append(
            AgentManifestIssue(
                severity="error",
                path="tool_schema_path",
                message="tool_schema_path must be a non-empty string when provided.",
            )
        )

    guard_policy = payload.get("guard_policy")
    if guard_policy is not None and not isinstance(guard_policy, dict):
        issues.append(
            AgentManifestIssue(
                severity="error",
                path="guard_policy",
                message="guard_policy must be an object when provided.",
            )
        )

    return AgentManifestValidationResult(manifest_path=manifest_path, issues=tuple(issues))


def format_agent_manifest_validation(result: AgentManifestValidationResult) -> str:
    status = "valid" if result.valid else "invalid"
    lines = [
        f"Agent manifest validation: {status}",
        f"Manifest: {result.manifest_path}",
        f"Errors: {len(result.errors)}",
        f"Warnings: {len(result.warnings)}",
    ]
    for issue in result.issues:
        lines.append(f"{issue.severity.upper()} {issue.path}: {issue.message}")
    return "\n".join(lines)


def _required_object(
    payload: dict[str, object],
    key: str,
    issues: list[AgentManifestIssue],
    *,
    prefix: str = "",
) -> dict[str, object] | None:
    value = payload.get(key)
    path = f"{prefix}{key}"
    if not isinstance(value, dict):
        issues.append(
            AgentManifestIssue(
                severity="error",
                path=path,
                message=f"{path} must be an object.",
            )
        )
        return None
    return value


def _required_string(
    payload: dict[str, object],
    key: str,
    issues: list[AgentManifestIssue],
    *,
    prefix: str = "",
) -> None:
    path = f"{prefix}{key}"
    if not _non_empty_string(payload.get(key)):
        issues.append(
            AgentManifestIssue(
                severity="error",
                path=path,
                message=f"{path} must be a non-empty string.",
            )
        )


def _enum_string(
    payload: dict[str, object],
    key: str,
    allowed: set[str],
    issues: list[AgentManifestIssue],
    *,
    prefix: str = "",
) -> None:
    path = f"{prefix}{key}"
    value = payload.get(key)
    if not _non_empty_string(value):
        issues.append(
            AgentManifestIssue(
                severity="error",
                path=path,
                message=f"{path} must be one of: {', '.join(sorted(allowed))}.",
            )
        )
        return
    if value not in allowed:
        issues.append(
            AgentManifestIssue(
                severity="error",
                path=path,
                message=f"{path} must be one of: {', '.join(sorted(allowed))}.",
            )
        )


def _required_int(
    payload: dict[str, object],
    key: str,
    issues: list[AgentManifestIssue],
    *,
    minimum: int,
    maximum: int | None = None,
) -> None:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        message = f"{key} must be an integer >= {minimum}."
        if maximum is not None:
            message = f"{key} must be an integer between {minimum} and {maximum}."
        issues.append(AgentManifestIssue(severity="error", path=key, message=message))
        return
    if maximum is not None and value > maximum:
        issues.append(
            AgentManifestIssue(
                severity="error",
                path=key,
                message=f"{key} must be an integer between {minimum} and {maximum}.",
            )
        )


def _matching_payload_string(
    left: dict[str, object],
    left_key: str,
    right: dict[str, object],
    right_key: str,
    left_path: str,
    right_path: str,
    issues: list[AgentManifestIssue],
) -> None:
    left_value = left.get(left_key)
    right_value = right.get(right_key)
    if _non_empty_string(left_value) and _non_empty_string(right_value) and left_value != right_value:
        issues.append(
            AgentManifestIssue(
                severity="error",
                path=f"{left_path}/{right_path}",
                message=f"{left_path} and {right_path} must match for agent handoff consistency.",
            )
        )


def _optional_object(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    return value if _non_empty_string(value) else None


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())

