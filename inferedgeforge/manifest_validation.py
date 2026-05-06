"""Manifest sanity validation for Runtime/Lab handoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from inferedgeforge.build import load_manifest

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ManifestIssue:
    severity: Severity
    path: str
    message: str


@dataclass(frozen=True)
class ManifestValidationResult:
    manifest_path: Path
    issues: tuple[ManifestIssue, ...]

    @property
    def errors(self) -> tuple[ManifestIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ManifestIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def valid(self) -> bool:
        return not self.errors


def resolve_manifest_path(
    *,
    manifest: str | Path | None = None,
    build_dir: str | Path | None = None,
) -> Path:
    if manifest is None and build_dir is None:
        raise ValueError("provide --manifest or --build-dir")
    if manifest is not None and build_dir is not None:
        raise ValueError("provide only one of --manifest or --build-dir")
    if manifest is not None:
        return Path(manifest)
    return Path(build_dir) / "manifest.json"  # type: ignore[arg-type]


def validate_manifest(
    *,
    manifest: str | Path | None = None,
    build_dir: str | Path | None = None,
) -> ManifestValidationResult:
    manifest_path = resolve_manifest_path(manifest=manifest, build_dir=build_dir)
    payload = load_manifest(manifest_path)
    issues: list[ManifestIssue] = []

    _validate_top_level(payload, issues)
    build = _section(payload, "build", issues)
    source_model = _section(payload, "source_model", issues)
    artifact = _section(payload, "artifact", issues)
    runtime = _section(payload, "runtime", issues)

    if build is not None:
        for key in ("build_id", "timestamp", "preset_name", "backend", "target"):
            _required_string(build, f"build.{key}", issues)

    if source_model is not None:
        _required_string(source_model, "source_model.path", issues)
        _required_string(source_model, "source_model.format", issues)
        _recommended_string(source_model, "source_model.sha256", issues)

    if artifact is not None:
        _required_string(artifact, "artifact.path", issues)
        _required_string(artifact, "artifact.format", issues)
        _required_string(artifact, "artifact.role", issues)
        _recommended_string(artifact, "artifact.sha256", issues)
        _recommended_string(artifact, "artifact.model_name", issues)

    if runtime is not None:
        for key in ("engine", "device", "precision", "model_path", "artifact_path"):
            _required_string(runtime, f"runtime.{key}", issues)
        for key in ("batch", "height", "width"):
            _recommended_positive_int(runtime, f"runtime.{key}", issues)

    if build is not None and runtime is not None:
        _matching_string(build, "backend", runtime, "engine", "build.backend", "runtime.engine", issues)
        _matching_string(build, "target", runtime, "device", "build.target", "runtime.device", issues)

    if artifact is not None and runtime is not None:
        _matching_string(artifact, "path", runtime, "artifact_path", "artifact.path", "runtime.artifact_path", issues)

    _validate_compare_context(payload, issues)
    return ManifestValidationResult(manifest_path=manifest_path, issues=tuple(issues))


def format_manifest_validation(result: ManifestValidationResult) -> str:
    status = "valid" if result.valid else "invalid"
    lines = [
        f"Manifest validation: {status}",
        f"Manifest: {result.manifest_path}",
        f"Errors: {len(result.errors)}",
        f"Warnings: {len(result.warnings)}",
    ]
    for issue in result.issues:
        lines.append(f"{issue.severity.upper()} {issue.path}: {issue.message}")
    return "\n".join(lines)


def _validate_top_level(payload: dict[str, object], issues: list[ManifestIssue]) -> None:
    if not _non_empty_string(payload.get("schema_version")):
        issues.append(
            ManifestIssue(
                severity="warning",
                path="schema_version",
                message="schema_version is recommended for manifest contract traceability.",
            )
        )
    if not isinstance(payload.get("tool"), dict):
        issues.append(
            ManifestIssue(
                severity="warning",
                path="tool",
                message="tool metadata is recommended for provenance review.",
            )
        )


def _section(
    payload: dict[str, object],
    name: str,
    issues: list[ManifestIssue],
) -> dict[str, object] | None:
    value = payload.get(name)
    if not isinstance(value, dict):
        issues.append(
            ManifestIssue(
                severity="error",
                path=name,
                message=f"manifest.{name} section is required for Runtime/Lab handoff.",
            )
        )
        return None
    return value


def _required_string(section: dict[str, object], path: str, issues: list[ManifestIssue]) -> None:
    key = path.rsplit(".", 1)[-1]
    if not _non_empty_string(section.get(key)):
        issues.append(
            ManifestIssue(
                severity="error",
                path=path,
                message=f"{path} must be a non-empty string.",
            )
        )


def _recommended_string(section: dict[str, object], path: str, issues: list[ManifestIssue]) -> None:
    key = path.rsplit(".", 1)[-1]
    if not _non_empty_string(section.get(key)):
        issues.append(
            ManifestIssue(
                severity="warning",
                path=path,
                message=f"{path} is recommended for provenance and reviewability.",
            )
        )


def _recommended_positive_int(
    section: dict[str, object],
    path: str,
    issues: list[ManifestIssue],
) -> None:
    key = path.rsplit(".", 1)[-1]
    value = section.get(key)
    if not isinstance(value, int) or value <= 0:
        issues.append(
            ManifestIssue(
                severity="warning",
                path=path,
                message=f"{path} is recommended for compare_key and input-shape context.",
            )
        )


def _matching_string(
    left: dict[str, object],
    left_key: str,
    right: dict[str, object],
    right_key: str,
    left_path: str,
    right_path: str,
    issues: list[ManifestIssue],
) -> None:
    left_value = left.get(left_key)
    right_value = right.get(right_key)
    if _non_empty_string(left_value) and _non_empty_string(right_value) and left_value != right_value:
        issues.append(
            ManifestIssue(
                severity="error",
                path=f"{left_path}/{right_path}",
                message=f"{left_path} and {right_path} must match for handoff consistency.",
            )
        )


def _validate_compare_context(payload: dict[str, object], issues: list[ManifestIssue]) -> None:
    source_model = payload.get("source_model")
    runtime = payload.get("runtime")
    if not isinstance(source_model, dict) or not isinstance(runtime, dict):
        return
    missing = []
    if not _non_empty_string(source_model.get("path")):
        missing.append("source_model.path")
    for key in ("engine", "device", "precision"):
        if not _non_empty_string(runtime.get(key)):
            missing.append(f"runtime.{key}")
    for key in ("batch", "height", "width"):
        value = runtime.get(key)
        if not isinstance(value, int) or value <= 0:
            missing.append(f"runtime.{key}")
    if missing:
        issues.append(
            ManifestIssue(
                severity="warning",
                path="compare_context",
                message="compare_key/backend_key context is incomplete: " + ", ".join(missing),
            )
        )


def _non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
