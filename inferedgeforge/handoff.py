"""Helpers for projecting Forge build records into downstream handoff payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from inferedgeforge.schemas import BuildMetadata


def build_worker_runtime_summary(
    metadata: BuildMetadata | Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
    *,
    metadata_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the Forge summary consumed by Lab worker requests and Runtime config."""

    metadata_payload = (
        metadata.to_dict() if isinstance(metadata, BuildMetadata) else dict(metadata)
    )
    manifest_payload = dict(manifest) if manifest is not None else {}

    build = _require_object(metadata_payload, "build")
    source_model = _require_object(metadata_payload, "source_model")
    artifacts = metadata_payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("metadata.artifacts must contain at least one artifact")
    artifact = artifacts[0]
    if not isinstance(artifact, Mapping):
        raise ValueError("metadata.artifacts[0] must be an object")
    artifact = dict(artifact)

    manifest_build = _optional_object(manifest_payload, "build")
    manifest_source_model = _optional_object(manifest_payload, "source_model")
    manifest_artifact = _optional_object(manifest_payload, "artifact")
    manifest_runtime = _optional_object(manifest_payload, "runtime")

    lab_compat = _optional_object(metadata_payload, "lab_compat")
    metadata_runtime = _optional_object(lab_compat, "runtime")
    preset_snapshot = _optional_object(metadata_payload, "preset_snapshot")

    summary = {
        "source_model_path": _first_string(
            manifest_source_model.get("path"),
            source_model.get("path"),
        ),
        "source_model_sha256": _first_string(
            manifest_source_model.get("sha256"),
            source_model.get("sha256"),
        ),
        "backend": _first_string(
            manifest_runtime.get("engine"),
            manifest_build.get("backend"),
            metadata_runtime.get("engine"),
            build.get("backend"),
        ),
        "target": _first_string(
            manifest_runtime.get("device"),
            manifest_build.get("target"),
            metadata_runtime.get("device"),
            build.get("target"),
        ),
        "precision": _first_string(
            manifest_runtime.get("precision"),
            manifest_artifact.get("precision"),
            metadata_runtime.get("precision"),
            _nested_string(preset_snapshot, "build_options", "precision"),
        ),
        "batch": _first_int(
            manifest_runtime.get("batch"),
            metadata_runtime.get("requested_batch"),
            _nested_value(preset_snapshot, "metadata", "requested_batch"),
        ),
        "height": _first_int(
            manifest_runtime.get("height"),
            metadata_runtime.get("requested_height"),
            _nested_value(preset_snapshot, "metadata", "requested_height"),
        ),
        "width": _first_int(
            manifest_runtime.get("width"),
            metadata_runtime.get("requested_width"),
            _nested_value(preset_snapshot, "metadata", "requested_width"),
        ),
        "artifact_path": _first_string(
            manifest_runtime.get("artifact_path"),
            manifest_artifact.get("model_path"),
            manifest_artifact.get("path"),
            metadata_runtime.get("runtime_artifact_path"),
            artifact.get("path"),
        ),
        "artifact_sha256": _first_string(
            manifest_artifact.get("sha256"),
            artifact.get("sha256"),
        ),
        "artifact_type": _first_string(
            manifest_artifact.get("format"),
            artifact.get("format"),
        ),
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "preset_name": _first_string(
            manifest_build.get("preset_name"),
            build.get("preset_name"),
            preset_snapshot.get("name"),
        ),
        "build_id": _first_string(
            manifest_build.get("build_id"),
            build.get("build_id"),
        ),
    }
    _validate_worker_runtime_summary(summary)
    return summary


metadata_to_worker_runtime_summary = build_worker_runtime_summary


def _validate_worker_runtime_summary(summary: Mapping[str, Any]) -> None:
    required_strings = (
        "source_model_path",
        "source_model_sha256",
        "backend",
        "target",
        "precision",
        "artifact_path",
        "artifact_sha256",
        "artifact_type",
        "preset_name",
        "build_id",
    )
    for field in required_strings:
        if not isinstance(summary.get(field), str) or not summary[field]:
            raise ValueError(f"worker/runtime summary missing {field}")

    for field in ("batch", "height", "width"):
        value = summary.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"worker/runtime summary field {field} must be a positive integer")


def _require_object(data: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = data.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _optional_object(data: Mapping[str, Any] | None, field: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        return {}
    value = data.get(field)
    if not isinstance(value, Mapping):
        return {}
    return dict(value)


def _nested_value(data: Mapping[str, Any], first: str, second: str) -> Any:
    first_value = data.get(first)
    if not isinstance(first_value, Mapping):
        return None
    return first_value.get(second)


def _nested_string(data: Mapping[str, Any], first: str, second: str) -> str | None:
    value = _nested_value(data, first, second)
    return value if isinstance(value, str) and value else None


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
    return None
