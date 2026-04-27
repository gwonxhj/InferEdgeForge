from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inferedgeforge.build import build_manifest_from_metadata
from inferedgeforge.schemas import BuildMetadata


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_runtime_handoff_metadata_fixture_satisfies_contract() -> None:
    metadata = load_fixture("runtime_handoff_metadata.json")

    validate_metadata_runtime_handoff(metadata)


def test_runtime_handoff_manifest_fixture_satisfies_contract() -> None:
    manifest = load_fixture("runtime_handoff_manifest.json")

    validate_manifest_runtime_handoff(manifest)


def test_manifest_builder_preserves_runtime_handoff_fields() -> None:
    metadata = BuildMetadata.from_dict(load_fixture("runtime_handoff_metadata.json"))

    manifest = build_manifest_from_metadata(metadata)

    validate_manifest_runtime_handoff(manifest)
    assert manifest["artifact"]["model_path"] == metadata.artifacts[0].path
    assert manifest["artifact"]["sha256"] == metadata.artifacts[0].sha256
    assert manifest["runtime"]["engine"] == metadata.lab_compat.runtime.engine
    assert manifest["runtime"]["device"] == metadata.lab_compat.runtime.device
    assert manifest["runtime"]["precision"] == metadata.lab_compat.runtime.precision
    assert manifest["runtime"]["artifact_path"] == metadata.lab_compat.runtime.runtime_artifact_path


def validate_metadata_runtime_handoff(metadata: dict[str, Any]) -> None:
    build = require_object(metadata, "build")
    source_model = require_object(metadata, "source_model")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AssertionError("metadata.artifacts must contain at least one artifact")
    artifact = artifacts[0]
    if not isinstance(artifact, dict):
        raise AssertionError("metadata.artifacts[0] must be an object")
    lab_compat = require_object(metadata, "lab_compat")
    runtime = require_object(lab_compat, "runtime")
    preset_snapshot = require_object(metadata, "preset_snapshot")

    require_string(source_model, "path")
    require_string(source_model, "sha256")
    require_string(build, "backend")
    require_string(build, "target")
    require_string(build, "preset_name")
    require_string(build, "build_id")
    require_string(build, "timestamp")
    require_string(artifact, "path")
    require_string(artifact, "sha256")
    require_string(artifact, "format")
    require_string(runtime, "engine")
    require_string(runtime, "device")
    require_string(runtime, "precision")
    require_string(runtime, "runtime_artifact_path")
    require_object(preset_snapshot, "build_options")


def validate_manifest_runtime_handoff(manifest: dict[str, Any]) -> None:
    build = require_object(manifest, "build")
    source_model = require_object(manifest, "source_model")
    artifact = require_object(manifest, "artifact")
    runtime = require_object(manifest, "runtime")
    preset_snapshot = require_object(manifest, "preset_snapshot")
    tool = require_object(manifest, "tool")

    require_string(source_model, "path")
    require_string(source_model, "sha256")
    require_string(build, "backend")
    require_string(build, "target")
    require_string(build, "preset_name")
    require_string(build, "build_id")
    require_string(build, "timestamp")
    require_string(artifact, "path")
    require_string(artifact, "model_path")
    require_string(artifact, "model_name")
    require_string(artifact, "sha256")
    require_string(artifact, "format")
    require_string(artifact, "precision")
    require_string(runtime, "engine")
    require_string(runtime, "device")
    require_string(runtime, "precision")
    require_string(runtime, "model_path")
    require_string(runtime, "artifact_path")
    require_object(preset_snapshot, "build_options")
    require_string(tool, "inferedgeforge_version")

    if artifact["model_path"] != runtime["model_path"]:
        raise AssertionError("artifact.model_path must match runtime.model_path")
    if artifact["path"] != runtime["artifact_path"]:
        raise AssertionError("artifact.path must match runtime.artifact_path")
    if build["backend"] != runtime["engine"]:
        raise AssertionError("build.backend must match runtime.engine")
    if build["target"] != runtime["device"]:
        raise AssertionError("build.target must match runtime.device")


def require_object(data: dict[str, Any], field: str) -> dict[str, Any]:
    value = data.get(field)
    if not isinstance(value, dict):
        raise AssertionError(f"{field} must be an object")
    return value


def require_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{field} must be a non-empty string")
    return value
