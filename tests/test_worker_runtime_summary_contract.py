from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inferedgeforge.handoff import build_worker_runtime_summary
from inferedgeforge.handoff import metadata_to_worker_runtime_summary
from inferedgeforge.schemas import BuildMetadata


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_metadata_and_manifest_project_to_worker_runtime_summary() -> None:
    metadata = BuildMetadata.from_dict(load_fixture("runtime_handoff_metadata.json"))
    manifest = load_fixture("runtime_handoff_manifest.json")

    summary = build_worker_runtime_summary(
        metadata,
        manifest,
        metadata_path="builds/yolov8n__jetson__tensorrt__jetson_fp16/metadata.json",
        manifest_path="builds/yolov8n__jetson__tensorrt__jetson_fp16/manifest.json",
    )

    assert summary == load_fixture("worker_runtime_summary.json")
    validate_worker_runtime_summary(summary)


def test_summary_preserves_source_and_artifact_hashes() -> None:
    metadata = load_fixture("runtime_handoff_metadata.json")
    manifest = load_fixture("runtime_handoff_manifest.json")

    summary = metadata_to_worker_runtime_summary(metadata, manifest)

    assert summary["source_model_sha256"] == metadata["source_model"]["sha256"]
    assert summary["artifact_sha256"] == metadata["artifacts"][0]["sha256"]
    assert summary["artifact_sha256"] == manifest["artifact"]["sha256"]


def test_summary_preserves_runtime_shape_and_target_fields() -> None:
    summary = build_worker_runtime_summary(
        load_fixture("runtime_handoff_metadata.json"),
        load_fixture("runtime_handoff_manifest.json"),
    )

    assert summary["backend"] == "tensorrt"
    assert summary["target"] == "jetson"
    assert summary["precision"] == "fp16"
    assert summary["batch"] == 1
    assert summary["height"] == 640
    assert summary["width"] == 640


def test_summary_contains_lab_worker_request_input_fields() -> None:
    summary = build_worker_runtime_summary(
        load_fixture("runtime_handoff_metadata.json"),
        load_fixture("runtime_handoff_manifest.json"),
        metadata_path="artifacts/metadata.json",
        manifest_path="artifacts/manifest.json",
    )

    input_summary = {
        "workflow": "analyze",
        "model_path": summary["source_model_path"],
        "artifact_path": summary["artifact_path"],
        "metadata_path": summary["metadata_path"],
        "manifest_path": summary["manifest_path"],
        "options": {
            "backend": summary["backend"],
            "target": summary["target"],
            "precision": summary["precision"],
            "batch": summary["batch"],
            "height": summary["height"],
            "width": summary["width"],
        },
    }

    assert input_summary["model_path"] == "models/yolov8n.onnx"
    assert input_summary["artifact_path"].endswith("model.engine")
    assert input_summary["options"]["backend"] == "tensorrt"
    assert input_summary["options"]["height"] == 640


def test_summary_contains_runtime_invocation_fields() -> None:
    summary = build_worker_runtime_summary(
        load_fixture("runtime_handoff_metadata.json"),
        load_fixture("runtime_handoff_manifest.json"),
    )

    runtime_config = {
        "model_path": summary["artifact_path"],
        "source_model_path": summary["source_model_path"],
        "engine": summary["backend"],
        "device": summary["target"],
        "precision": summary["precision"],
        "batch": summary["batch"],
        "height": summary["height"],
        "width": summary["width"],
    }

    assert runtime_config == {
        "model_path": "builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine",
        "source_model_path": "models/yolov8n.onnx",
        "engine": "tensorrt",
        "device": "jetson",
        "precision": "fp16",
        "batch": 1,
        "height": 640,
        "width": 640,
    }


def validate_worker_runtime_summary(summary: dict[str, Any]) -> None:
    for field in (
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
    ):
        value = summary.get(field)
        if not isinstance(value, str) or not value:
            raise AssertionError(f"{field} must be a non-empty string")

    for field in ("batch", "height", "width"):
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AssertionError(f"{field} must be a positive integer")
