from __future__ import annotations

import json

import pytest

from inferedgeforge.metadata import read_build_metadata, write_build_metadata
from inferedgeforge.schemas import (
    ArtifactRecord,
    BuildInfo,
    BuildMetadata,
    SourceModelInfo,
    ValidationHandoff,
)


def _sample_metadata() -> BuildMetadata:
    return BuildMetadata(
        schema_version="0.1.0",
        build=BuildInfo(
            build_id="build-001",
            timestamp="2026-04-22T12:00:00Z",
            preset_name="rknn/rk3588_fp16",
            backend="rknn",
            target="rk3588",
        ),
        source_model=SourceModelInfo(
            path="models/resnet50.onnx",
            format="onnx",
            sha256="abc123",
        ),
        artifacts=[
            ArtifactRecord(
                path="artifacts/resnet50/model.rknn",
                format="rknn",
                role="deployment_model",
            )
        ],
        handoff=ValidationHandoff(consumer="InferEdgeLab", ready=True),
    )


def test_artifact_record_valid() -> None:
    artifact = ArtifactRecord(path="artifacts/model.rknn", format="rknn", role="deployment_model")

    assert artifact.path == "artifacts/model.rknn"
    assert artifact.format == "rknn"
    assert artifact.role == "deployment_model"


def test_artifact_record_empty_path_raises() -> None:
    with pytest.raises(ValueError, match="path"):
        ArtifactRecord(path="", format="rknn", role="deployment_model")


def test_build_metadata_to_dict_structure() -> None:
    metadata = _sample_metadata()

    payload = metadata.to_dict()

    assert payload["schema_version"] == "0.1.0"
    assert payload["build"]["backend"] == "rknn"
    assert payload["source_model"]["format"] == "onnx"
    assert payload["artifacts"][0]["role"] == "deployment_model"
    assert payload["handoff"]["consumer"] == "InferEdgeLab"


def test_build_metadata_from_dict_roundtrip() -> None:
    original = _sample_metadata()

    restored = BuildMetadata.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()


def test_write_and_read_build_metadata(tmp_path) -> None:
    metadata = _sample_metadata()
    metadata_path = tmp_path / "artifacts" / "build-metadata.json"

    written_path = write_build_metadata(metadata, metadata_path)
    restored = read_build_metadata(written_path)

    assert written_path == metadata_path
    assert restored.to_dict() == metadata.to_dict()


def test_read_build_metadata_invalid_shape_raises(tmp_path) -> None:
    metadata_path = tmp_path / "invalid.json"
    metadata_path.write_text(json.dumps({"schema_version": "0.1.0"}), encoding="utf-8")

    with pytest.raises(ValueError, match="BuildInfo data must be a dictionary|artifacts"):
        read_build_metadata(metadata_path)
