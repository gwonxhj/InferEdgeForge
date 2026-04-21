"""Build orchestration helpers for the MVP flow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from inferedgeforge.builders import BuildRequest, get_builder
from inferedgeforge.metadata import write_build_metadata
from inferedgeforge.presets import load_preset_by_id
from inferedgeforge.schemas import (
    ArtifactRecord,
    BuildInfo,
    BuildMetadata,
    SourceModelInfo,
    ValidationHandoff,
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_id(model_path: Path, preset_name: str, backend: str, timestamp: str) -> str:
    safe_timestamp = "".join(char for char in timestamp if char.isalnum())
    preset_suffix = preset_name.rsplit("/", 1)[-1]
    return f"{model_path.stem}-{backend}-{preset_suffix}-{safe_timestamp}"


def create_build_metadata(
    model_path: str | Path,
    preset_name: str,
    backend: str,
    target: str,
    artifact_paths: list[str],
    artifact_format: str,
    handoff_consumer: str = "InferEdgeLab",
    source_sha256: str | None = None,
) -> BuildMetadata:
    source_path = Path(model_path)
    timestamp = _utc_timestamp()
    build_id = _build_id(source_path, preset_name, backend, timestamp)
    artifacts = [
        ArtifactRecord(path=artifact_path, format=artifact_format, role="deployment_model")
        for artifact_path in artifact_paths
    ]

    return BuildMetadata(
        schema_version="0.1.0",
        build=BuildInfo(
            build_id=build_id,
            timestamp=timestamp,
            preset_name=preset_name,
            backend=backend,
            target=target,
        ),
        source_model=SourceModelInfo(
            path=str(source_path),
            format=source_path.suffix.lstrip(".") or "onnx",
            sha256=source_sha256,
        ),
        artifacts=artifacts,
        handoff=ValidationHandoff(consumer=handoff_consumer, ready=True),
    )


def run_build(
    model_path: str | Path,
    preset_id: str,
    output_dir: str | Path,
    presets_root: str | Path = "presets",
) -> BuildMetadata:
    source_path = Path(model_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Model file not found: {source_path}")

    preset = load_preset_by_id(preset_id, presets_root=presets_root)
    builder = get_builder(preset.backend)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    build_dir = output_root / f"{source_path.stem}__{preset.target}__{preset.backend}"
    build_dir.mkdir(parents=True, exist_ok=True)

    request = BuildRequest(
        model_path=str(source_path),
        preset=preset,
        output_dir=str(build_dir),
    )
    result = builder.build(request)

    metadata = create_build_metadata(
        model_path=source_path,
        preset_name=preset.name,
        backend=result.backend,
        target=result.target,
        artifact_paths=result.artifact_paths,
        artifact_format=preset.artifact_format,
    )
    write_build_metadata(metadata, build_dir / "metadata.json")
    return metadata
