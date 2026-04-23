"""Build orchestration helpers for the MVP flow."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from inferedgeforge.builders import BuildRequest, get_builder
from inferedgeforge.metadata import write_build_metadata
from inferedgeforge.presets import load_preset_by_id, validate_preset_for_build
from inferedgeforge.schemas import (
    ArtifactRecord,
    BuildInfo,
    BuildMetadata,
    LabCompatibility,
    LabRuntimeMapping,
    SourceModelInfo,
    ValidationHandoff,
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_id(model_path: Path, preset_name: str, backend: str, timestamp: str) -> str:
    safe_timestamp = "".join(char for char in timestamp if char.isalnum())
    preset_suffix = preset_name.rsplit("/", 1)[-1]
    return f"{model_path.stem}-{backend}-{preset_suffix}-{safe_timestamp}"


def _compute_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_requested_shape_from_preset_metadata(
    preset_metadata: dict[str, object] | None,
) -> tuple[int | None, int | None, int | None]:
    if not isinstance(preset_metadata, dict):
        return None, None, None

    def _maybe_positive_int(key: str) -> int | None:
        value = preset_metadata.get(key)
        if isinstance(value, int) and value > 0:
            return value
        return None

    return (
        _maybe_positive_int("requested_batch"),
        _maybe_positive_int("requested_height"),
        _maybe_positive_int("requested_width"),
    )


def create_lab_compatibility(
    backend: str,
    target: str,
    preset_precision: str,
    artifact_path: str,
    preset_metadata: dict[str, object] | None = None,
) -> LabCompatibility:
    requested_batch, requested_height, requested_width = _extract_requested_shape_from_preset_metadata(
        preset_metadata
    )
    return LabCompatibility(
        profile_ready=True,
        runtime=LabRuntimeMapping(
            engine=backend,
            device=target,
            precision=preset_precision,
            engine_path=artifact_path,
            runtime_artifact_path=artifact_path,
            requested_batch=requested_batch,
            requested_height=requested_height,
            requested_width=requested_width,
        ),
    )


def create_build_metadata(
    model_path: str | Path,
    preset_name: str,
    backend: str,
    target: str,
    artifact_paths: list[str],
    artifact_format: str,
    handoff_consumer: str = "InferEdgeLab",
    source_sha256: str | None = None,
    preset_metadata: dict[str, object] | None = None,
    preset_build_options: dict[str, object] | None = None,
) -> BuildMetadata:
    source_path = Path(model_path)
    timestamp = _utc_timestamp()
    build_id = _build_id(source_path, preset_name, backend, timestamp)
    artifacts = [
        ArtifactRecord(path=artifact_path, format=artifact_format, role="deployment_model")
        for artifact_path in artifact_paths
    ]
    precision = "unknown"
    if isinstance(preset_build_options, dict):
        raw_precision = preset_build_options.get("precision")
        if isinstance(raw_precision, str) and raw_precision.strip():
            precision = raw_precision

    lab_compat = None
    if artifact_paths:
        lab_compat = create_lab_compatibility(
            backend=backend,
            target=target,
            preset_precision=precision,
            artifact_path=artifact_paths[0],
            preset_metadata=preset_metadata,
        )

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
        lab_compat=lab_compat,
    )


def _preview_build_dir(model_path: Path, target: str, backend: str) -> Path:
    return Path("builds") / f"{model_path.stem}__{target}__{backend}"


def _preview_artifact_path(
    model_path: Path,
    target: str,
    backend: str,
    artifact_format: str,
) -> Path:
    return _preview_build_dir(model_path, target, backend) / f"model.{artifact_format}"


def _preview_metadata_path(model_path: Path, target: str, backend: str) -> Path:
    return _preview_build_dir(model_path, target, backend) / "metadata.json"


def _preview_run_summary_path(model_path: Path, target: str, backend: str) -> Path:
    return _preview_build_dir(model_path, target, backend) / "run_summary.json"


def create_build_plan(
    model_path: str | Path,
    preset_id: str,
    presets_root: str | Path = "presets",
) -> dict[str, object]:
    source_path = Path(model_path)
    preset = load_preset_by_id(preset_id, presets_root=presets_root)
    validate_preset_for_build(preset)
    source_sha256 = _compute_file_sha256(source_path) if source_path.is_file() else None

    artifact_path = _preview_artifact_path(
        model_path=source_path,
        target=preset.target,
        backend=preset.backend,
        artifact_format=preset.artifact_format,
    )
    metadata_path = _preview_metadata_path(source_path, preset.target, preset.backend)
    run_summary_path = _preview_run_summary_path(source_path, preset.target, preset.backend)
    metadata = create_build_metadata(
        model_path=source_path,
        preset_name=preset.name,
        backend=preset.backend,
        target=preset.target,
        artifact_paths=[str(artifact_path)],
        artifact_format=preset.artifact_format,
        source_sha256=source_sha256,
        preset_metadata=preset.metadata,
        preset_build_options=preset.build_options,
    )

    return {
        "model": str(source_path),
        "preset": preset.name,
        "backend": preset.backend,
        "target": preset.target,
        "artifact_format": preset.artifact_format,
        "source_model_sha256": source_sha256,
        "artifact_path_preview": str(artifact_path),
        "metadata_path_preview": str(metadata_path),
        "run_summary_path_preview": str(run_summary_path),
        "lab_profile_preview": to_lab_profile_input(metadata),
    }


def to_lab_profile_input(metadata: BuildMetadata) -> dict[str, object]:
    if metadata.lab_compat is None:
        raise ValueError("Build metadata does not include lab_compat.")

    runtime = metadata.lab_compat.runtime
    return {
        "engine": runtime.engine,
        "device": runtime.device,
        "precision": runtime.precision,
        "engine_path": runtime.engine_path,
        "runtime_artifact_path": runtime.runtime_artifact_path,
        "requested_batch": runtime.requested_batch,
        "requested_height": runtime.requested_height,
        "requested_width": runtime.requested_width,
    }


def to_lab_profile_command(metadata: BuildMetadata) -> str:
    payload = to_lab_profile_input(metadata)
    parts = [
        "python",
        "-m",
        "inferedgelab.cli",
        "profile",
        shlex.quote(metadata.source_model.path),
        "--engine",
        shlex.quote(str(payload["engine"])),
        "--engine-path",
        shlex.quote(str(payload["engine_path"])),
        "--device-name",
        shlex.quote(str(payload["device"])),
        "--precision",
        shlex.quote(str(payload["precision"])),
    ]

    if payload["requested_batch"] is not None:
        parts.extend(["--batch", shlex.quote(str(payload["requested_batch"]))])
    if payload["requested_height"] is not None:
        parts.extend(["--height", shlex.quote(str(payload["requested_height"]))])
    if payload["requested_width"] is not None:
        parts.extend(["--width", shlex.quote(str(payload["requested_width"]))])

    return " ".join(parts)


def _extract_saved_report_path(stdout: str) -> str | None:
    match = re.search(r"^Saved:\s+(.+)$", stdout, flags=re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip()


def _extract_structured_result_path(stdout: str) -> str | None:
    match = re.search(r"^Saved structured result:\s+(.+)$", stdout, flags=re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip()


def run_lab_profile(metadata: BuildMetadata) -> dict[str, object]:
    command = to_lab_profile_command(metadata)

    result = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("InferEdgeLab profile execution failed.")

    return {
        "command": command,
        "returncode": result.returncode,
        "raw_report_path": _extract_saved_report_path(result.stdout),
        "structured_result_path": _extract_structured_result_path(result.stdout),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def write_run_summary(metadata: BuildMetadata, summary: dict[str, object]) -> Path:
    artifact_path = Path(metadata.artifacts[0].path)
    build_dir = artifact_path.parent
    build_dir.mkdir(parents=True, exist_ok=True)

    summary_path = build_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _find_run_summary_path(metadata: BuildMetadata) -> Path:
    artifact_path = Path(metadata.artifacts[0].path)
    return artifact_path.parent / "run_summary.json"


def _load_run_summary_if_present(metadata: BuildMetadata) -> dict[str, object] | None:
    summary_path = _find_run_summary_path(metadata)
    if not summary_path.is_file():
        return None

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Run summary file is not valid JSON: {summary_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Run summary file must contain a JSON object: {summary_path}")
    return payload


def inspect_build_metadata(metadata: BuildMetadata) -> dict[str, object]:
    lab_profile_input: dict[str, object] | None
    lab_profile_command: str | None
    try:
        lab_profile_input = to_lab_profile_input(metadata)
        lab_profile_command = to_lab_profile_command(metadata)
    except ValueError:
        lab_profile_input = None
        lab_profile_command = None

    return {
        "build": metadata.build.to_dict(),
        "source_model": metadata.source_model.to_dict(),
        "artifacts": [artifact.to_dict() for artifact in metadata.artifacts],
        "handoff": metadata.handoff.to_dict(),
        "lab_profile_input": lab_profile_input,
        "lab_profile_command": lab_profile_command,
        "run_summary": _load_run_summary_if_present(metadata),
    }


def format_inspect_summary(metadata: BuildMetadata, run_summary: dict[str, object] | None) -> str:
    artifact_path = metadata.artifacts[0].path if metadata.artifacts else "none"
    lines = [
        "Build:",
        f"  build_id: {metadata.build.build_id}",
        f"  backend: {metadata.build.backend}",
        f"  target: {metadata.build.target}",
        f"  Source SHA256: {metadata.source_model.sha256 or 'unknown'}",
        "Artifact:",
        f"  path: {artifact_path}",
        "Run Status:",
    ]

    if run_summary is None:
        lines.append("  no benchmark run yet")
        lines.extend(
            [
                "Next Step:",
                "  run run-benchmark to generate validation results",
            ]
        )
    else:
        status = run_summary.get("status", "unknown")
        lines.append(f"  status: {status}")
        structured_result_path = run_summary.get("structured_result_path")
        if structured_result_path:
            lines.append(f"  structured_result_path: {structured_result_path}")
        lines.extend(
            [
                "Next Step:",
                "  ready for comparison in InferEdgeLab",
            ]
        )

    return "\n".join(lines)


def run_build(
    model_path: str | Path,
    preset_id: str,
    output_dir: str | Path,
    presets_root: str | Path = "presets",
) -> BuildMetadata:
    source_path = Path(model_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Model file not found: {source_path}")
    source_sha256 = _compute_file_sha256(source_path)

    preset = load_preset_by_id(preset_id, presets_root=presets_root)
    validate_preset_for_build(preset)
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
        source_sha256=source_sha256,
        preset_metadata=preset.metadata,
        preset_build_options=preset.build_options,
    )
    write_build_metadata(metadata, build_dir / "metadata.json")
    return metadata
