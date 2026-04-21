"""Build metadata schema definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Metadata field '{field_name}' must be a non-empty string.")
    return value


@dataclass(slots=True)
class ArtifactRecord:
    """Describes a generated artifact."""

    path: str
    format: str
    role: str

    def __post_init__(self) -> None:
        self.path = _require_non_empty_string(self.path, "path")
        self.format = _require_non_empty_string(self.format, "format")
        self.role = _require_non_empty_string(self.role, "role")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRecord":
        if not isinstance(data, dict):
            raise ValueError("ArtifactRecord data must be a dictionary.")

        return cls(
            path=data.get("path"),
            format=data.get("format"),
            role=data.get("role"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "format": self.format,
            "role": self.role,
        }


@dataclass(slots=True)
class BuildInfo:
    """Core traceability information for a build."""

    build_id: str
    timestamp: str
    preset_name: str
    backend: str
    target: str

    def __post_init__(self) -> None:
        self.build_id = _require_non_empty_string(self.build_id, "build.build_id")
        self.timestamp = _require_non_empty_string(self.timestamp, "build.timestamp")
        self.preset_name = _require_non_empty_string(self.preset_name, "build.preset_name")
        self.backend = _require_non_empty_string(self.backend, "build.backend")
        self.target = _require_non_empty_string(self.target, "build.target")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuildInfo":
        if not isinstance(data, dict):
            raise ValueError("BuildInfo data must be a dictionary.")

        return cls(
            build_id=data.get("build_id"),
            timestamp=data.get("timestamp"),
            preset_name=data.get("preset_name"),
            backend=data.get("backend"),
            target=data.get("target"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "build_id": self.build_id,
            "timestamp": self.timestamp,
            "preset_name": self.preset_name,
            "backend": self.backend,
            "target": self.target,
        }


@dataclass(slots=True)
class SourceModelInfo:
    """Source model information used to produce artifacts."""

    path: str
    format: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        self.path = _require_non_empty_string(self.path, "source_model.path")
        self.format = _require_non_empty_string(self.format, "source_model.format")
        if self.sha256 is not None:
            self.sha256 = _require_non_empty_string(self.sha256, "source_model.sha256")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceModelInfo":
        if not isinstance(data, dict):
            raise ValueError("SourceModelInfo data must be a dictionary.")

        return cls(
            path=data.get("path"),
            format=data.get("format"),
            sha256=data.get("sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        data: dict[str, str] = {
            "path": self.path,
            "format": self.format,
        }
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        return data


@dataclass(slots=True)
class ValidationHandoff:
    """Indicates whether metadata is ready for downstream handoff."""

    consumer: str
    ready: bool

    def __post_init__(self) -> None:
        self.consumer = _require_non_empty_string(self.consumer, "handoff.consumer")
        if not isinstance(self.ready, bool):
            raise ValueError("Metadata field 'handoff.ready' must be a boolean.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationHandoff":
        if not isinstance(data, dict):
            raise ValueError("ValidationHandoff data must be a dictionary.")

        return cls(
            consumer=data.get("consumer"),
            ready=data.get("ready"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "consumer": self.consumer,
            "ready": self.ready,
        }


@dataclass(slots=True)
class BuildMetadata:
    """Structured metadata emitted for a build."""

    schema_version: str
    build: BuildInfo
    source_model: SourceModelInfo
    artifacts: list[ArtifactRecord]
    handoff: ValidationHandoff

    def __post_init__(self) -> None:
        self.schema_version = _require_non_empty_string(self.schema_version, "schema_version")

        if not isinstance(self.build, BuildInfo):
            raise ValueError("Metadata field 'build' must be a BuildInfo instance.")
        if not isinstance(self.source_model, SourceModelInfo):
            raise ValueError("Metadata field 'source_model' must be a SourceModelInfo instance.")
        if not isinstance(self.artifacts, list):
            raise ValueError("Metadata field 'artifacts' must be a list of ArtifactRecord instances.")
        if not all(isinstance(artifact, ArtifactRecord) for artifact in self.artifacts):
            raise ValueError("Metadata field 'artifacts' must contain only ArtifactRecord instances.")
        if not isinstance(self.handoff, ValidationHandoff):
            raise ValueError("Metadata field 'handoff' must be a ValidationHandoff instance.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuildMetadata":
        if not isinstance(data, dict):
            raise ValueError("BuildMetadata data must be a dictionary.")

        artifacts = data.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("Metadata field 'artifacts' must be a list.")

        return cls(
            schema_version=data.get("schema_version"),
            build=BuildInfo.from_dict(data.get("build")),
            source_model=SourceModelInfo.from_dict(data.get("source_model")),
            artifacts=[ArtifactRecord.from_dict(item) for item in artifacts],
            handoff=ValidationHandoff.from_dict(data.get("handoff")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "build": self.build.to_dict(),
            "source_model": self.source_model.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "handoff": self.handoff.to_dict(),
        }
