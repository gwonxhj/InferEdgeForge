"""Preset schema definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Preset field '{field_name}' must be a non-empty string.")
    return value


@dataclass(slots=True)
class PresetDefinition:
    """Validated preset contract for build configuration."""

    name: str
    backend: str
    target: str
    source_format: str
    artifact_format: str
    build_options: dict[str, object]
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        self.name = _require_non_empty_string(self.name, "name")
        self.backend = _require_non_empty_string(self.backend, "backend")
        self.target = _require_non_empty_string(self.target, "target")
        self.source_format = _require_non_empty_string(self.source_format, "source_format")
        self.artifact_format = _require_non_empty_string(self.artifact_format, "artifact_format")

        if not isinstance(self.build_options, dict):
            raise ValueError("Preset field 'build_options' must be a dictionary.")

        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("Preset field 'metadata' must be a dictionary when provided.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PresetDefinition":
        if not isinstance(data, dict):
            raise ValueError("Preset data must be a dictionary.")

        return cls(
            name=data.get("name"),
            backend=data.get("backend"),
            target=data.get("target"),
            source_format=data.get("source_format"),
            artifact_format=data.get("artifact_format"),
            build_options=data.get("build_options"),
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "backend": self.backend,
            "target": self.target,
            "source_format": self.source_format,
            "artifact_format": self.artifact_format,
            "build_options": dict(self.build_options),
        }
        if self.metadata is not None:
            data["metadata"] = dict(self.metadata)
        return data
