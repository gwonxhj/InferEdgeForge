"""Preset schema definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_BACKENDS = ("rknn", "tensorrt")
SUPPORTED_SOURCE_FORMATS = ("onnx",)
BACKEND_ARTIFACT_FORMATS = {
    "rknn": "rknn",
    "tensorrt": "engine",
}
BACKEND_TARGETS = {
    "rknn": "rk3588",
    "tensorrt": "jetson",
}


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
        if not self.build_options:
            raise ValueError("Preset field 'build_options' must not be empty.")

        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("Preset field 'metadata' must be a dictionary when provided.")

        self.validate_for_build()

    def validate_for_build(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            supported = ", ".join(SUPPORTED_BACKENDS)
            raise ValueError(
                f"Preset field 'backend' must be one of: {supported}. Got '{self.backend}'."
            )

        if self.source_format not in SUPPORTED_SOURCE_FORMATS:
            supported = ", ".join(SUPPORTED_SOURCE_FORMATS)
            raise ValueError(
                f"Preset field 'source_format' must be one of: {supported}. Got '{self.source_format}'."
            )

        expected_artifact_format = BACKEND_ARTIFACT_FORMATS[self.backend]
        if self.artifact_format != expected_artifact_format:
            raise ValueError(
                "Preset field 'artifact_format' must match backend "
                f"'{self.backend}': expected '{expected_artifact_format}', got '{self.artifact_format}'."
            )

        expected_target = BACKEND_TARGETS[self.backend]
        if self.target != expected_target:
            raise ValueError(
                f"Preset field 'target' must match backend '{self.backend}': "
                f"expected '{expected_target}', got '{self.target}'."
            )

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
