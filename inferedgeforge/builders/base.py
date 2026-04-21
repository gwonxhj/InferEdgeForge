"""Shared builder contracts."""

from __future__ import annotations

from dataclasses import dataclass

from inferedgeforge.schemas import PresetDefinition


@dataclass(slots=True)
class BuildRequest:
    model_path: str
    preset: PresetDefinition
    output_dir: str


@dataclass(slots=True)
class BuildResult:
    backend: str
    target: str
    artifact_paths: list[str]
    metadata_path: str | None = None


class BaseBuilder:
    """Base contract for backend-specific artifact builders."""

    backend_name: str = ""

    def build(self, request: BuildRequest) -> BuildResult:
        raise NotImplementedError
