"""Utilities for reading and writing build metadata."""

from __future__ import annotations

import json
from pathlib import Path

from inferedgeforge.schemas import BuildMetadata


def write_build_metadata(metadata: BuildMetadata, path: str | Path) -> Path:
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    return metadata_path


def read_build_metadata(path: str | Path) -> BuildMetadata:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Build metadata file not found: {metadata_path}")

    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Build metadata file is not valid JSON: {metadata_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Build metadata file must contain a JSON object: {metadata_path}")

    return BuildMetadata.from_dict(data)
