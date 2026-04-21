"""Preset loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inferedgeforge.schemas import PresetDefinition


def load_preset_file(path: str | Path) -> PresetDefinition:
    preset_path = Path(path)
    if not preset_path.is_file():
        raise FileNotFoundError(f"Preset file not found: {preset_path}")

    try:
        data = json.loads(preset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Preset file is not valid JSON: {preset_path}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Preset file must contain a JSON object: {preset_path}")

    return PresetDefinition.from_dict(data)


def load_preset_by_id(preset_id: str, presets_root: str | Path = "presets") -> PresetDefinition:
    if not isinstance(preset_id, str) or not preset_id.strip():
        raise ValueError("Preset ID must be a non-empty string in 'backend/name' format.")

    parts = preset_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Preset ID must use the format 'backend/name'.")

    backend, name = parts
    preset_path = Path(presets_root) / backend / f"{name}.json"
    if not preset_path.is_file():
        raise FileNotFoundError(f"Preset not found for ID '{preset_id}': {preset_path}")

    preset = load_preset_file(preset_path)
    expected_name = preset_id
    if preset.name != expected_name:
        raise ValueError(
            f"Preset name mismatch for '{preset_id}': expected '{expected_name}', found '{preset.name}'."
        )
    return preset
