from __future__ import annotations

from pathlib import Path

import pytest

from inferedgeforge.presets import load_preset_by_id
from inferedgeforge.schemas import PresetDefinition


def test_preset_definition_from_dict_valid() -> None:
    preset = PresetDefinition.from_dict(
        {
            "name": "rknn/rk3588_fp16",
            "backend": "rknn",
            "target": "rk3588",
            "source_format": "onnx",
            "artifact_format": "rknn",
            "build_options": {"precision": "fp16"},
            "metadata": {"validation_handoff": "inferedgelab"},
        }
    )

    assert preset.name == "rknn/rk3588_fp16"
    assert preset.backend == "rknn"
    assert preset.target == "rk3588"


def test_preset_definition_missing_required_field_raises() -> None:
    with pytest.raises(ValueError, match="artifact_format"):
        PresetDefinition.from_dict(
            {
                "name": "rknn/rk3588_fp16",
                "backend": "rknn",
                "target": "rk3588",
                "source_format": "onnx",
                "build_options": {"precision": "fp16"},
            }
        )


def test_load_preset_by_id_reads_example_preset() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    preset = load_preset_by_id("rknn/rk3588_fp16", presets_root=repo_root / "presets")

    assert preset.backend == "rknn"
    assert preset.target == "rk3588"
    assert preset.build_options["precision"] == "fp16"


def test_load_preset_by_id_missing_file_raises() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        load_preset_by_id("rknn/does_not_exist", presets_root=repo_root / "presets")
