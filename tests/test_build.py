from __future__ import annotations

from pathlib import Path

import pytest

from inferedgeforge.build import run_build
from inferedgeforge.cli import main


def test_run_build_creates_metadata_and_artifact(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")

    metadata = run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )

    build_dir = output_dir / "resnet50__rk3588__rknn"
    assert metadata.schema_version == "0.1.0"
    assert metadata.build.preset_name == "rknn/rk3588_fp16"
    assert metadata.build.backend == "rknn"
    assert metadata.build.target == "rk3588"
    assert (build_dir / "metadata.json").exists()
    assert (build_dir / "model.rknn").exists()


def test_run_build_with_tensorrt_preset_creates_engine_artifact(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "classifier.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")

    metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )

    build_dir = output_dir / "classifier__jetson__tensorrt"
    assert (build_dir / "model.engine").exists()
    assert metadata.build.backend == "tensorrt"
    assert metadata.build.target == "jetson"


def test_run_build_missing_model_raises(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        run_build(
            model_path=tmp_path / "missing.onnx",
            preset_id="rknn/rk3588_fp16",
            output_dir=tmp_path / "builds",
            presets_root=repo_root / "presets",
        )


def test_cli_build_command_succeeds(tmp_path, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")

    exit_code = main(
        [
            "build",
            "--model",
            str(model_path),
            "--preset",
            "rknn/rk3588_fp16",
            "--output",
            str(output_dir),
            "--presets-root",
            str(repo_root / "presets"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Build completed" in captured.out
    assert "Metadata :" in captured.out


def test_cli_build_unknown_preset_fails(tmp_path, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")

    exit_code = main(
        [
            "build",
            "--model",
            str(model_path),
            "--preset",
            "rknn/unknown_preset",
            "--output",
            str(output_dir),
            "--presets-root",
            str(repo_root / "presets"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.err
