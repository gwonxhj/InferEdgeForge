from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from inferedgeforge.build import run_build, to_lab_profile_input
from inferedgeforge.cli import main


def _install_fake_rknn(monkeypatch, tmp_path: Path) -> None:
    class FakeRKNN:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def config(self, **kwargs):
            self.calls.append(("config", kwargs))
            return 0

        def load_onnx(self, model: str):
            self.calls.append(("load_onnx", model))
            return 0

        def build(self, do_quantization: bool = False):
            self.calls.append(("build", do_quantization))
            return 0

        def export_rknn(self, path: str):
            self.calls.append(("export_rknn", path))
            Path(path).write_text("fake rknn artifact", encoding="utf-8")
            return 0

        def release(self):
            self.calls.append(("release", None))
            return None

    api_module = types.ModuleType("rknn.api")
    api_module.RKNN = FakeRKNN
    package_module = types.ModuleType("rknn")
    package_module.api = api_module
    monkeypatch.setitem(sys.modules, "rknn", package_module)
    monkeypatch.setitem(sys.modules, "rknn.api", api_module)


def test_run_build_creates_metadata_and_artifact(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch, tmp_path)

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
    assert metadata.lab_compat is not None
    assert metadata.lab_compat.profile_ready is True
    assert metadata.lab_compat.runtime.engine == "rknn"
    assert metadata.lab_compat.runtime.device == "rk3588"
    assert metadata.lab_compat.runtime.precision == "fp16"
    assert metadata.lab_compat.runtime.engine_path == str(build_dir / "model.rknn")
    assert metadata.lab_compat.runtime.runtime_artifact_path == str(build_dir / "model.rknn")
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


def test_run_build_rknn_toolkit_unavailable_raises(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "rknn", raising=False)
    monkeypatch.delitem(sys.modules, "rknn.api", raising=False)

    with pytest.raises(RuntimeError, match="RKNN toolkit is unavailable"):
        run_build(
            model_path=model_path,
            preset_id="rknn/rk3588_fp16",
            output_dir=output_dir,
            presets_root=repo_root / "presets",
        )

    build_dir = output_dir / "resnet50__rk3588__rknn"
    assert not (build_dir / "model.rknn").exists()
    assert not (build_dir / "metadata.json").exists()


def test_run_build_rknn_stub_creates_real_artifact_and_metadata(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch, tmp_path)

    metadata = run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )

    build_dir = output_dir / "resnet50__rk3588__rknn"
    artifact_path = build_dir / "model.rknn"
    metadata_path = build_dir / "metadata.json"
    assert artifact_path.exists()
    assert metadata_path.exists()
    assert metadata.artifacts[0].path == str(artifact_path)
    assert metadata.build.backend == "rknn"
    assert metadata.build.target == "rk3588"


def test_to_lab_profile_input_returns_expected_mapping(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch, tmp_path)

    metadata = run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )

    payload = to_lab_profile_input(metadata)

    assert payload["engine"] == "rknn"
    assert payload["device"] == "rk3588"
    assert payload["precision"] == "fp16"
    assert payload["engine_path"].endswith("model.rknn")
    assert payload["runtime_artifact_path"].endswith("model.rknn")


def test_run_build_propagates_shape_hints_into_lab_compat(tmp_path) -> None:
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    presets_root = tmp_path / "presets"
    preset_dir = presets_root / "tensorrt"
    preset_dir.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    (preset_dir / "jetson_shape.json").write_text(
        json.dumps(
            {
                "name": "tensorrt/jetson_shape",
                "backend": "tensorrt",
                "target": "jetson",
                "source_format": "onnx",
                "artifact_format": "engine",
                "build_options": {"precision": "fp16", "workspace_mb": 2048},
                "metadata": {
                    "requested_batch": 1,
                    "requested_height": 224,
                    "requested_width": 224,
                },
            }
        ),
        encoding="utf-8",
    )

    metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_shape",
        output_dir=output_dir,
        presets_root=presets_root,
    )

    assert metadata.lab_compat is not None
    assert metadata.lab_compat.runtime.requested_batch == 1
    assert metadata.lab_compat.runtime.requested_height == 224
    assert metadata.lab_compat.runtime.requested_width == 224


def test_run_build_missing_model_raises(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        run_build(
            model_path=tmp_path / "missing.onnx",
            preset_id="rknn/rk3588_fp16",
            output_dir=tmp_path / "builds",
            presets_root=repo_root / "presets",
        )


def test_cli_build_command_succeeds(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch, tmp_path)

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


def test_run_build_invalid_preset_fails_before_creating_output(tmp_path) -> None:
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    presets_root = tmp_path / "presets"
    preset_dir = presets_root / "rknn"
    preset_dir.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    (preset_dir / "invalid.json").write_text(
        json.dumps(
            {
                "name": "rknn/invalid",
                "backend": "rknn",
                "target": "jetson",
                "source_format": "onnx",
                "artifact_format": "rknn",
                "build_options": {"precision": "fp16"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target"):
        run_build(
            model_path=model_path,
            preset_id="rknn/invalid",
            output_dir=output_dir,
            presets_root=presets_root,
        )

    build_dir = output_dir / "resnet50__jetson__rknn"
    assert not build_dir.exists()
    assert not (output_dir / "metadata.json").exists()
