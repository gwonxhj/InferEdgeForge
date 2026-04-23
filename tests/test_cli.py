from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from inferedgeforge.build import run_build
from inferedgeforge.cli import build_parser, main


def _install_fake_rknn(monkeypatch) -> None:
    class FakeRKNN:
        def config(self, **kwargs):
            return 0

        def load_onnx(self, model: str):
            return 0

        def build(self, do_quantization: bool = False):
            return 0

        def export_rknn(self, path: str):
            Path(path).write_text("fake rknn artifact", encoding="utf-8")
            return 0

        def release(self):
            return None

    api_module = types.ModuleType("rknn.api")
    api_module.RKNN = FakeRKNN
    package_module = types.ModuleType("rknn")
    package_module.api = api_module
    monkeypatch.setitem(sys.modules, "rknn", package_module)
    monkeypatch.setitem(sys.modules, "rknn.api", api_module)


def test_list_presets_prints_expected_ids(capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    exit_code = main(["list-presets", "--presets-root", str(repo_root / "presets")])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip().splitlines() == [
        "rknn/rk3588_fp16",
        "tensorrt/jetson_fp16",
    ]


def test_show_preset_prints_json(capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    exit_code = main(
        [
            "show-preset",
            "rknn/rk3588_fp16",
            "--presets-root",
            str(repo_root / "presets"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["backend"] == "rknn"
    assert payload["target"] == "rk3588"


def test_build_parser_constructs() -> None:
    parser = build_parser()

    assert parser.prog == "inferedgeforge"


def test_show_lab_profile_input_prints_json(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch)

    metadata = run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    metadata_path = output_dir / "resnet50__rk3588__rknn" / "metadata.json"

    exit_code = main(["show-lab-profile-input", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["engine"] == "rknn"
    assert payload["device"] == "rk3588"
    assert payload["precision"] == "fp16"
    assert payload["engine_path"].endswith("model.rknn")
    assert payload["runtime_artifact_path"].endswith("model.rknn")
    assert metadata.lab_compat is not None


def test_show_lab_profile_input_fails_without_lab_compat(tmp_path, capsys) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "build": {
                    "build_id": "build-001",
                    "timestamp": "2026-04-23T12:00:00Z",
                    "preset_name": "rknn/rk3588_fp16",
                    "backend": "rknn",
                    "target": "rk3588",
                },
                "source_model": {
                    "path": "models/resnet50.onnx",
                    "format": "onnx",
                },
                "artifacts": [
                    {
                        "path": "artifacts/model.rknn",
                        "format": "rknn",
                        "role": "deployment_model",
                    }
                ],
                "handoff": {
                    "consumer": "InferEdgeLab",
                    "ready": True,
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["show-lab-profile-input", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.err


def test_show_lab_profile_command_prints_command(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch)

    run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    metadata_path = output_dir / "resnet50__rk3588__rknn" / "metadata.json"

    exit_code = main(["show-lab-profile-command", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "python -m inferedgelab.cli profile" in captured.out
    assert "--engine rknn" in captured.out
    assert "--device-name rk3588" in captured.out
    assert "--precision fp16" in captured.out


def test_show_lab_profile_command_fails_without_lab_compat(tmp_path, capsys) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "build": {
                    "build_id": "build-001",
                    "timestamp": "2026-04-23T12:00:00Z",
                    "preset_name": "rknn/rk3588_fp16",
                    "backend": "rknn",
                    "target": "rk3588",
                },
                "source_model": {
                    "path": "models/resnet50.onnx",
                    "format": "onnx",
                },
                "artifacts": [
                    {
                        "path": "artifacts/model.rknn",
                        "format": "rknn",
                        "role": "deployment_model",
                    }
                ],
                "handoff": {
                    "consumer": "InferEdgeLab",
                    "ready": True,
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["show-lab-profile-command", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.err
