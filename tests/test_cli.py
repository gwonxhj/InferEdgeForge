from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

from inferedgeforge.build import run_build, write_run_summary
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


def test_build_dry_run_prints_json_without_creating_output_dir(tmp_path, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "custom-builds"
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
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["model"] == str(model_path)
    assert payload["preset"] == "rknn/rk3588_fp16"
    assert payload["backend"] == "rknn"
    assert payload["target"] == "rk3588"
    assert payload["artifact_format"] == "rknn"
    assert payload["artifact_path_preview"] == "builds/resnet50__rk3588__rknn/model.rknn"
    assert payload["metadata_path_preview"] == "builds/resnet50__rk3588__rknn/metadata.json"
    assert payload["run_summary_path_preview"] == "builds/resnet50__rk3588__rknn/run_summary.json"
    assert payload["lab_profile_preview"]["engine"] == "rknn"
    assert not output_dir.exists()


def test_build_dry_run_with_invalid_preset_returns_error(tmp_path, capsys) -> None:
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    presets_root = tmp_path / "presets"
    preset_dir = presets_root / "rknn"
    preset_dir.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    (preset_dir / "broken.json").write_text(
        json.dumps(
            {
                "name": "rknn/broken",
                "backend": "rknn",
                "target": "rk3588",
                "source_format": "onnx",
                "artifact_format": "engine",
                "build_options": {"precision": "fp16"},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "build",
            "--model",
            str(model_path),
            "--preset",
            "rknn/broken",
            "--output",
            str(output_dir),
            "--presets-root",
            str(presets_root),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "Error:" in captured.err
    assert not output_dir.exists()


def test_run_benchmark_executes_command(tmp_path, monkeypatch, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy", encoding="utf-8")

    class DummyResult:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = "\n".join(
                [
                    "OK",
                    "Saved: reports/test.json",
                    "Saved structured result: results/test.json",
                ]
            )
            self.stderr = ""

    def fake_run(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr("subprocess.run", fake_run)

    run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )

    metadata_path = output_dir / "model__jetson__tensorrt" / "metadata.json"

    exit_code = main(["run-benchmark", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "OK" in captured.out
    lines = captured.out.strip().splitlines()
    summary_start = len(lines) - 1 - lines[::-1].index("{")
    summary = json.loads("\n".join(lines[summary_start:]))
    assert summary["structured_result_path"] == "results/test.json"
    assert summary["summary_file_path"].endswith("builds/model__jetson__tensorrt/run_summary.json")
    assert Path(summary["summary_file_path"]).exists()
    assert summary["status"] == "completed"


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


def test_inspect_build_prints_json(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    expected_sha256 = hashlib.sha256(b"dummy onnx content").hexdigest()
    _install_fake_rknn(monkeypatch)

    run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    metadata_path = output_dir / "resnet50__rk3588__rknn" / "metadata.json"

    exit_code = main(["inspect-build", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["build"]["backend"] == "rknn"
    assert payload["source_model"]["sha256"] == expected_sha256
    assert payload["artifacts"][0]["sha256"] == hashlib.sha256(b"fake rknn artifact").hexdigest()
    assert payload["lab_profile_input"]["engine"] == "rknn"
    assert "python -m inferedgelab.cli profile" in payload["lab_profile_command"]
    assert payload["run_summary"] is None


def test_inspect_build_summary_output(tmp_path, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    expected_sha256 = hashlib.sha256(b"dummy onnx content").hexdigest()

    metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    write_run_summary(
        metadata,
        {
            "command": "python -m inferedgelab.cli profile",
            "returncode": 0,
            "structured_result_path": "results/test.json",
            "summary_file_path": str(output_dir / "model__jetson__tensorrt" / "run_summary.json"),
            "status": "completed",
        },
    )
    metadata_path = output_dir / "model__jetson__tensorrt" / "metadata.json"

    exit_code = main(["inspect-build", "--summary", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Build:" in captured.out
    assert "Artifact:" in captured.out
    assert f"Source SHA256: {expected_sha256}" in captured.out
    assert "Artifact SHA256:" in captured.out
    assert "Run Status:" in captured.out
    assert "status: completed" in captured.out
    assert "structured_result_path: results/test.json" in captured.out
    assert "ready for comparison in InferEdgeLab" in captured.out


def test_inspect_build_summary_without_run(tmp_path, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    expected_sha256 = hashlib.sha256(b"dummy onnx content").hexdigest()

    run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    metadata_path = output_dir / "model__jetson__tensorrt" / "metadata.json"

    exit_code = main(["inspect-build", "--summary", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Build:" in captured.out
    assert "Artifact:" in captured.out
    assert f"Source SHA256: {expected_sha256}" in captured.out
    assert "Artifact SHA256:" in captured.out
    assert "Run Status:" in captured.out
    assert "no benchmark run yet" in captured.out
    assert "run run-benchmark to generate validation results" in captured.out


def test_inspect_build_includes_run_summary_when_present(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")

    metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    summary_path = write_run_summary(
        metadata,
        {
            "command": "python -m inferedgelab.cli profile",
            "returncode": 0,
            "structured_result_path": "results/test.json",
            "summary_file_path": str(output_dir / "model__jetson__tensorrt" / "run_summary.json"),
            "status": "completed",
        },
    )
    metadata_path = output_dir / "model__jetson__tensorrt" / "metadata.json"

    exit_code = main(["inspect-build", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["run_summary"]["command"] == "python -m inferedgelab.cli profile"
    assert payload["run_summary"]["structured_result_path"] == "results/test.json"
    assert payload["run_summary"]["summary_file_path"] == str(summary_path)


def test_inspect_build_without_lab_compat_prints_null_lab_fields(tmp_path, capsys) -> None:
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

    exit_code = main(["inspect-build", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["lab_profile_input"] is None
    assert payload["lab_profile_command"] is None
    assert payload["run_summary"] is None
