from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

from inferedgeforge.build import run_build, write_run_summary
from inferedgeforge.cli import build_parser, main


def _expected_build_dir(
    output_dir: Path,
    model_stem: str,
    target: str,
    backend: str,
    preset_name: str,
) -> Path:
    preset_suffix = preset_name.rsplit("/", 1)[-1]
    return output_dir / f"{model_stem}__{target}__{backend}__{preset_suffix}"


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


def _mock_tensorrt_builder_success(monkeypatch) -> None:
    def fake_run_trtexec(self, command):
        save_engine_arg = next(part for part in command if part.startswith("--saveEngine="))
        artifact_path = Path(save_engine_arg.split("=", 1)[1])
        artifact_path.write_text("fake tensorrt engine", encoding="utf-8")

    monkeypatch.setattr(
        "inferedgeforge.builders.tensorrt.TensorRTBuilder._run_trtexec",
        fake_run_trtexec,
    )


def test_list_presets_prints_expected_ids(capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    exit_code = main(["list-presets", "--presets-root", str(repo_root / "presets")])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip().splitlines() == [
        "rknn/rk3588_fp16",
        "tensorrt/jetson_fp16",
        "tensorrt/jetson_fp32",
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


def test_list_builds_groups_builds_by_source_model(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "models" / "test.onnx"
    output_dir = tmp_path / "builds"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch)
    _mock_tensorrt_builder_success(monkeypatch)

    tensorrt_metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    write_run_summary(
        tensorrt_metadata,
        {
            "command": "python -m inferedgelab.cli profile",
            "returncode": 0,
            "structured_result_path": "results/test.json",
            "status": "completed",
        },
    )

    exit_code = main(["list-builds", "--dir", str(output_dir), "--model", str(model_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"Build Group: {model_path}" in captured.out
    assert "tensorrt/jetson_fp16" in captured.out
    assert "rknn/rk3588_fp16" in captured.out
    assert "artifact : " in captured.out
    assert "model.engine" in captured.out
    assert "model.rknn" in captured.out
    assert "status   : ready (benchmark available)" in captured.out
    assert "status   : pending (no benchmark)" in captured.out


def test_show_compare_candidates_lists_ready_variants(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "models" / "test.onnx"
    output_dir = tmp_path / "builds"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch)
    _mock_tensorrt_builder_success(monkeypatch)

    tensorrt_metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    rknn_metadata = run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    for metadata in (tensorrt_metadata, rknn_metadata):
        write_run_summary(
            metadata,
            {
                "command": "python -m inferedgelab.cli profile",
                "returncode": 0,
                "structured_result_path": "results/test.json",
                "status": "completed",
            },
        )

    exit_code = main(["show-compare-candidates", "--dir", str(output_dir)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"Compare Candidates: {model_path}" in captured.out
    assert "Ready" in captured.out
    assert "Pending" in captured.out
    assert "tensorrt/jetson_fp16" in captured.out
    assert "rknn/rk3588_fp16" in captured.out
    assert "engine   : tensorrt" in captured.out
    assert "engine   : rknn" in captured.out
    assert "device   : jetson" in captured.out
    assert "device   : rk3588" in captured.out
    assert "precision: fp16" in captured.out
    assert "- ready builds : 2" in captured.out
    assert "- pending builds : 0" in captured.out
    assert "- next action : use InferEdgeLab compare on ready variants" in captured.out


def test_show_compare_candidates_filters_model_and_shows_pending(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "builds"
    model_path = tmp_path / "models" / "test.onnx"
    other_model_path = tmp_path / "models" / "other.onnx"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    other_model_path.write_text("other dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch)
    _mock_tensorrt_builder_success(monkeypatch)

    tensorrt_metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    run_build(
        model_path=other_model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    write_run_summary(
        tensorrt_metadata,
        {
            "command": "python -m inferedgelab.cli profile",
            "returncode": 0,
            "structured_result_path": "results/test.json",
            "status": "completed",
        },
    )

    exit_code = main(
        [
            "show-compare-candidates",
            "--dir",
            str(output_dir),
            "--model",
            str(model_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"Compare Candidates: {model_path}" in captured.out
    assert str(other_model_path) not in captured.out
    assert "tensorrt/jetson_fp16" in captured.out
    assert "rknn/rk3588_fp16" in captured.out
    assert "status   : benchmark missing" in captured.out
    assert "- ready builds : 1" in captured.out
    assert "- pending builds : 1" in captured.out
    assert "- next action : benchmark one more variant to enable comparison" in captured.out


def test_show_compare_command_previews_requested_pair(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "models" / "test.onnx"
    output_dir = tmp_path / "builds"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch)
    _mock_tensorrt_builder_success(monkeypatch)

    tensorrt_metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    rknn_metadata = run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    write_run_summary(
        tensorrt_metadata,
        {
            "command": "python -m inferedgelab.cli profile",
            "returncode": 0,
            "structured_result_path": "results/tensorrt.json",
            "status": "completed",
        },
    )
    write_run_summary(
        rknn_metadata,
        {
            "command": "python -m inferedgelab.cli profile",
            "returncode": 0,
            "structured_result_path": "results/rknn.json",
            "status": "completed",
        },
    )

    exit_code = main(
        [
            "show-compare-command",
            "--dir",
            str(output_dir),
            "--model",
            str(model_path),
            "--left",
            "rknn/rk3588_fp16",
            "--right",
            "tensorrt/jetson_fp16",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Compare Command Preview" in captured.out
    assert f"Model       : {model_path}" in captured.out
    assert "Left Preset : rknn/rk3588_fp16" in captured.out
    assert "Right Preset: tensorrt/jetson_fp16" in captured.out
    assert (
        "python -m inferedgelab.cli compare results/rknn.json results/tensorrt.json"
        in captured.out
    )
    assert "run the compare command in InferEdgeLab" in captured.out


def test_show_compare_command_uses_distinct_tensorrt_variant_results(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "models" / "yolov8n.onnx"
    output_dir = tmp_path / "builds"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _mock_tensorrt_builder_success(monkeypatch)

    fp16_metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    fp32_metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp32",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    write_run_summary(
        fp16_metadata,
        {
            "command": "python -m inferedgelab.cli profile",
            "returncode": 0,
            "structured_result_path": "results/yolov8n_fp16.json",
            "status": "completed",
        },
    )
    write_run_summary(
        fp32_metadata,
        {
            "command": "python -m inferedgelab.cli profile",
            "returncode": 0,
            "structured_result_path": "results/yolov8n_fp32.json",
            "status": "completed",
        },
    )

    exit_code = main(
        [
            "show-compare-command",
            "--dir",
            str(output_dir),
            "--model",
            str(model_path),
            "--left",
            "tensorrt/jetson_fp16",
            "--right",
            "tensorrt/jetson_fp32",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Left Preset : tensorrt/jetson_fp16" in captured.out
    assert "Right Preset: tensorrt/jetson_fp32" in captured.out
    assert (
        "python -m inferedgelab.cli compare results/yolov8n_fp16.json "
        "results/yolov8n_fp32.json"
    ) in captured.out
    assert "results/yolov8n_fp16.json results/yolov8n_fp16.json" not in captured.out


def test_show_compare_command_requires_two_ready_builds(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "models" / "test.onnx"
    output_dir = tmp_path / "builds"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch)
    _mock_tensorrt_builder_success(monkeypatch)

    tensorrt_metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    write_run_summary(
        tensorrt_metadata,
        {
            "command": "python -m inferedgelab.cli profile",
            "returncode": 0,
            "structured_result_path": "results/tensorrt.json",
            "status": "completed",
        },
    )

    exit_code = main(["show-compare-command", "--dir", str(output_dir), "--model", str(model_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Status: unavailable" in captured.out
    assert "fewer than two benchmark-ready builds are available" in captured.out
    assert "benchmark one more variant to enable comparison" in captured.out
    assert "python -m inferedgelab.cli compare" not in captured.out


def test_show_compare_command_requires_structured_result_paths(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "models" / "test.onnx"
    output_dir = tmp_path / "builds"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch)
    _mock_tensorrt_builder_success(monkeypatch)

    tensorrt_metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    rknn_metadata = run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    write_run_summary(
        tensorrt_metadata,
        {
            "command": "python -m inferedgelab.cli profile",
            "returncode": 0,
            "structured_result_path": "results/tensorrt.json",
            "status": "completed",
        },
    )
    write_run_summary(
        rknn_metadata,
        {
            "command": "python -m inferedgelab.cli profile",
            "returncode": 0,
            "status": "completed",
        },
    )

    exit_code = main(["show-compare-command", "--dir", str(output_dir), "--model", str(model_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Status: unavailable" in captured.out
    assert "structured_result_path is missing for: rknn/rk3588_fp16" in captured.out
    assert "rerun benchmark so structured_result_path is persisted" in captured.out
    assert "python -m inferedgelab.cli compare" not in captured.out


def test_build_parser_constructs() -> None:
    parser = build_parser()

    assert parser.prog == "inferedgeforge"


def test_rebuild_from_manifest_succeeds(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "models" / "test.onnx"
    output_dir = tmp_path / "builds"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _mock_tensorrt_builder_success(monkeypatch)

    run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    manifest_path = _expected_build_dir(
        output_dir,
        "test",
        "jetson",
        "tensorrt",
        "tensorrt/jetson_fp16",
    ) / "manifest.json"

    exit_code = main(["rebuild-from-manifest", str(manifest_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Rebuild completed" in captured.out
    assert f"Manifest : {manifest_path}" in captured.out
    assert "Preset   : tensorrt/jetson_fp16" in captured.out
    assert f"Model    : {model_path}" in captured.out
    assert f"Output   : {output_dir}" in captured.out
    rebuilt_dir = _expected_build_dir(output_dir, "test", "jetson", "tensorrt", "tensorrt/jetson_fp16")
    assert (rebuilt_dir / "metadata.json").exists()
    assert (rebuilt_dir / "manifest.json").exists()


def test_rebuild_from_manifest_requires_preset_name(tmp_path, capsys) -> None:
    model_path = tmp_path / "models" / "test.onnx"
    manifest_path = tmp_path / "builds" / "test__jetson__tensorrt__jetson_fp16" / "manifest.json"
    model_path.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "build": {
                    "backend": "tensorrt",
                    "target": "jetson",
                },
                "source_model": {
                    "path": str(model_path),
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["rebuild-from-manifest", str(manifest_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "manifest is missing preset_name required for rebuild" in captured.err


def test_rebuild_from_manifest_uses_output_override(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "models" / "test.onnx"
    output_dir = tmp_path / "builds"
    rebuild_output_dir = tmp_path / "rebuilt"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _mock_tensorrt_builder_success(monkeypatch)

    run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    manifest_path = _expected_build_dir(
        output_dir,
        "test",
        "jetson",
        "tensorrt",
        "tensorrt/jetson_fp16",
    ) / "manifest.json"

    exit_code = main(
        [
            "rebuild-from-manifest",
            str(manifest_path),
            "--output",
            str(rebuild_output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert f"Output   : {rebuild_output_dir}" in captured.out
    rebuilt_dir = _expected_build_dir(
        rebuild_output_dir,
        "test",
        "jetson",
        "tensorrt",
        "tensorrt/jetson_fp16",
    )
    assert (rebuilt_dir / "metadata.json").exists()
    assert (rebuilt_dir / "manifest.json").exists()


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
    assert payload["artifact_path_preview"] == "builds/resnet50__rk3588__rknn__rk3588_fp16/model.rknn"
    assert payload["metadata_path_preview"] == "builds/resnet50__rk3588__rknn__rk3588_fp16/metadata.json"
    assert payload["run_summary_path_preview"] == "builds/resnet50__rk3588__rknn__rk3588_fp16/run_summary.json"
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


def test_build_tensorrt_builder_error_is_reported(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")

    def fake_run_trtexec(self, command):
        raise RuntimeError("trtexec is required for TensorRT builds but was not found in PATH")

    monkeypatch.setattr(
        "inferedgeforge.builders.tensorrt.TensorRTBuilder._run_trtexec",
        fake_run_trtexec,
    )

    exit_code = main(
        [
            "build",
            "--model",
            str(model_path),
            "--preset",
            "tensorrt/jetson_fp16",
            "--output",
            str(output_dir),
            "--presets-root",
            str(repo_root / "presets"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "trtexec is required for TensorRT builds" in captured.err


def test_run_benchmark_executes_command(tmp_path, monkeypatch, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy", encoding="utf-8")
    _mock_tensorrt_builder_success(monkeypatch)

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

    metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )

    metadata_path = _expected_build_dir(
        output_dir,
        "model",
        "jetson",
        "tensorrt",
        "tensorrt/jetson_fp16",
    ) / "metadata.json"

    exit_code = main(["run-benchmark", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "OK" in captured.out
    lines = captured.out.strip().splitlines()
    summary_start = lines.index("{")
    summary = json.loads("\n".join(lines[summary_start:]))
    assert summary["structured_result_path"] == "results/test.json"
    assert summary["summary_file_path"].endswith(
        "builds/model__jetson__tensorrt__jetson_fp16/run_summary.json"
    )
    assert Path(summary["summary_file_path"]).exists()
    assert summary["status"] == "completed"
    assert summary["build_id"] == metadata.build.build_id
    assert summary["preset_name"] == "tensorrt/jetson_fp16"
    assert summary["backend"] == "tensorrt"
    assert summary["target"] == "jetson"
    assert summary["source_model"]["sha256"] == hashlib.sha256(b"dummy").hexdigest()
    assert summary["primary_artifact"]["sha256"] == metadata.artifacts[0].sha256
    persisted_summary = json.loads(Path(summary["summary_file_path"]).read_text(encoding="utf-8"))
    assert persisted_summary["build_id"] == summary["build_id"]
    assert persisted_summary["primary_artifact"]["sha256"] == summary["primary_artifact"]["sha256"]


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
    metadata_path = _expected_build_dir(
        output_dir,
        "resnet50",
        "rk3588",
        "rknn",
        "rknn/rk3588_fp16",
    ) / "metadata.json"

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
    metadata_path = _expected_build_dir(
        output_dir,
        "resnet50",
        "rk3588",
        "rknn",
        "rknn/rk3588_fp16",
    ) / "metadata.json"

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
    metadata_path = _expected_build_dir(
        output_dir,
        "resnet50",
        "rk3588",
        "rknn",
        "rknn/rk3588_fp16",
    ) / "metadata.json"

    exit_code = main(["inspect-build", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["build"]["backend"] == "rknn"
    assert payload["source_model"]["sha256"] == expected_sha256
    assert payload["artifacts"][0]["sha256"] == hashlib.sha256(b"fake rknn artifact").hexdigest()
    assert payload["preset_snapshot"]["name"] == "rknn/rk3588_fp16"
    assert payload["preset_snapshot"]["backend"] == "rknn"
    assert payload["preset_snapshot"]["target"] == "rk3588"
    assert payload["preset_snapshot"]["build_options"]["precision"] == "fp16"
    assert payload["lab_profile_input"]["engine"] == "rknn"
    assert "python -m inferedgelab.cli profile" in payload["lab_profile_command"]
    assert payload["run_summary"] is None


def test_inspect_build_summary_output(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    expected_sha256 = hashlib.sha256(b"dummy onnx content").hexdigest()
    _mock_tensorrt_builder_success(monkeypatch)

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
            "summary_file_path": str(
                _expected_build_dir(
                    output_dir,
                    "model",
                    "jetson",
                    "tensorrt",
                    "tensorrt/jetson_fp16",
                )
                / "run_summary.json"
            ),
            "status": "completed",
            "engine": "tensorrt",
            "device": "jetson",
            "precision": "fp16",
            "mean_ms": 12.34,
            "p99_ms": 15.67,
        },
    )
    metadata_path = _expected_build_dir(
        output_dir,
        "model",
        "jetson",
        "tensorrt",
        "tensorrt/jetson_fp16",
    ) / "metadata.json"

    exit_code = main(["inspect-build", "--summary", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Build:" in captured.out
    assert "Preset:" in captured.out
    assert "Source Model:" in captured.out
    assert "Artifact:" in captured.out
    assert "Run Status:" in captured.out
    assert "Next Step:" in captured.out
    assert "Run Summary:" in captured.out
    assert "Execution Insight:" in captured.out
    assert "Compare Readiness:" in captured.out
    assert "Compare Context:" in captured.out
    assert "name: tensorrt/jetson_fp16" in captured.out
    assert "precision=fp16" in captured.out
    assert f"path: {model_path}" in captured.out
    assert f"sha256: {expected_sha256}" in captured.out
    assert f"sha256: {metadata.artifacts[0].sha256}" in captured.out
    assert "status: completed" in captured.out
    assert "structured_result_path: results/test.json" in captured.out
    assert "summary_file_path:" in captured.out
    assert "Status     : ready" in captured.out
    assert "benchmark result is present for this artifact" in captured.out
    assert "Workflow   : build another preset variant, then compare results in InferEdgeLab" in captured.out
    assert "Preset     : tensorrt/jetson_fp16" in captured.out
    assert "Backend    : tensorrt" in captured.out
    assert "Target     : jetson" in captured.out
    assert f"Source     : {model_path}" in captured.out
    assert f"Artifact   : {metadata.artifacts[0].path}" in captured.out
    assert "Engine     : tensorrt" in captured.out
    assert "Device     : jetson" in captured.out
    assert "Precision  : fp16" in captured.out
    assert "Mean (ms)  : 12.34" in captured.out
    assert "P99 (ms)   : 15.67" in captured.out
    assert "ready for comparison in InferEdgeLab" in captured.out
    assert "Build another preset variant for the same source model." in captured.out
    assert "Run benchmark for that artifact as well." in captured.out
    assert "Use InferEdgeLab compare or compare-latest to evaluate trade-offs." in captured.out


def test_inspect_build_summary_without_run(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    expected_sha256 = hashlib.sha256(b"dummy onnx content").hexdigest()
    _mock_tensorrt_builder_success(monkeypatch)

    run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )
    metadata_path = _expected_build_dir(
        output_dir,
        "model",
        "jetson",
        "tensorrt",
        "tensorrt/jetson_fp16",
    ) / "metadata.json"

    exit_code = main(["inspect-build", "--summary", str(metadata_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Build:" in captured.out
    assert "Preset:" in captured.out
    assert "Source Model:" in captured.out
    assert "Artifact:" in captured.out
    assert "Run Status:" in captured.out
    assert "Next Step:" in captured.out
    assert "Compare Readiness:" in captured.out
    assert "Compare Context:" in captured.out
    assert f"path: {model_path}" in captured.out
    assert f"sha256: {expected_sha256}" in captured.out
    assert "no benchmark run yet" in captured.out
    assert "Status     : pending" in captured.out
    assert "run-benchmark has not been executed yet" in captured.out
    assert "run run-benchmark to generate validation results" in captured.out


def test_inspect_build_includes_run_summary_when_present(tmp_path, capsys, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "model.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _mock_tensorrt_builder_success(monkeypatch)

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
            "summary_file_path": str(
                _expected_build_dir(
                    output_dir,
                    "model",
                    "jetson",
                    "tensorrt",
                    "tensorrt/jetson_fp16",
                )
                / "run_summary.json"
            ),
            "status": "completed",
        },
    )
    metadata_path = _expected_build_dir(
        output_dir,
        "model",
        "jetson",
        "tensorrt",
        "tensorrt/jetson_fp16",
    ) / "metadata.json"

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
