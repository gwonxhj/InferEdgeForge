from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from inferedgeforge.builders import BuildRequest, TensorRTBuilder
from inferedgeforge.build import (
    _extract_saved_report_path,
    _extract_structured_result_path,
    _find_run_summary_path,
    _load_run_summary_if_present,
    build_manifest_from_metadata,
    create_build_plan,
    inspect_build_metadata,
    resolve_rebuild_inputs,
    run_build,
    run_lab_profile,
    to_lab_profile_command,
    to_lab_profile_input,
    write_run_summary,
)
from inferedgeforge.cli import main
from inferedgeforge.metadata import read_build_metadata
from inferedgeforge.manifest_validation import validate_manifest
from inferedgeforge.presets import load_preset_by_id
from inferedgeforge.schemas import (
    ArtifactRecord,
    BuildInfo,
    BuildMetadata,
    LabCompatibility,
    LabRuntimeMapping,
    SourceModelInfo,
    ValidationHandoff,
)


def _expected_build_dir(
    output_dir: Path,
    model_stem: str,
    target: str,
    backend: str,
    preset_name: str,
) -> Path:
    preset_suffix = preset_name.rsplit("/", 1)[-1]
    return output_dir / f"{model_stem}__{target}__{backend}__{preset_suffix}"


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


def _mock_tensorrt_builder_success(monkeypatch):
    calls = []

    def fake_run_trtexec(self, command):
        calls.append(command)
        save_engine_arg = next(part for part in command if part.startswith("--saveEngine="))
        artifact_path = Path(save_engine_arg.split("=", 1)[1])
        artifact_path.write_text("fake tensorrt engine", encoding="utf-8")

    monkeypatch.setattr(TensorRTBuilder, "_run_trtexec", fake_run_trtexec)
    return calls


def test_run_build_creates_metadata_and_artifact(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    expected_sha256 = hashlib.sha256(b"dummy onnx content").hexdigest()
    _install_fake_rknn(monkeypatch, tmp_path)

    metadata = run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )

    build_dir = _expected_build_dir(output_dir, "resnet50", "rk3588", "rknn", "rknn/rk3588_fp16")
    assert metadata.schema_version == "0.1.0"
    assert metadata.build.preset_name == "rknn/rk3588_fp16"
    assert metadata.build.backend == "rknn"
    assert metadata.build.target == "rk3588"
    assert metadata.source_model.sha256 == expected_sha256
    assert metadata.artifacts[0].sha256 == hashlib.sha256(b"fake rknn artifact").hexdigest()
    assert metadata.preset_snapshot is not None
    assert metadata.preset_snapshot.name == "rknn/rk3588_fp16"
    assert metadata.preset_snapshot.backend == "rknn"
    assert metadata.preset_snapshot.target == "rk3588"
    assert metadata.preset_snapshot.build_options["precision"] == "fp16"
    assert metadata.preset_snapshot.build_options["target_platform"] == "rk3588"
    assert metadata.preset_snapshot.metadata["validation_handoff"] == "inferedgelab"
    assert metadata.lab_compat is not None
    assert metadata.lab_compat.profile_ready is True
    assert metadata.lab_compat.runtime.engine == "rknn"
    assert metadata.lab_compat.runtime.device == "rk3588"
    assert metadata.lab_compat.runtime.precision == "fp16"
    assert metadata.lab_compat.runtime.engine_path == str(build_dir / "model.rknn")
    assert metadata.lab_compat.runtime.runtime_artifact_path == str(build_dir / "model.rknn")
    assert (build_dir / "metadata.json").exists()
    assert (build_dir / "manifest.json").exists()
    assert (build_dir / "model.rknn").exists()
    persisted_metadata = read_build_metadata(build_dir / "metadata.json")
    assert persisted_metadata.artifacts[0].sha256 == metadata.artifacts[0].sha256
    assert persisted_metadata.preset_snapshot is not None
    assert persisted_metadata.preset_snapshot.to_dict() == metadata.preset_snapshot.to_dict()

    manifest = json.loads((build_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "inferedgeforge-manifest-v1"
    assert manifest["build"]["preset_name"] == "rknn/rk3588_fp16"
    assert manifest["build"]["backend"] == "rknn"
    assert manifest["build"]["target"] == "rk3588"
    assert manifest["source_model"]["path"] == str(model_path)
    assert manifest["source_model"]["sha256"] == expected_sha256
    assert manifest["artifact"]["path"] == str(build_dir / "model.rknn")
    assert manifest["artifact"]["sha256"] == hashlib.sha256(b"fake rknn artifact").hexdigest()
    assert manifest["preset_snapshot"] == metadata.preset_snapshot.to_dict()
    assert "python_version" in manifest["tool"]


def test_build_manifest_from_metadata_uses_existing_metadata_values(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "classifier.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _mock_tensorrt_builder_success(monkeypatch)

    metadata = run_build(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )

    manifest = build_manifest_from_metadata(metadata)

    assert manifest["schema_version"] == "inferedgeforge-manifest-v1"
    assert manifest["build"]["preset_name"] == metadata.build.preset_name
    assert manifest["source_model"]["path"] == metadata.source_model.path
    assert manifest["source_model"]["sha256"] == metadata.source_model.sha256
    assert manifest["artifact"]["path"] == metadata.artifacts[0].path
    assert manifest["artifact"]["sha256"] == metadata.artifacts[0].sha256


def test_validate_manifest_accepts_generated_manifest(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "classifier.onnx"
    output_dir = tmp_path / "builds"
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
        "classifier",
        "jetson",
        "tensorrt",
        "tensorrt/jetson_fp16",
    ) / "manifest.json"

    result = validate_manifest(manifest=manifest_path)

    assert result.valid
    assert result.errors == ()


def test_validate_manifest_reports_actionable_errors(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "build": {"preset_name": "tensorrt/jetson_fp16", "backend": "tensorrt"},
                "source_model": {"path": "models/yolov8n.onnx"},
                "artifact": {"format": "engine"},
            }
        ),
        encoding="utf-8",
    )

    result = validate_manifest(manifest=manifest_path)

    assert not result.valid
    error_paths = {issue.path for issue in result.errors}
    assert "runtime" in error_paths
    assert "artifact.path" in error_paths
    assert "build.target" in error_paths


def test_resolve_rebuild_inputs_uses_manifest_values(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "classifier.onnx"
    output_dir = tmp_path / "builds"
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
        "classifier",
        "jetson",
        "tensorrt",
        "tensorrt/jetson_fp16",
    ) / "manifest.json"
    inputs = resolve_rebuild_inputs(manifest_path)

    assert inputs["model_path"] == model_path
    assert inputs["preset_name"] == "tensorrt/jetson_fp16"
    assert inputs["output_dir"] == output_dir
    assert inputs["manifest_path"] == manifest_path


def test_create_build_plan_returns_expected_preview(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    expected_sha256 = hashlib.sha256(b"dummy onnx content").hexdigest()

    payload = create_build_plan(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        presets_root=repo_root / "presets",
    )

    assert payload["model"] == str(model_path)
    assert payload["preset"] == "rknn/rk3588_fp16"
    assert payload["backend"] == "rknn"
    assert payload["target"] == "rk3588"
    assert payload["artifact_format"] == "rknn"
    assert payload["source_model_sha256"] == expected_sha256
    assert payload["artifact_path_preview"] == "builds/resnet50__rk3588__rknn__rk3588_fp16/model.rknn"
    assert payload["metadata_path_preview"] == "builds/resnet50__rk3588__rknn__rk3588_fp16/metadata.json"
    assert payload["run_summary_path_preview"] == "builds/resnet50__rk3588__rknn__rk3588_fp16/run_summary.json"
    assert payload["lab_profile_preview"] == {
        "engine": "rknn",
        "device": "rk3588",
        "precision": "fp16",
        "engine_path": "builds/resnet50__rk3588__rknn__rk3588_fp16/model.rknn",
        "runtime_artifact_path": "builds/resnet50__rk3588__rknn__rk3588_fp16/model.rknn",
        "requested_batch": None,
        "requested_height": None,
        "requested_width": None,
    }


def test_create_build_plan_preview_paths_share_build_directory(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "test.onnx"
    model_path.write_text("dummy onnx content", encoding="utf-8")

    payload = create_build_plan(
        model_path=model_path,
        preset_id="tensorrt/jetson_fp16",
        presets_root=repo_root / "presets",
    )

    artifact_path = Path(str(payload["artifact_path_preview"]))
    metadata_path = Path(str(payload["metadata_path_preview"]))
    run_summary_path = Path(str(payload["run_summary_path_preview"]))

    assert metadata_path == Path("builds/test__jetson__tensorrt__jetson_fp16/metadata.json")
    assert run_summary_path == Path("builds/test__jetson__tensorrt__jetson_fp16/run_summary.json")
    assert artifact_path.parent == metadata_path.parent == run_summary_path.parent


def test_run_build_uses_preset_specific_build_directories(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "yolov8n.onnx"
    output_dir = tmp_path / "builds"
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

    fp16_dir = _expected_build_dir(output_dir, "yolov8n", "jetson", "tensorrt", "tensorrt/jetson_fp16")
    fp32_dir = _expected_build_dir(output_dir, "yolov8n", "jetson", "tensorrt", "tensorrt/jetson_fp32")

    assert fp16_dir != fp32_dir
    assert Path(fp16_metadata.artifacts[0].path).parent == fp16_dir
    assert Path(fp32_metadata.artifacts[0].path).parent == fp32_dir
    assert (fp16_dir / "metadata.json").exists()
    assert (fp16_dir / "manifest.json").exists()
    assert (fp32_dir / "metadata.json").exists()
    assert (fp32_dir / "manifest.json").exists()


def test_create_build_plan_invalid_preset_raises(tmp_path) -> None:
    model_path = tmp_path / "resnet50.onnx"
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

    with pytest.raises(ValueError, match="artifact_format"):
        create_build_plan(
            model_path=model_path,
            preset_id="rknn/broken",
            presets_root=presets_root,
        )


def test_tensorrt_builder_runs_trtexec_and_returns_engine_artifact(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "classifier.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    preset = load_preset_by_id("tensorrt/jetson_fp16", presets_root=repo_root / "presets")
    calls = []

    class DummyResult:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        save_engine_arg = next(part for part in command if part.startswith("--saveEngine="))
        Path(save_engine_arg.split("=", 1)[1]).write_text(
            "fake tensorrt engine",
            encoding="utf-8",
        )
        return DummyResult()

    monkeypatch.setattr("subprocess.run", fake_run)

    result = TensorRTBuilder().build(
        BuildRequest(
            model_path=str(model_path),
            preset=preset,
            output_dir=str(output_dir),
        )
    )

    artifact_path = output_dir / "model.engine"
    command = calls[0][0]
    assert command[0] == "trtexec"
    assert f"--onnx={model_path}" in command
    assert f"--saveEngine={artifact_path}" in command
    assert "--fp16" in command
    assert "--memPoolSize=workspace:2048" in command
    assert calls[0][1] == {"text": True, "capture_output": True}
    assert artifact_path.exists()
    assert result.backend == "tensorrt"
    assert result.target == "jetson"
    assert result.artifact_paths == [str(artifact_path)]


def test_tensorrt_builder_missing_trtexec_raises(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "classifier.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    preset = load_preset_by_id("tensorrt/jetson_fp16", presets_root=repo_root / "presets")

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("trtexec")

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="trtexec is required"):
        TensorRTBuilder().build(
            BuildRequest(
                model_path=str(model_path),
                preset=preset,
                output_dir=str(output_dir),
            )
        )


def test_tensorrt_builder_subprocess_failure_raises(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "classifier.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    preset = load_preset_by_id("tensorrt/jetson_fp16", presets_root=repo_root / "presets")

    class DummyResult:
        returncode = 1
        stdout = ""
        stderr = "builder failed"

    def fake_run(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="TensorRT engine build failed via trtexec"):
        TensorRTBuilder().build(
            BuildRequest(
                model_path=str(model_path),
                preset=preset,
                output_dir=str(output_dir),
            )
        )


def test_tensorrt_builder_missing_engine_output_raises(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "classifier.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    preset = load_preset_by_id("tensorrt/jetson_fp16", presets_root=repo_root / "presets")

    class DummyResult:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_run(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="completed without producing model.engine"):
        TensorRTBuilder().build(
            BuildRequest(
                model_path=str(model_path),
                preset=preset,
                output_dir=str(output_dir),
            )
        )


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

    build_dir = _expected_build_dir(output_dir, "resnet50", "rk3588", "rknn", "rknn/rk3588_fp16")
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

    build_dir = _expected_build_dir(output_dir, "resnet50", "rk3588", "rknn", "rknn/rk3588_fp16")
    artifact_path = build_dir / "model.rknn"
    metadata_path = build_dir / "metadata.json"
    assert artifact_path.exists()
    assert metadata_path.exists()
    assert metadata.artifacts[0].path == str(artifact_path)
    assert metadata.artifacts[0].sha256 == hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
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


def test_run_build_metadata_still_maps_to_lab_profile_input(tmp_path, monkeypatch) -> None:
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

    assert payload == {
        "engine": "rknn",
        "device": "rk3588",
        "precision": "fp16",
        "engine_path": str(
            _expected_build_dir(output_dir, "resnet50", "rk3588", "rknn", "rknn/rk3588_fp16")
            / "model.rknn"
        ),
        "runtime_artifact_path": str(
            _expected_build_dir(output_dir, "resnet50", "rk3588", "rknn", "rknn/rk3588_fp16")
            / "model.rknn"
        ),
        "requested_batch": None,
        "requested_height": None,
        "requested_width": None,
    }


def test_to_lab_profile_command_returns_expected_command(tmp_path, monkeypatch) -> None:
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

    command = to_lab_profile_command(metadata)

    assert "python -m inferedgelab.cli profile" in command
    assert str(model_path) in command
    assert "--engine rknn" in command
    assert "--engine-path" in command
    assert "--device-name rk3588" in command
    assert "--precision fp16" in command


def test_to_lab_profile_command_includes_shape_hints_when_present(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    presets_root = tmp_path / "presets"
    preset_dir = presets_root / "tensorrt"
    preset_dir.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _mock_tensorrt_builder_success(monkeypatch)
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

    command = to_lab_profile_command(metadata)

    assert "--batch 1" in command
    assert "--height 224" in command
    assert "--width 224" in command


def test_inspect_build_metadata_includes_lab_handoff_details(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _install_fake_rknn(monkeypatch, tmp_path)

    from inferedgeforge.build import inspect_build_metadata

    metadata = run_build(
        model_path=model_path,
        preset_id="rknn/rk3588_fp16",
        output_dir=output_dir,
        presets_root=repo_root / "presets",
    )

    payload = inspect_build_metadata(metadata)

    assert payload["build"]["backend"] == "rknn"
    assert payload["source_model"]["format"] == "onnx"
    assert payload["artifacts"][0]["format"] == "rknn"
    assert payload["artifacts"][0]["sha256"] == metadata.artifacts[0].sha256
    assert payload["preset_snapshot"]["name"] == "rknn/rk3588_fp16"
    assert payload["preset_snapshot"]["backend"] == "rknn"
    assert payload["preset_snapshot"]["target"] == "rk3588"
    assert payload["preset_snapshot"]["build_options"]["precision"] == "fp16"
    assert payload["preset_snapshot"]["metadata"]["validation_handoff"] == "inferedgelab"
    assert payload["handoff"]["consumer"] == "InferEdgeLab"
    assert payload["lab_profile_input"]["engine"] == "rknn"
    assert "python -m inferedgelab.cli profile" in payload["lab_profile_command"]
    assert payload["run_summary"] is None


def test_inspect_build_metadata_without_lab_compat_sets_null_fields() -> None:
    from inferedgeforge.build import inspect_build_metadata

    metadata = BuildMetadata(
        schema_version="0.1.0",
        build=BuildInfo(
            build_id="build-001",
            timestamp="2026-04-23T12:00:00Z",
            preset_name="rknn/rk3588_fp16",
            backend="rknn",
            target="rk3588",
        ),
        source_model=SourceModelInfo(
            path="models/resnet50.onnx",
            format="onnx",
        ),
        artifacts=[
            ArtifactRecord(
                path="artifacts/model.rknn",
                format="rknn",
                role="deployment_model",
            )
        ],
        handoff=ValidationHandoff(consumer="InferEdgeLab", ready=True),
    )

    payload = inspect_build_metadata(metadata)

    assert payload["lab_profile_input"] is None
    assert payload["lab_profile_command"] is None
    assert payload["run_summary"] is None


def test_load_run_summary_if_present_returns_none_when_missing(tmp_path) -> None:
    metadata = BuildMetadata(
        schema_version="0.1.0",
        build=BuildInfo(
            build_id="build-001",
            timestamp="2026-04-23T12:00:00Z",
            preset_name="tensorrt/jetson_fp16",
            backend="tensorrt",
            target="jetson",
        ),
        source_model=SourceModelInfo(
            path="models/test.onnx",
            format="onnx",
        ),
        artifacts=[
            ArtifactRecord(
                path=str(
                    tmp_path / "builds" / "test__jetson__tensorrt__jetson_fp16" / "model.engine"
                ),
                format="engine",
                role="deployment_model",
            )
        ],
        handoff=ValidationHandoff(consumer="InferEdgeLab", ready=True),
    )

    assert _find_run_summary_path(metadata) == (
        tmp_path / "builds" / "test__jetson__tensorrt__jetson_fp16" / "run_summary.json"
    )
    assert _load_run_summary_if_present(metadata) is None


def test_inspect_build_metadata_includes_run_summary_when_present(tmp_path) -> None:
    build_dir = tmp_path / "builds" / "test__jetson__tensorrt__jetson_fp16"
    summary_path = build_dir / "run_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "command": "python -m inferedgelab.cli profile",
                "returncode": 0,
                "structured_result_path": "results/test.json",
                "summary_file_path": str(summary_path),
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    metadata = BuildMetadata(
        schema_version="0.1.0",
        build=BuildInfo(
            build_id="build-001",
            timestamp="2026-04-23T12:00:00Z",
            preset_name="tensorrt/jetson_fp16",
            backend="tensorrt",
            target="jetson",
        ),
        source_model=SourceModelInfo(
            path="models/test.onnx",
            format="onnx",
            sha256="source-sha",
        ),
        artifacts=[
            ArtifactRecord(
                path=str(build_dir / "model.engine"),
                format="engine",
                role="deployment_model",
                sha256="artifact-sha",
            )
        ],
        handoff=ValidationHandoff(consumer="InferEdgeLab", ready=True),
    )

    payload = inspect_build_metadata(metadata)

    assert payload["run_summary"] == {
        "command": "python -m inferedgelab.cli profile",
        "returncode": 0,
        "structured_result_path": "results/test.json",
        "summary_file_path": str(summary_path),
        "status": "completed",
    }


def test_run_lab_profile_failure_raises(monkeypatch) -> None:
    class DummyResult:
        returncode = 1
        stdout = ""
        stderr = "error"

    def fake_run(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr("subprocess.run", fake_run)

    metadata = BuildMetadata(
        schema_version="0.1.0",
        build=BuildInfo(
            build_id="id",
            timestamp="time",
            preset_name="preset",
            backend="backend",
            target="target",
        ),
        source_model=SourceModelInfo(
            path="m.onnx",
            format="onnx",
        ),
        artifacts=[
            ArtifactRecord(
                path="a",
                format="fmt",
                role="role",
            )
        ],
        handoff=ValidationHandoff(consumer="lab", ready=True),
        lab_compat=LabCompatibility(
            profile_ready=True,
            runtime=LabRuntimeMapping(
                engine="backend",
                device="target",
                precision="fp16",
                engine_path="artifact.engine",
                runtime_artifact_path="artifact.engine",
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="InferEdgeLab profile execution failed."):
        run_lab_profile(metadata)


def test_extract_run_lab_profile_paths_from_stdout() -> None:
    stdout = "\n".join(
        [
            "Profile complete",
            "Saved: reports/test.json",
            "Saved structured result: results/test.json",
        ]
    )

    assert _extract_saved_report_path(stdout) == "reports/test.json"
    assert _extract_structured_result_path(stdout) == "results/test.json"


def test_run_lab_profile_success_returns_execution_summary(monkeypatch) -> None:
    class DummyResult:
        returncode = 0
        stdout = "\n".join(
            [
                "Profile complete",
                "Saved: reports/test.json",
                "Saved structured result: results/test.json",
            ]
        )
        stderr = ""

    def fake_run(*args, **kwargs):
        return DummyResult()

    monkeypatch.setattr("subprocess.run", fake_run)

    metadata = BuildMetadata(
        schema_version="0.1.0",
        build=BuildInfo(
            build_id="build-001",
            timestamp="2026-04-23T12:00:00Z",
            preset_name="tensorrt/jetson_fp16",
            backend="tensorrt",
            target="jetson",
        ),
        source_model=SourceModelInfo(
            path="models/test.onnx",
            format="onnx",
        ),
        artifacts=[
            ArtifactRecord(
                path="builds/test__jetson__tensorrt__jetson_fp16/model.engine",
                format="engine",
                role="deployment_model",
            )
        ],
        handoff=ValidationHandoff(consumer="InferEdgeLab", ready=True),
        lab_compat=LabCompatibility(
            profile_ready=True,
            runtime=LabRuntimeMapping(
                engine="tensorrt",
                device="jetson",
                precision="fp16",
                engine_path="builds/test__jetson__tensorrt__jetson_fp16/model.engine",
                runtime_artifact_path="builds/test__jetson__tensorrt__jetson_fp16/model.engine",
            ),
        ),
    )

    payload = run_lab_profile(metadata)

    assert payload["command"] == (
        "python -m inferedgelab.cli profile models/test.onnx --engine tensorrt "
        "--engine-path builds/test__jetson__tensorrt__jetson_fp16/model.engine "
        "--device-name jetson --precision fp16"
    )
    assert payload["returncode"] == 0
    assert payload["raw_report_path"] == "reports/test.json"
    assert payload["structured_result_path"] == "results/test.json"
    assert "Saved: reports/test.json" in str(payload["stdout"])
    assert payload["stderr"] == ""


def test_write_run_summary_persists_json_in_build_directory(tmp_path) -> None:
    build_dir = tmp_path / "builds" / "test__jetson__tensorrt__jetson_fp16"
    metadata = BuildMetadata(
        schema_version="0.1.0",
        build=BuildInfo(
            build_id="build-001",
            timestamp="2026-04-23T12:00:00Z",
            preset_name="tensorrt/jetson_fp16",
            backend="tensorrt",
            target="jetson",
        ),
        source_model=SourceModelInfo(
            path="models/test.onnx",
            format="onnx",
            sha256="source-sha",
        ),
        artifacts=[
            ArtifactRecord(
                path=str(build_dir / "model.engine"),
                format="engine",
                role="deployment_model",
                sha256="artifact-sha",
            )
        ],
        handoff=ValidationHandoff(consumer="InferEdgeLab", ready=True),
        lab_compat=LabCompatibility(
            profile_ready=True,
            runtime=LabRuntimeMapping(
                engine="tensorrt",
                device="jetson",
                precision="fp16",
                engine_path=str(build_dir / "model.engine"),
                runtime_artifact_path=str(build_dir / "model.engine"),
            ),
        ),
    )
    summary = {
        "command": "python -m inferedgelab.cli profile",
        "returncode": 0,
        "raw_report_path": "reports/test.json",
        "structured_result_path": "results/test.json",
        "stdout": "Saved structured result: results/test.json",
        "stderr": "",
    }

    summary_path = write_run_summary(metadata, summary)

    assert summary_path == build_dir / "run_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["command"] == "python -m inferedgelab.cli profile"
    assert payload["structured_result_path"] == "results/test.json"
    assert payload["build_id"] == "build-001"
    assert payload["preset_name"] == "tensorrt/jetson_fp16"
    assert payload["backend"] == "tensorrt"
    assert payload["target"] == "jetson"
    assert payload["source_model"]["path"] == "models/test.onnx"
    assert payload["source_model"]["sha256"] == "source-sha"
    assert payload["primary_artifact"]["path"] == str(build_dir / "model.engine")
    assert payload["primary_artifact"]["sha256"] == "artifact-sha"
    assert payload["summary_file_path"] == str(summary_path)
    assert payload["status"] == "completed"
    assert summary["build_id"] == "build-001"
    assert summary["primary_artifact"]["sha256"] == "artifact-sha"


def test_run_build_propagates_shape_hints_into_lab_compat(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "resnet50.onnx"
    output_dir = tmp_path / "builds"
    presets_root = tmp_path / "presets"
    preset_dir = presets_root / "tensorrt"
    preset_dir.mkdir(parents=True)
    model_path.write_text("dummy onnx content", encoding="utf-8")
    _mock_tensorrt_builder_success(monkeypatch)
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
