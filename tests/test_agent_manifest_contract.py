from __future__ import annotations

import json
from pathlib import Path

from inferedgeforge.agent_manifest import (
    AGENT_MANIFEST_SCHEMA_VERSION,
    create_agent_manifest_from_manifest,
    validate_agent_manifest,
)
from inferedgeforge.cli import main


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_agent_manifest_fixture_satisfies_contract() -> None:
    result = validate_agent_manifest(FIXTURES / "agent_manifest_vision.json")

    assert result.valid, result.issues
    assert result.errors == ()


def test_create_agent_manifest_from_existing_manifest_preserves_handoff_context() -> None:
    payload = create_agent_manifest_from_manifest(
        manifest_path=FIXTURES / "runtime_handoff_manifest.json",
        agent_id="vision_detector",
        agent_type="vision",
        priority=90,
        latency_budget_ms=33,
        deadline_ms=40,
        input_type="frame",
        output_type="detections",
        fallback_mode="drop_stale",
    )

    assert payload["schema_version"] == AGENT_MANIFEST_SCHEMA_VERSION
    assert payload["agent_id"] == "vision_detector"
    assert payload["agent_type"] == "vision"
    assert payload["required_backend"] == "tensorrt"
    assert payload["device_target"] == "jetson"
    assert payload["precision"] == "fp16"
    assert payload["runtime_artifact_path"] == (
        "builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine"
    )
    assert payload["lab_compat"] == {
        "source_manifest_path": str(FIXTURES / "runtime_handoff_manifest.json"),
        "runtime_artifact_path": "builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine",
        "required_backend": "tensorrt",
        "device_target": "jetson",
        "precision": "fp16",
    }


def test_validate_agent_manifest_rejects_invalid_agent_type(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "agent_manifest_vision.json").read_text(encoding="utf-8"))
    payload["agent_type"] = "general_ai_os"
    manifest_path = tmp_path / "agent_manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_agent_manifest(manifest_path)

    assert not result.valid
    assert any(issue.path == "agent_type" for issue in result.errors)


def test_create_agent_manifest_cli_writes_valid_manifest(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "agent_manifest.json"

    exit_code = main(
        [
            "create-agent-manifest",
            "--manifest",
            str(FIXTURES / "runtime_handoff_manifest.json"),
            "--agent-id",
            "vision_detector",
            "--agent-type",
            "vision",
            "--priority",
            "90",
            "--latency-budget-ms",
            "33",
            "--deadline-ms",
            "40",
            "--input-type",
            "frame",
            "--output-type",
            "detections",
            "--fallback-mode",
            "drop_stale",
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert f"Agent manifest written: {output_path}" in captured.out
    assert validate_agent_manifest(output_path).valid


def test_validate_agent_manifest_cli_reports_valid(capsys) -> None:
    exit_code = main(["validate-agent-manifest", "--manifest", str(FIXTURES / "agent_manifest_vision.json")])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "Agent manifest validation: valid" in captured.out
    assert "Errors: 0" in captured.out

