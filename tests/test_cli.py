from __future__ import annotations

import json
from pathlib import Path

from inferedgeforge.cli import build_parser, main


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
