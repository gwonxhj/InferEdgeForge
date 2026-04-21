"""CLI entry points for InferEdgeForge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from inferedgeforge.build import run_build
from inferedgeforge.presets import load_preset_by_id
from inferedgeforge.schemas import BuildMetadata


def _iter_preset_ids(presets_root: Path) -> list[str]:
    if not presets_root.exists():
        return []

    preset_ids = []
    for path in presets_root.rglob("*.json"):
        relative = path.relative_to(presets_root)
        if len(relative.parts) < 2:
            continue
        backend = relative.parts[0]
        name = relative.with_suffix("").as_posix().split("/", 1)[1]
        preset_ids.append(f"{backend}/{name}")
    return sorted(preset_ids)


def _cmd_list_presets(args: argparse.Namespace) -> int:
    presets_root = Path(args.presets_root)
    for preset_id in _iter_preset_ids(presets_root):
        print(preset_id)
    return 0


def _cmd_show_preset(args: argparse.Namespace) -> int:
    preset = load_preset_by_id(args.preset_id, presets_root=args.presets_root)
    print(json.dumps(preset.to_dict(), indent=2))
    return 0


def _metadata_path(metadata: BuildMetadata) -> Path:
    artifact_path = Path(metadata.artifacts[0].path)
    return artifact_path.parent / "metadata.json"


def _cmd_build(args: argparse.Namespace) -> int:
    metadata = run_build(
        model_path=args.model,
        preset_id=args.preset,
        output_dir=args.output,
        presets_root=args.presets_root,
    )
    print("Build completed")
    print(f"Preset   : {metadata.build.preset_name}")
    print(f"Backend  : {metadata.build.backend}")
    print(f"Target   : {metadata.build.target}")
    print(f"Artifacts: {len(metadata.artifacts)}")
    print(f"Metadata : {_metadata_path(metadata)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inferedgeforge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-presets", help="List available preset identifiers.")
    list_parser.add_argument(
        "--presets-root",
        default="presets",
        help="Preset root directory to scan. Defaults to %(default)s.",
    )
    list_parser.set_defaults(func=_cmd_list_presets)

    show_parser = subparsers.add_parser("show-preset", help="Show a preset definition as JSON.")
    show_parser.add_argument("preset_id", help="Preset identifier in 'backend/name' format.")
    show_parser.add_argument(
        "--presets-root",
        default="presets",
        help="Preset root directory. Defaults to %(default)s.",
    )
    show_parser.set_defaults(func=_cmd_show_preset)

    build_parser = subparsers.add_parser("build", help="Run the MVP build flow.")
    build_parser.add_argument("--model", required=True, help="Path to the source ONNX model.")
    build_parser.add_argument("--preset", required=True, help="Preset identifier in 'backend/name' format.")
    build_parser.add_argument("--output", required=True, help="Output directory for build artifacts.")
    build_parser.add_argument(
        "--presets-root",
        default="presets",
        help="Preset root directory. Defaults to %(default)s.",
    )
    build_parser.set_defaults(func=_cmd_build)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
