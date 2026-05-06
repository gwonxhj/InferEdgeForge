"""CLI entry points for InferEdgeForge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from inferedgeforge.build import (
    collect_metadata_files,
    create_build_plan,
    format_compare_candidates,
    format_compare_command_preview,
    format_inspect_summary,
    format_build_list,
    group_by_source_model,
    inspect_build_metadata,
    rebuild_from_manifest,
    run_lab_profile,
    run_build,
    to_lab_profile_command,
    to_lab_profile_input,
    write_run_summary,
)
from inferedgeforge.metadata import read_build_metadata
from inferedgeforge.manifest_validation import format_manifest_validation, validate_manifest
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


def _cmd_show_lab_profile_input(args: argparse.Namespace) -> int:
    metadata = read_build_metadata(args.metadata_path)
    payload = to_lab_profile_input(metadata)
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_show_lab_profile_command(args: argparse.Namespace) -> int:
    metadata = read_build_metadata(args.metadata_path)
    command = to_lab_profile_command(metadata)
    print(command)
    return 0


def _cmd_inspect_build(args: argparse.Namespace) -> int:
    metadata = read_build_metadata(args.metadata_path)
    payload = inspect_build_metadata(metadata)
    if args.summary:
        print(format_inspect_summary(metadata, payload["run_summary"]))
        return 0
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_run_benchmark(args: argparse.Namespace) -> int:
    metadata = read_build_metadata(args.metadata_path)
    result = run_lab_profile(metadata)
    write_run_summary(metadata, result)
    print(json.dumps(result, indent=2))
    return int(result["returncode"])


def _cmd_list_builds(args: argparse.Namespace) -> int:
    metadata_items = []
    for metadata_path in collect_metadata_files(args.dir):
        metadata = read_build_metadata(metadata_path)
        if args.model is not None and metadata.source_model.path != args.model:
            continue
        metadata_items.append(metadata)

    print(format_build_list(group_by_source_model(metadata_items)))
    return 0


def _cmd_show_compare_candidates(args: argparse.Namespace) -> int:
    metadata_items = []
    for metadata_path in collect_metadata_files(args.dir):
        metadata = read_build_metadata(metadata_path)
        if args.model is not None and metadata.source_model.path != args.model:
            continue
        metadata_items.append(metadata)

    print(format_compare_candidates(group_by_source_model(metadata_items)))
    return 0


def _cmd_show_compare_command(args: argparse.Namespace) -> int:
    metadata_items = []
    for metadata_path in collect_metadata_files(args.dir):
        metadata = read_build_metadata(metadata_path)
        if metadata.source_model.path != args.model:
            continue
        metadata_items.append(metadata)

    print(
        format_compare_command_preview(
            group_by_source_model(metadata_items),
            model=args.model,
            left=args.left,
            right=args.right,
        )
    )
    return 0


def _cmd_rebuild_from_manifest(args: argparse.Namespace) -> int:
    metadata = rebuild_from_manifest(
        manifest_path=args.manifest_path,
        output_dir=args.output,
    )
    artifact_path = Path(metadata.artifacts[0].path)
    output_root = artifact_path.parent.parent

    print("Rebuild completed")
    print(f"Manifest : {args.manifest_path}")
    print(f"Preset   : {metadata.build.preset_name}")
    print(f"Model    : {metadata.source_model.path}")
    print(f"Output   : {output_root}")
    print(f"Metadata : {_metadata_path(metadata)}")
    return 0


def _cmd_validate_manifest(args: argparse.Namespace) -> int:
    result = validate_manifest(manifest=args.manifest, build_dir=args.build_dir)
    print(format_manifest_validation(result))
    return 0 if result.valid else 1


def _metadata_path(metadata: BuildMetadata) -> Path:
    artifact_path = Path(metadata.artifacts[0].path)
    return artifact_path.parent / "metadata.json"


def _cmd_build(args: argparse.Namespace) -> int:
    if args.dry_run:
        payload = create_build_plan(
            model_path=args.model,
            preset_id=args.preset,
            presets_root=args.presets_root,
        )
        print(json.dumps(payload, indent=2))
        return 0

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

    show_lab_parser = subparsers.add_parser(
        "show-lab-profile-input",
        help="Show the InferEdgeLab profile input mapping derived from build metadata.",
    )
    show_lab_parser.add_argument("metadata_path", help="Path to a build metadata.json file.")
    show_lab_parser.set_defaults(func=_cmd_show_lab_profile_input)

    show_lab_command_parser = subparsers.add_parser(
        "show-lab-profile-command",
        help="Show a runnable InferEdgeLab profile command derived from build metadata.",
    )
    show_lab_command_parser.add_argument("metadata_path", help="Path to a build metadata.json file.")
    show_lab_command_parser.set_defaults(func=_cmd_show_lab_profile_command)

    inspect_parser = subparsers.add_parser(
        "inspect-build",
        help="Show a consolidated build inspection summary derived from metadata.",
    )
    inspect_parser.add_argument("metadata_path", help="Path to a build metadata.json file.")
    inspect_parser.add_argument(
        "--summary",
        action="store_true",
        help="Show a concise human-readable build inspection summary.",
    )
    inspect_parser.set_defaults(func=_cmd_inspect_build)

    run_parser = subparsers.add_parser(
        "run-benchmark",
        help="Execute InferEdgeLab profile using build metadata.",
    )
    run_parser.add_argument("metadata_path", help="Path to metadata.json")
    run_parser.set_defaults(func=_cmd_run_benchmark)

    list_builds_parser = subparsers.add_parser(
        "list-builds",
        help="List build metadata grouped by source model.",
    )
    list_builds_parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing build metadata.json files.",
    )
    list_builds_parser.add_argument(
        "--model",
        help="Optional source model path to filter by.",
    )
    list_builds_parser.set_defaults(func=_cmd_list_builds)

    compare_parser = subparsers.add_parser(
        "show-compare-candidates",
        help="Show compare-ready build variants grouped by source model.",
    )
    compare_parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing build metadata.json files.",
    )
    compare_parser.add_argument(
        "--model",
        help="Optional source model path to filter by.",
    )
    compare_parser.set_defaults(func=_cmd_show_compare_candidates)

    compare_command_parser = subparsers.add_parser(
        "show-compare-command",
        help="Preview an InferEdgeLab compare command for ready build variants.",
    )
    compare_command_parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing build metadata.json files.",
    )
    compare_command_parser.add_argument(
        "--model",
        required=True,
        help="Source model path to select a build group.",
    )
    compare_command_parser.add_argument(
        "--left",
        help="Optional left preset name.",
    )
    compare_command_parser.add_argument(
        "--right",
        help="Optional right preset name.",
    )
    compare_command_parser.set_defaults(func=_cmd_show_compare_command)

    rebuild_parser = subparsers.add_parser(
        "rebuild-from-manifest",
        help="Rebuild an artifact using a stored manifest.json file.",
    )
    rebuild_parser.add_argument("manifest_path", help="Path to manifest.json")
    rebuild_parser.add_argument(
        "--output",
        help="Optional output root for rebuilt artifacts.",
    )
    rebuild_parser.set_defaults(func=_cmd_rebuild_from_manifest)

    validate_parser = subparsers.add_parser(
        "validate-manifest",
        help="Validate manifest.json handoff sanity without rebuilding.",
    )
    validate_source = validate_parser.add_mutually_exclusive_group(required=True)
    validate_source.add_argument("--manifest", help="Path to manifest.json.")
    validate_source.add_argument("--build-dir", help="Build directory containing manifest.json.")
    validate_parser.set_defaults(func=_cmd_validate_manifest)

    build_parser = subparsers.add_parser("build", help="Run the MVP build flow.")
    build_parser.add_argument("--model", required=True, help="Path to the source ONNX model.")
    build_parser.add_argument("--preset", required=True, help="Preset identifier in 'backend/name' format.")
    build_parser.add_argument("--output", required=True, help="Output directory for build artifacts.")
    build_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the build plan as JSON without executing the backend builder.",
    )
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
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
