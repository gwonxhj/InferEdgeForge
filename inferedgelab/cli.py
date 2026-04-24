from __future__ import annotations

import argparse
import sys
from typing import Sequence

from inferedgelab.commands.evaluate_detection import evaluate_detection_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inferedgelab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate-detection",
        help="Accuracy evaluation for YOLOv8 detection datasets",
    )
    evaluate_parser.add_argument("model_path", help="Path to ONNX model")
    evaluate_parser.add_argument("--engine", default="tensorrt")
    evaluate_parser.add_argument("--engine-path", required=True)
    evaluate_parser.add_argument("--image-dir", required=True)
    evaluate_parser.add_argument("--label-dir", required=True)
    evaluate_parser.add_argument("--num-classes", type=int, default=1)
    evaluate_parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    evaluate_parser.add_argument("--conf-threshold", type=float, default=0.2)
    evaluate_parser.add_argument("--nms-threshold", type=float, default=0.45)
    evaluate_parser.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate_parser.add_argument("--rgb", dest="rgb", action="store_true", default=True)
    evaluate_parser.add_argument("--bgr", dest="rgb", action="store_false")
    evaluate_parser.add_argument("--out-json")
    evaluate_parser.add_argument("--out-dir", default="results")
    evaluate_parser.add_argument(
        "--save-structured-result",
        dest="save_structured_result",
        action="store_true",
        default=True,
    )
    evaluate_parser.add_argument(
        "--no-save-structured-result",
        dest="save_structured_result",
        action="store_false",
    )
    evaluate_parser.set_defaults(func=evaluate_detection_cmd)

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
