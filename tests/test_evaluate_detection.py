from __future__ import annotations

import json
from pathlib import Path

from inferedgelab.cli import build_parser
from inferedgelab.core.detection_evaluator import (
    Detection,
    calculate_iou,
    load_ground_truth,
    nms,
    save_accuracy_payload,
)


def test_load_ground_truth_converts_normalized_to_absolute(tmp_path: Path) -> None:
    label_file = tmp_path / "sample.txt"
    label_file.write_text("0 0.5 0.25 0.2 0.4\n", encoding="utf-8")

    items = load_ground_truth(label_file, image_width=200, image_height=100)

    assert len(items) == 1
    assert items[0].class_id == 0
    assert items[0].box == (100.0, 25.0, 40.0, 40.0)


def test_calculate_iou_basic_case() -> None:
    iou = calculate_iou((50, 50, 40, 40), (60, 60, 40, 40))
    assert 0.38 < iou < 0.40


def test_nms_keeps_highest_confidence() -> None:
    detections = [
        Detection(class_id=0, confidence=0.9, box=(50, 50, 40, 40)),
        Detection(class_id=0, confidence=0.8, box=(50, 50, 40, 40)),
        Detection(class_id=1, confidence=0.7, box=(50, 50, 40, 40)),
    ]

    kept = nms(detections, iou_threshold=0.5)

    assert len(kept) == 2
    assert kept[0].confidence == 0.9
    assert kept[1].class_id == 1


def test_perfect_prediction_metrics_near_one() -> None:
    pred = Detection(class_id=0, confidence=0.99, box=(100, 100, 50, 50))
    gt_box = (100, 100, 50, 50)
    iou = calculate_iou(pred.box, gt_box)
    assert iou == 1.0


def test_accuracy_payload_format_preserves_task_and_metrics(tmp_path: Path) -> None:
    out = tmp_path / "accuracy.json"
    eval_result = {
        "metrics": {
            "map50": 1.0,
            "map50_95": 1.0,
            "f1_score": 1.0,
            "precision": 1.0,
            "recall": 1.0,
        },
        "dataset": {"image_dir": "images", "label_dir": "labels", "sample_count": 1},
        "evaluation_config": {"conf_threshold": 0.2},
    }

    save_accuracy_payload(out, eval_result)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["task"] == "detection"
    assert payload["metrics"]["map50"] == 1.0


def test_cli_help_contains_evaluate_detection() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "evaluate-detection" in help_text
