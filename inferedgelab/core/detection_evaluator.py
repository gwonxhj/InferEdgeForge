from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised in integration environments
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]  # absolute xywh


@dataclass(frozen=True)
class GroundTruth:
    class_id: int
    box: tuple[float, float, float, float]  # absolute xywh


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required for detection evaluation but is not installed.") from exc
    return np


def _require_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for detection evaluation but is not installed.")
    return cv2


def letterbox(image: Any, target_size: int = 640, use_rgb: bool = True):
    _cv2 = _require_cv2()
    np = _require_numpy()
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = _cv2.resize(image, (nw, nh), interpolation=_cv2.INTER_LINEAR)
    if use_rgb:
        resized = _cv2.cvtColor(resized, _cv2.COLOR_BGR2RGB)

    canvas = np.full((target_size, target_size, 3), 114, dtype=resized.dtype)
    pad_w = (target_size - nw) / 2.0
    pad_h = (target_size - nh) / 2.0
    left, top = int(round(pad_w - 0.1)), int(round(pad_h - 0.1))
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, float(left), float(top)


def scale_coords(box, scale: float, pad_w: float, pad_h: float, image_shape: tuple[int, int]):
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2 - pad_w) / scale
    y1 = (cy - bh / 2 - pad_h) / scale
    x2 = (cx + bw / 2 - pad_w) / scale
    y2 = (cy + bh / 2 - pad_h) / scale

    h, w = image_shape
    x1 = float(max(0.0, min(x1, w)))
    x2 = float(max(0.0, min(x2, w)))
    y1 = float(max(0.0, min(y1, h)))
    y2 = float(max(0.0, min(y2, h)))

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return cx, cy, bw, bh


def _xywh_to_xyxy(box):
    cx, cy, w, h = box
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def calculate_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def nms(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
    ordered = sorted(detections, key=lambda det: det.confidence, reverse=True)
    kept: list[Detection] = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        remaining: list[Detection] = []
        for det in ordered:
            if det.class_id != current.class_id or calculate_iou(current.box, det.box) < iou_threshold:
                remaining.append(det)
        ordered = remaining
    return kept


def load_ground_truth(label_path: Path, image_width: int, image_height: int) -> list[GroundTruth]:
    if not label_path.exists():
        return []
    items: list[GroundTruth] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cls, xc, yc, w, h = line.split()[:5]
        items.append(GroundTruth(int(float(cls)), (float(xc) * image_width, float(yc) * image_height, float(w) * image_width, float(h) * image_height)))
    return items


def get_image_files(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(image_dir.glob(ext))
    return sorted(files)


def _ensure_nxc(array: Any, num_classes: int):
    np = _require_numpy()
    c = 4 + num_classes
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        if array.shape[1] == c:
            return array
        if array.shape[0] == c:
            return array.T
    if array.ndim == 3:
        if array.shape[-1] == c:
            return array.reshape(-1, c)
        if array.shape[-2] == c:
            return np.swapaxes(array, -1, -2).reshape(-1, c)
    raise ValueError(f"Unsupported YOLO output shape: {array.shape}")


def postprocess_yolov8(outputs: Any, num_classes: int, conf_threshold: float, nms_threshold: float, scale: float, pad_w: float, pad_h: float, image_shape: tuple[int, int]) -> list[Detection]:
    np = _require_numpy()
    if isinstance(outputs, dict):
        tensors = [np.asarray(v) for v in outputs.values()]
    elif isinstance(outputs, (list, tuple)):
        tensors = [np.asarray(v) for v in outputs]
    else:
        tensors = [np.asarray(outputs)]

    detections: list[Detection] = []
    if len(tensors) == 1:
        preds = _ensure_nxc(tensors[0], num_classes=num_classes)
        boxes = preds[:, :4]
        scores = preds[:, 4:]
    else:
        tensors = sorted(tensors, key=lambda x: x.shape[-1] if x.ndim > 1 else -1)
        first, second = tensors[0], tensors[1]
        first2 = first.reshape(-1, first.shape[-1]) if first.ndim > 2 else first
        second2 = second.reshape(-1, second.shape[-1]) if second.ndim > 2 else second
        if first2.shape[-1] == 4:
            boxes, scores = first2, second2
        elif second2.shape[-1] == 4:
            boxes, scores = second2, first2
        else:
            merged = _ensure_nxc(first if first.size >= second.size else second, num_classes)
            boxes = merged[:, :4]
            scores = merged[:, 4:]

    n = min(boxes.shape[0], scores.shape[0])
    boxes = boxes[:n]
    scores = scores[:n]
    if scores.ndim == 1:
        scores = scores[:, None]

    for i in range(boxes.shape[0]):
        class_id = int(np.argmax(scores[i]))
        confidence = float(scores[i, class_id])
        if confidence < conf_threshold:
            continue
        raw_box = tuple(float(v) for v in boxes[i, :4])
        scaled_box = scale_coords(raw_box, scale, pad_w, pad_h, image_shape)
        detections.append(Detection(class_id=class_id, confidence=confidence, box=scaled_box))

    return nms(detections, iou_threshold=nms_threshold)


def _compute_precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def _evaluate_matches(all_predictions: list[list[Detection]], all_ground_truths: list[list[GroundTruth]], iou_threshold: float) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for preds, gts in zip(all_predictions, all_ground_truths):
        matched_gt: set[int] = set()
        for pred in sorted(preds, key=lambda x: x.confidence, reverse=True):
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gts):
                if idx in matched_gt or gt.class_id != pred.class_id:
                    continue
                iou = calculate_iou(pred.box, gt.box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                matched_gt.add(best_idx)
                tp += 1
            else:
                fp += 1
        fn += len(gts) - len(matched_gt)
    return tp, fp, fn


def _compute_map(all_predictions: list[list[Detection]], all_ground_truths: list[list[GroundTruth]], thresholds: list[float]) -> float:
    values = []
    for thr in thresholds:
        tp, fp, fn = _evaluate_matches(all_predictions, all_ground_truths, iou_threshold=thr)
        precision, recall, _ = _compute_precision_recall_f1(tp, fp, fn)
        values.append((precision * recall) ** 0.5 if precision > 0 and recall > 0 else 0.0)
    return sum(values) / len(values) if values else 0.0


def evaluate_detection_engine(model_path: str | Path, engine: str, engine_path: str | Path, image_dir: str | Path, label_dir: str | Path, num_classes: int = 1, conf_threshold: float = 0.2, nms_threshold: float = 0.45, iou_threshold: float = 0.5, use_rgb: bool = True) -> dict[str, Any]:
    np = _require_numpy()
    _cv2 = _require_cv2()
    from inferedgelab.engines import create_engine

    runtime = create_engine(engine)
    runtime.load(str(model_path), engine_path=str(engine_path))

    input_meta = runtime.inputs[0]
    input_name = input_meta.name
    input_shape = tuple(input_meta.shape)
    input_dtype = np.dtype(input_meta.dtype)

    all_predictions: list[list[Detection]] = []
    all_ground_truths: list[list[GroundTruth]] = []

    images = get_image_files(Path(image_dir))
    for image_path in images:
        image = _cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        image_lb, scale, pad_w, pad_h = letterbox(image, target_size=640, use_rgb=use_rgb)

        if np.issubdtype(input_dtype, np.floating):
            arr = image_lb.astype(np.float32) / 255.0
            if input_dtype == np.float16:
                arr = arr.astype(np.float16)
        else:
            arr = image_lb.astype(np.uint8)

        if len(input_shape) == 4 and input_shape[1] == 3:
            arr = np.transpose(arr, (2, 0, 1))[None, ...]
        else:
            arr = arr[None, ...]

        outputs = runtime.run({input_name: arr})
        preds = postprocess_yolov8(outputs, num_classes, conf_threshold, nms_threshold, scale, pad_w, pad_h, (h, w))
        gts = load_ground_truth(Path(label_dir) / f"{image_path.stem}.txt", image_width=w, image_height=h)
        all_predictions.append(preds)
        all_ground_truths.append(gts)

    tp, fp, fn = _evaluate_matches(all_predictions, all_ground_truths, iou_threshold=iou_threshold)
    precision, recall, f1 = _compute_precision_recall_f1(tp, fp, fn)
    map50 = _compute_map(all_predictions, all_ground_truths, thresholds=[0.5])
    map50_95 = _compute_map(all_predictions, all_ground_truths, thresholds=[round(v, 2) for v in np.arange(0.5, 1.0, 0.05)])

    return {
        "engine": engine,
        "device": getattr(runtime, "device", "unknown"),
        "sample_count": len(images),
        "metrics": {"map50": map50, "map50_95": map50_95, "f1_score": f1, "precision": precision, "recall": recall},
        "dataset": {"image_dir": str(image_dir), "label_dir": str(label_dir), "sample_count": len(images)},
        "evaluation_config": {"conf_threshold": conf_threshold, "nms_threshold": nms_threshold, "iou_threshold": iou_threshold, "input_size": 640, "rgb": bool(use_rgb)},
    }


def save_accuracy_payload(path: str | Path, eval_result: dict[str, Any]) -> None:
    payload = {
        "task": "detection",
        "metrics": eval_result["metrics"],
        "dataset": eval_result.get("dataset", {}),
        "evaluation_config": eval_result.get("evaluation_config", {}),
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_structured_result(out_dir: str | Path, model_path: str | Path, precision: str, eval_result: dict[str, Any]) -> Path:
    result = {
        "model": Path(model_path).name,
        "engine": eval_result["engine"],
        "device": eval_result["device"],
        "precision": precision,
        "batch": 1,
        "height": 640,
        "width": 640,
        "mean_ms": None,
        "p99_ms": None,
        "run_config": {"mode": "evaluate-detection"},
        "accuracy": {"task": "detection", "metrics": eval_result["metrics"], "dataset": eval_result.get("dataset", {})},
        "extra": {"evaluation": eval_result.get("evaluation_config", {})},
    }
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(model_path).stem}__detection_result.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return output_path
