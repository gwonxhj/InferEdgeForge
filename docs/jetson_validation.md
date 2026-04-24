# Jetson TensorRT Validation Record

## Purpose

This document records a real Jetson validation pass for the InferEdgeForge TensorRT build path.

The goal is to leave a reproducible, reviewable record that the current workflow works on Jetson as an actual deployment path rather than only as a local placeholder flow. The specific focus of this validation is that the same build recipe can be run, inspected, benchmarked, tracked, and, when needed, rebuilt from `manifest.json`.

This document is intentionally conservative. Only confirmed execution results should be marked as complete. Any value not yet observed on Jetson should stay as `TBD` or `실기 결과 입력 예정`.

## Validation Environment

- OS: `Ubuntu 22.04.5 LTS`
- Kernel: `Linux 5.15.148-tegra`
- Architecture: `aarch64`
- Jetson board model: `Jetson Orin series (exact SKU TBD)`
- JetPack version: `TBD`
- GPU device name from `trtexec`: `Orin`
- TensorRT version: `10.3.0`
- TensorRT package version: `10.3.0.30-1+cuda12.5`
- Python version: `3.10.12`
- Python environment: `yolo_env`
- `trtexec` path: `/usr/src/tensorrt/bin/trtexec`
- Execution date: `TBD`
- Additional environment notes: `TBD`

## Inputs Used

- Source ONNX path: `models/onnx/yolov8n.onnx`
- Source SHA-256: `4b31ebf8213f2971b8136f7ccca475e27f40559a14bc27e0d8a531a933273eb7`
- Presets validated:
  - `tensorrt/jetson_fp16`
  - `tensorrt/jetson_fp32`
- Output root: `builds`
- Manifest used: `Yes`
- Rebuild destination override: `TBD`

## Execution Order

Run the validation steps in this order on the Jetson system.

### 1. Build

```bash
python -m inferedgeforge.cli build \
  --model models/onnx/yolov8n.onnx \
  --preset tensorrt/jetson_fp16 \
  --output builds
```

### 2. Build FP32

```bash
python -m inferedgeforge.cli build \
  --model models/onnx/yolov8n.onnx \
  --preset tensorrt/jetson_fp32 \
  --output builds
```

### 3. Inspect Build Summary

```bash
python -m inferedgeforge.cli inspect-build --summary \
  builds/yolov8n__jetson__tensorrt__jetson_fp16/metadata.json
```

```bash
python -m inferedgeforge.cli inspect-build --summary \
  builds/yolov8n__jetson__tensorrt__jetson_fp32/metadata.json
```

### 4. Run Benchmark

```bash
python -m inferedgeforge.cli run-benchmark \
  builds/yolov8n__jetson__tensorrt__jetson_fp16/metadata.json
```

```bash
python -m inferedgeforge.cli run-benchmark \
  builds/yolov8n__jetson__tensorrt__jetson_fp32/metadata.json
```

### 5. List Builds

```bash
python -m inferedgeforge.cli list-builds --dir builds
```

### 6. Show Compare Candidates

```bash
python -m inferedgeforge.cli show-compare-candidates \
  --dir builds \
  --model models/onnx/yolov8n.onnx
```

### 7. Show Compare Command

```bash
python -m inferedgeforge.cli show-compare-command \
  --dir builds \
  --model models/onnx/yolov8n.onnx \
  --left tensorrt/jetson_fp16 \
  --right tensorrt/jetson_fp32
```

### 8. Run Compare in InferEdgeLab

```bash
python -m inferedgelab.cli compare \
  results/yolov8n.onnx__tensorrt__gpu__fp16__b1__h640w640__20260424-035442.json \
  results/yolov8n.onnx__tensorrt__gpu__fp32__b1__h640w640__20260424-035938.json
```

### 9. Rebuild From Manifest

```bash
python -m inferedgeforge.cli rebuild-from-manifest \
  builds/yolov8n__jetson__tensorrt__jetson_fp16/manifest.json
```

Optional rebuild destination override:

```bash
python -m inferedgeforge.cli rebuild-from-manifest \
  builds/yolov8n__jetson__tensorrt__jetson_fp16/manifest.json \
  --output rebuilt
```

## Validation Checklist

- [x] FP16 `.engine` generated successfully
- [x] FP32 `.engine` generated successfully
- [x] `metadata.json` generated successfully
- [x] `manifest.json` generated successfully
- [x] `inspect-build --summary` printed correctly
- [x] `run_summary.json` generated successfully
- [x] `structured_result_path` persisted
- [x] compare candidate discovery worked
- [x] compare command preview worked
- [ ] `rebuild-from-manifest` succeeded

## Result Record

Record observed Jetson results here after execution.

### FP16 Build and Benchmark

| Item | Observed Value |
| --- | --- |
| Preset | `tensorrt/jetson_fp16` |
| Build directory | `builds/yolov8n__jetson__tensorrt__jetson_fp16` |
| Engine artifact path | `builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine` |
| Engine size | `8.9M` |
| Artifact SHA-256 | `302bb66ffbe58bef49ef17343ea5b42f9a783673650dd351f68c795018a1046c` |
| Metadata path | `builds/yolov8n__jetson__tensorrt__jetson_fp16/metadata.json` |
| Manifest path | `builds/yolov8n__jetson__tensorrt__jetson_fp16/manifest.json` |
| Run summary path | `builds/yolov8n__jetson__tensorrt__jetson_fp16/run_summary.json` |
| Raw report path | `reports/yolov8n__tensorrt_gpu__b1__h640w640__r100__20260424-035442.json` |
| Structured result path | `results/yolov8n.onnx__tensorrt__gpu__fp16__b1__h640w640__20260424-035442.json` |

### FP32 Build and Benchmark

| Item | Observed Value |
| --- | --- |
| Preset | `tensorrt/jetson_fp32` |
| Build directory | `builds/yolov8n__jetson__tensorrt__jetson_fp32` |
| Engine artifact path | `builds/yolov8n__jetson__tensorrt__jetson_fp32/model.engine` |
| Engine size | `14M` |
| Artifact SHA-256 | `99c84230aede1fe3a1f7ebec7b981fa1671b3f842aa99637a8fa386c5b017442` |
| Metadata path | `builds/yolov8n__jetson__tensorrt__jetson_fp32/metadata.json` |
| Manifest path | `builds/yolov8n__jetson__tensorrt__jetson_fp32/manifest.json` |
| Run summary path | `builds/yolov8n__jetson__tensorrt__jetson_fp32/run_summary.json` |
| Raw report path | `reports/yolov8n__tensorrt_gpu__b1__h640w640__r100__20260424-035938.json` |
| Structured result path | `results/yolov8n.onnx__tensorrt__gpu__fp32__b1__h640w640__20260424-035938.json` |

### Compare Result

| Item | Observed Value |
| --- | --- |
| Compare command | `python -m inferedgelab.cli compare results/yolov8n.onnx__tensorrt__gpu__fp16__b1__h640w640__20260424-035442.json results/yolov8n.onnx__tensorrt__gpu__fp32__b1__h640w640__20260424-035938.json` |
| Comparison mode | `cross_precision` |
| Precision pair | `fp16_vs_fp32` |
| Overall judgement | `tradeoff_slower` |
| Shape match | `True` |
| System match | `True` |
| Mean judgement | `neutral` |
| P99 judgement | `regression` |
| Accuracy judge | `unknown` |
| Trade-off risk | `not_beneficial` |
| FP16 mean_ms | `13.3527` |
| FP32 mean_ms | `13.4897` |
| Mean delta | `+0.1370 ms` |
| Mean delta percent | `+1.03%` |
| FP16 p99_ms | `16.8170` |
| FP32 p99_ms | `20.9434` |
| P99 delta | `+4.1264 ms` |
| P99 delta percent | `+24.54%` |

### Accuracy Evidence Status

- This Jetson FP16 vs FP32 validation remains a latency-only compare.
- The recorded compare output kept `accuracy judge` as `unknown`.
- Existing RKNN accuracy payloads in InferEdgeLab are schema examples and prior evidence examples only. They were not attached to these Jetson TensorRT results.
- To make this TensorRT comparison accuracy-aware, matching TensorRT evaluation results need to be generated separately for the same task, dataset, and metric definition, then attached with `enrich-result` or `enrich-pair`.
- A typical accuracy payload uses a `task` field plus a `metrics` object, for example:

```json
{
  "task": "detection",
  "metrics": {
    "map50": 0.7791,
    "f1_score": 0.8180,
    "precision": 0.7950,
    "recall": 0.8424
  }
}
```

- After that evidence is attached downstream in InferEdgeLab, compare can be rerun with latency and accuracy evidence together. That did not happen in this Jetson validation record.

### Workflow Status Snapshot

| Item | Observed Value |
| --- | --- |
| `list-builds` | `success` |
| `show-compare-candidates` | `success` |
| `show-compare-command` | `success` |
| Compare execution | `success` |
| `rebuild-from-manifest` | `TBD` |

## Issues / Follow-Up Notes

- `trtexec` must be present in `PATH` on the Jetson system.
- TensorRT engine generation is environment-dependent and tied to the local Jetson/TensorRT installation.
- `show-compare-command` depends on persisted `structured_result_path` values from downstream InferEdgeLab profiling.
- `run-benchmark` depends on a working InferEdgeLab environment on the target system.
- During `run-benchmark`, stderr included the ONNX Runtime GPU discovery warning `GPU device discovery failed: Failed to open /sys/class/drm/card1/device/vendor`.
- That warning was non-blocking in this validation pass because `returncode` was `0`, the raw report was saved, the structured result was saved, `run_summary.json` was saved, and the compare workflow completed successfully.
- FP32 was not beneficial in this run despite similar mean latency because the P99 latency regression was materially larger.
- Accuracy evidence was not attached for these TensorRT results, so the compare outcome should not be read as an accuracy-aware deployment decision.
- Existing RKNN accuracy payload examples are useful for schema reference only here; they were not reused for the TensorRT Jetson results.
- Accuracy data was not collected in this pass, so the compare interpretation is latency-only and should not be treated as a full deployment-quality judgement.
- If rebuild behavior differs across environments, record whether the source ONNX path, preset availability, or TensorRT toolchain layout changed.

## Conclusion

- Jetson TensorRT FP16 and FP32 engine build validation completed successfully on a Jetson Orin environment.
- Forge traceability was also validated end-to-end through `metadata.json`, `manifest.json`, artifact/source SHA-256 recording, and persisted `run_summary.json`.
- InferEdgeLab handoff through `run-benchmark`, compare candidate discovery, compare command preview, and actual `compare` execution was completed successfully.
- In this run, FP32 was effectively neutral on mean latency versus FP16 but regressed materially on P99 latency, so the observed trade-off was `not_beneficial`.
- Accuracy remained `unknown` in the compare output, so this should be read as a latency-only trade-off result rather than a full accuracy-versus-performance conclusion.
- Remaining work: record the exact JetPack version, execute `rebuild-from-manifest` on Jetson if rebuild reproducibility needs to be documented in the same level of detail, and attach task-matched TensorRT accuracy evidence through InferEdgeLab if an accuracy-aware compare is needed.
