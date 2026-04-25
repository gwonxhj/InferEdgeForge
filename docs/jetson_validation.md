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
- Execution date: `2026-04-24`
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
  --output rebuilds
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
- [x] `rebuild-from-manifest` succeeded

## Result Record

Record observed Jetson results here after execution.

## Haeundae Custom YOLOv8n Accuracy-Aware Validation

This section records the official accuracy-aware Jetson TensorRT validation for the Haeundae custom YOLOv8n model.

This validation is intentionally separate from the earlier COCO YOLOv8n latency-only record above. It uses a different ONNX model, a task-matched Haeundae validation dataset, and matching TensorRT FP16/FP32 detection accuracy payloads. The earlier RKNN payload attachment was only a workflow smoke test and is not mixed into this result.

InferEdgeForge records the build, artifact lineage, metadata, run summary, and Lab handoff context. InferEdgeLab is treated as the downstream validation tool that produced the structured latency results, attached detection accuracy evidence, and compared the enriched results.

### Haeundae Validation Inputs

| Item | Observed Value |
| --- | --- |
| Source ONNX path | `models/onnx/yolov8n_haeundae.onnx` |
| Source SHA-256 | `c99a5563c0c00859d39e2d2c4afc5de7646b96a320ba7e9493d8cc367427d9a5` |
| Dataset image directory | `/home/risenano01/DeepStream-Yolo/datasets/images/val` |
| Dataset label directory | `/home/risenano01/DeepStream-Yolo/datasets/labels/val` |
| Samples | `1657` |
| Task | `detection` |
| Classes | `1` |
| Input | `RGB, 640x640` |
| Confidence threshold | `0.2` |
| NMS threshold | `0.45` |

### Haeundae FP16 Build, Benchmark, and Accuracy Evidence

| Item | Observed Value |
| --- | --- |
| Preset | `tensorrt/jetson_fp16` |
| Build directory | `builds/yolov8n_haeundae__jetson__tensorrt__jetson_fp16` |
| Engine artifact path | `builds/yolov8n_haeundae__jetson__tensorrt__jetson_fp16/model.engine` |
| Artifact SHA-256 | `1d2b0791207591007743a3fbbaa988aaf1f9b9fbab92f2037ca0900a13b4a14c` |
| Metadata path | `builds/yolov8n_haeundae__jetson__tensorrt__jetson_fp16/metadata.json` |
| Manifest path | `builds/yolov8n_haeundae__jetson__tensorrt__jetson_fp16/manifest.json` |
| Run summary path | `builds/yolov8n_haeundae__jetson__tensorrt__jetson_fp16/run_summary.json` |
| Raw report path | `reports/yolov8n_haeundae__tensorrt_gpu__b1__h640w640__r100__20260425-151440.json` |
| Structured latency result | `results/yolov8n_haeundae.onnx__tensorrt__gpu__fp16__b1__h640w640__20260425-151440.json` |
| Detection accuracy JSON | `accuracy/yolov8n_haeundae_tensorrt_fp16_detection_accuracy.json` |
| Enriched result | `results/yolov8n_haeundae.onnx__tensorrt__gpu__fp16__b1__h640w640__20260425-153017-751881.json` |

### Haeundae FP32 Build, Benchmark, and Accuracy Evidence

| Item | Observed Value |
| --- | --- |
| Preset | `tensorrt/jetson_fp32` |
| Build directory | `builds/yolov8n_haeundae__jetson__tensorrt__jetson_fp32` |
| Engine artifact path | `builds/yolov8n_haeundae__jetson__tensorrt__jetson_fp32/model.engine` |
| Artifact SHA-256 | `0ad5eccfc8b1d5602eb9fc5258a20b9139776fc20c75a2987f25d02de4217cb2` |
| Metadata path | `builds/yolov8n_haeundae__jetson__tensorrt__jetson_fp32/metadata.json` |
| Manifest path | `builds/yolov8n_haeundae__jetson__tensorrt__jetson_fp32/manifest.json` |
| Run summary path | `builds/yolov8n_haeundae__jetson__tensorrt__jetson_fp32/run_summary.json` |
| Raw report path | `reports/yolov8n_haeundae__tensorrt_gpu__b1__h640w640__r100__20260425-152321.json` |
| Structured latency result | `results/yolov8n_haeundae.onnx__tensorrt__gpu__fp32__b1__h640w640__20260425-152321.json` |
| Detection accuracy JSON | `accuracy/yolov8n_haeundae_tensorrt_fp32_detection_accuracy.json` |
| Enriched result | `results/yolov8n_haeundae.onnx__tensorrt__gpu__fp32__b1__h640w640__20260425-153017-753457.json` |

### Haeundae Latency Comparison

| Metric | FP16 | FP32 | Delta |
| --- | ---: | ---: | ---: |
| mean_ms | `8.8819` | `10.2869` | `+1.4049 ms (+15.82%)` |
| p99_ms | `13.7437` | `18.1921` | `+4.4484 ms (+32.37%)` |

### Haeundae Detection Accuracy Comparison

| Metric | FP16 | FP32 | Delta |
| --- | ---: | ---: | ---: |
| map50 | `0.8037` | `0.8041` | `+0.04pp` |
| map50_95 | `0.5519` | `0.5520` | `+0.01pp` |
| f1_score | `0.8195` | `0.8197` | `+0.02pp` |
| precision | `0.7983` | `0.7983` | `approximately 0.00pp` |
| recall | `0.8419` | `0.8423` | `+0.04pp` |

### Haeundae Enriched Compare Result

| Item | Observed Value |
| --- | --- |
| Comparison mode | `cross_precision` |
| Precision pair | `fp16_vs_fp32` |
| Overall judgement | `tradeoff_slower` |
| Mean judgement | `regression` |
| P99 judgement | `regression` |
| Accuracy judge | `neutral` |
| Trade-off risk | `not_beneficial` |

### Haeundae Validation Conclusion

- In the Haeundae validation context, FP32 provides almost no accuracy benefit over FP16.
- FP32 regresses both mean latency and P99 latency.
- Therefore, TensorRT FP16 is the selected precision for this validation condition.
- This conclusion is scoped to the Jetson Orin environment, the Haeundae validation dataset, and this custom YOLOv8n model.
- This is the official accuracy-aware validation record for the Haeundae custom model only. It should not be generalized to the earlier COCO YOLOv8n latency-only result.

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

### Rebuild From Manifest Validation

| Item | Observed Value |
| --- | --- |
| Rebuild command | `python -m inferedgeforge.cli rebuild-from-manifest builds/yolov8n__jetson__tensorrt__jetson_fp16/manifest.json --output rebuilds` |
| Rebuild output root | `rebuilds` |
| Rebuild metadata path | `rebuilds/yolov8n__jetson__tensorrt__jetson_fp16/metadata.json` |
| Rebuild engine path | `rebuilds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine` |
| Rebuild run summary path | `rebuilds/yolov8n__jetson__tensorrt__jetson_fp16/run_summary.json` |
| Rebuild raw report path | `reports/yolov8n__tensorrt_gpu__b1__h640w640__r100__20260424-134946.json` |
| Rebuild structured result path | `results/yolov8n.onnx__tensorrt__gpu__fp16__b1__h640w640__20260424-134946.json` |
| Rebuild benchmark status | `success` |

### Rebuild SHA Comparison

| Item | Observed Value |
| --- | --- |
| Original FP16 engine SHA-256 | `29484d824f5be2dfd3e1e801e927298f15f8e77af785711ac6fd429a7445ea22` |
| Rebuilt FP16 engine SHA-256 | `f667c6dc64e2030b6c8a2904c918d1402ce0ef2fee739aeff94553bee2161f3e` |
| SHA comparison | `different` |

### Rebuild Benchmark Result

| Item | Observed Value |
| --- | --- |
| `returncode` | `0` |
| `preset_name` | `tensorrt/jetson_fp16` |
| `backend` | `tensorrt` |
| `target` | `jetson` |
| `source_model.path` | `models/onnx/yolov8n.onnx` |
| `source_model.sha256` | `4b31ebf8213f2971b8136f7ccca475e27f40559a14bc27e0d8a531a933273eb7` |
| `primary_artifact.path` | `rebuilds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine` |
| `primary_artifact.sha256` | `f667c6dc64e2030b6c8a2904c918d1402ce0ef2fee739aeff94553bee2161f3e` |
| `summary_file_path` | `rebuilds/yolov8n__jetson__tensorrt__jetson_fp16/run_summary.json` |
| `status` | `completed` |

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

#### Accuracy-aware workflow smoke test

- An `enrich-pair` smoke test was run to confirm that accuracy payload attachment and accuracy-aware compare wiring work end to end.
- Command used:

`python -m inferedgelab.cli enrich-pair --base-result ~/InferEdgeForge/results/yolov8n.onnx__tensorrt__gpu__fp16__b1__h640w640__20260424-035442.json --base-accuracy-json ~/InferEdgeLab/benchmarks/rknn_accuracy_payloads/yolov8n_fp16_detection_accuracy.json --new-result ~/InferEdgeForge/results/yolov8n.onnx__tensorrt__gpu__fp32__b1__h640w640__20260424-035938.json --new-accuracy-json ~/InferEdgeLab/benchmarks/rknn_accuracy_payloads/yolov8n_int8_detection_accuracy.json --out-dir ~/InferEdgeLab/results`

- Enriched results produced:
  - `/home/risenano01/InferEdgeLab/results/yolov8n.onnx__tensorrt__gpu__fp16__b1__h640w640__20260424-131747-658979.json`
  - `/home/risenano01/InferEdgeLab/results/yolov8n.onnx__tensorrt__gpu__fp32__b1__h640w640__20260424-131747-660732.json`
- The resulting accuracy-aware compare surfaced:
  - `Accuracy judge: improvement`
  - `Primary metric: map50`
  - `map50 delta: +1.86pp`
  - `f1_score delta: -0.51pp`
  - `precision delta: -0.84pp`
  - `recall delta: -0.14pp`
  - `Overall judgement: tradeoff_slower`
  - `Trade-off risk: not_beneficial`
- This remains a workflow smoke test only. The attached accuracy payloads came from existing RKNN evidence examples, not from task-matched TensorRT evaluation on the Jetson FP16 and FP32 artifacts.
- Because of that provenance mismatch, this smoke test proves that the accuracy-aware attach and compare flow works, but it does not upgrade the official Jetson TensorRT validation result beyond latency-only.

### Workflow Status Snapshot

| Item | Observed Value |
| --- | --- |
| `list-builds` | `success` |
| `show-compare-candidates` | `success` |
| `show-compare-command` | `success` |
| Compare execution | `success` |
| `rebuild-from-manifest` | `success` |

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
- `rebuild-from-manifest` successfully regenerated a runnable and benchmarkable TensorRT engine in a separate output root.
- The rebuilt TensorRT engine SHA-256 differed from the original engine SHA-256. That is recorded as a known limitation of TensorRT engine serialization rather than a rebuild failure.
- This validation therefore supports functional reproducibility of the build recipe, not bitwise reproducibility of the serialized engine artifact.
- If rebuild behavior differs across environments, record whether the source ONNX path, preset availability, or TensorRT toolchain layout changed.

## Conclusion

- Jetson TensorRT FP16 and FP32 engine build validation completed successfully on a Jetson Orin environment.
- Forge traceability was also validated end-to-end through `metadata.json`, `manifest.json`, artifact/source SHA-256 recording, and persisted `run_summary.json`.
- InferEdgeLab handoff through `run-benchmark`, compare candidate discovery, compare command preview, and actual `compare` execution was completed successfully.
- Manifest-based rebuild also succeeded on Jetson, and the rebuilt FP16 TensorRT engine remained runnable and benchmarkable in a separate `rebuilds/` output root.
- In this run, FP32 was effectively neutral on mean latency versus FP16 but regressed materially on P99 latency, so the observed trade-off was `not_beneficial`.
- Accuracy remained `unknown` in the compare output, so this should be read as a latency-only trade-off result rather than a full accuracy-versus-performance conclusion.
- Original and rebuilt TensorRT engine hashes differed, so this should be interpreted as functional reproducibility of the same recipe rather than guaranteed bitwise reproducibility of the serialized engine artifact.
- Remaining work: record the exact JetPack version and attach task-matched TensorRT accuracy evidence through InferEdgeLab if an accuracy-aware compare is needed.
