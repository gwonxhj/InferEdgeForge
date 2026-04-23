# Jetson TensorRT Validation Record

## Purpose

This document records a real Jetson validation pass for the InferEdgeForge TensorRT build path.

The goal is to leave a reproducible, reviewable record that the current workflow works on Jetson as an actual deployment path rather than only as a local placeholder flow. The specific focus of this validation is that the same build recipe can be run, inspected, benchmarked, tracked, and, when needed, rebuilt from `manifest.json`.

This document is intentionally conservative. Only confirmed execution results should be marked as complete. Any value not yet observed on Jetson should stay as `TBD` or `실기 결과 입력 예정`.

## Validation Environment

- Jetson board model: `TBD`
- JetPack version: `TBD`
- TensorRT / `trtexec`: `TBD`
- Python version: `TBD`
- Execution date: `실기 결과 입력 예정`
- Additional environment notes: `TBD`

## Inputs Used

- Source ONNX path: `models/test.onnx`
- Preset name: `tensorrt/jetson_fp16`
- Output root: `builds`
- Manifest used: `Yes`
- Rebuild destination override: `TBD`

## Execution Order

Run the validation steps in this order on the Jetson system.

### 1. Build

```bash
python -m inferedgeforge.cli build \
  --model models/test.onnx \
  --preset tensorrt/jetson_fp16 \
  --output builds
```

### 2. Inspect Build Summary

```bash
python -m inferedgeforge.cli inspect-build --summary \
  builds/test__jetson__tensorrt/metadata.json
```

### 3. Run Benchmark

```bash
python -m inferedgeforge.cli run-benchmark \
  builds/test__jetson__tensorrt/metadata.json
```

### 4. List Builds

```bash
python -m inferedgeforge.cli list-builds --dir builds
```

### 5. Show Compare Candidates

```bash
python -m inferedgeforge.cli show-compare-candidates \
  --dir builds \
  --model models/test.onnx
```

### 6. Show Compare Command

This step only becomes fully useful after two benchmark-ready variants for the same source model exist and both have persisted `structured_result_path` values.

```bash
python -m inferedgeforge.cli show-compare-command \
  --dir builds \
  --model models/test.onnx \
  --left tensorrt/jetson_fp16 \
  --right rknn/rk3588_fp16
```

If a second Jetson-oriented TensorRT preset is being compared instead, replace `--right` with the actual ready preset name used during validation.

### 7. Rebuild From Manifest

```bash
python -m inferedgeforge.cli rebuild-from-manifest \
  builds/test__jetson__tensorrt/manifest.json
```

Optional rebuild destination override:

```bash
python -m inferedgeforge.cli rebuild-from-manifest \
  builds/test__jetson__tensorrt/manifest.json \
  --output rebuilt
```

## Validation Checklist

- [ ] `.engine` generated successfully
- [ ] `metadata.json` generated successfully
- [ ] `manifest.json` generated successfully
- [ ] `inspect-build --summary` printed correctly
- [ ] `run_summary.json` generated successfully
- [ ] `structured_result_path` persisted
- [ ] compare candidate discovery worked
- [ ] compare command preview worked
- [ ] `rebuild-from-manifest` succeeded

## Result Record

Record observed Jetson results here after execution.

| Item | Expected Value | Observed Value |
| --- | --- | --- |
| Engine artifact path | `builds/test__jetson__tensorrt/model.engine` | `TBD` |
| Metadata path | `builds/test__jetson__tensorrt/metadata.json` | `TBD` |
| Manifest path | `builds/test__jetson__tensorrt/manifest.json` | `TBD` |
| Run summary path | `builds/test__jetson__tensorrt/run_summary.json` | `TBD` |
| Structured result path | `실기 결과 입력 예정` | `TBD` |
| Benchmark success | `Yes/No` | `TBD` |
| Compare candidate discovery | `Yes/No` | `TBD` |
| Compare command preview | `Yes/No` | `TBD` |
| Rebuild from manifest | `Yes/No` | `TBD` |

## Issues / Follow-Up Notes

- `trtexec` must be present in `PATH` on the Jetson system.
- TensorRT engine generation is environment-dependent and tied to the local Jetson/TensorRT installation.
- `show-compare-command` depends on persisted `structured_result_path` values from downstream InferEdgeLab profiling.
- `run-benchmark` depends on a working InferEdgeLab environment on the target system.
- A single successful TensorRT build does not automatically guarantee compare readiness; at least two benchmark-ready variants are needed for a meaningful compare handoff.
- If rebuild behavior differs across environments, record whether the source ONNX path, preset availability, or TensorRT toolchain layout changed.

## Conclusion

- Jetson validation status: `실기 결과 입력 예정`
- Verified so far in this document: command sequence, expected output paths, checklist structure, and rebuild record format
- Remaining work: fill in actual hardware/software versions, command outputs, generated paths, benchmark status, compare readiness, and rebuild outcome from a real Jetson run
- Recommended next step: execute the full workflow on the target Jetson board, then replace each `TBD` value with observed results and note any environment-specific issues
