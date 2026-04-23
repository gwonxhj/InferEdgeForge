# Preset Build Examples

This document shows preset-specific build usage patterns in InferEdgeForge. The goal is to help users quickly understand preset selection and the handoff flow. Forge creates artifacts and metadata, while downstream validation remains the responsibility of InferEdgeLab.

## TensorRT Example

This minimal example assumes a Jetson-oriented target and uses the `tensorrt/jetson_fp16` preset.

```bash
python -m inferedgeforge.cli build --model models/test.onnx --preset tensorrt/jetson_fp16 --output builds --dry-run

python -m inferedgeforge.cli build --model models/test.onnx --preset tensorrt/jetson_fp16 --output builds

python -m inferedgeforge.cli inspect-build builds/test__jetson__tensorrt/metadata.json

python -m inferedgeforge.cli show-lab-profile-command builds/test__jetson__tensorrt/metadata.json
```

The dry-run command is preview-only. The real build creates the artifact and `metadata.json`. `inspect-build` reviews build metadata and can include a persisted run summary when one exists. `show-lab-profile-command` exposes the downstream execution command for InferEdgeLab.

## RKNN Example

This minimal example assumes an RK3588-oriented target and uses the `rknn/rk3588_fp16` preset.

```bash
python -m inferedgeforge.cli build --model models/test.onnx --preset rknn/rk3588_fp16 --output builds --dry-run

python -m inferedgeforge.cli build --model models/test.onnx --preset rknn/rk3588_fp16 --output builds

python -m inferedgeforge.cli inspect-build builds/test__rk3588__rknn/metadata.json

python -m inferedgeforge.cli run-benchmark builds/test__rk3588__rknn/metadata.json
```

`run-benchmark` executes the downstream Lab profile flow from metadata. Forge prints an execution summary and persists `run_summary.json` so the downstream execution trace can be inspected later.

## Notes

- Backend toolchain availability is still environment-dependent.
- `build --dry-run` does not create files.
- Build output paths are predictable.
- Forge and Lab remain intentionally separated.
