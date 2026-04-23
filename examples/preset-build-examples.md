# Preset Build Examples

This document shows preset-specific build usage patterns in InferEdgeForge. The goal is to help users quickly understand preset selection, experiment tracking, and the handoff flow. Forge creates artifacts and metadata, while downstream validation and comparison remain the responsibility of InferEdgeLab.

## TensorRT Example

This minimal example assumes a Jetson-oriented target and uses the `tensorrt/jetson_fp16` preset.

The flow is preview, real build, inspection, then explicit handoff command exposure.

```bash
python -m inferedgeforge.cli build --model models/test.onnx --preset tensorrt/jetson_fp16 --output builds --dry-run

python -m inferedgeforge.cli build --model models/test.onnx --preset tensorrt/jetson_fp16 --output builds

python -m inferedgeforge.cli inspect-build --summary builds/test__jetson__tensorrt/metadata.json

python -m inferedgeforge.cli show-lab-profile-command builds/test__jetson__tensorrt/metadata.json
```

The dry-run command is preview-only. The real build creates the artifact and `metadata.json`. `inspect-build --summary` gives a compact view of build, preset, source model, artifact, run status, and compare readiness. `show-lab-profile-command` exposes the next-step InferEdgeLab profile command without executing it.

## RKNN Example

This minimal example assumes an RK3588-oriented target and uses the `rknn/rk3588_fp16` preset.

The flow is preview, real build, inspection, then metadata-based downstream execution.

```bash
python -m inferedgeforge.cli build --model models/test.onnx --preset rknn/rk3588_fp16 --output builds --dry-run

python -m inferedgeforge.cli build --model models/test.onnx --preset rknn/rk3588_fp16 --output builds

python -m inferedgeforge.cli inspect-build --summary builds/test__rk3588__rknn/metadata.json

python -m inferedgeforge.cli run-benchmark builds/test__rk3588__rknn/metadata.json

python -m inferedgeforge.cli inspect-build --summary builds/test__rk3588__rknn/metadata.json
```

`run-benchmark` executes the downstream Lab profile flow from metadata. Forge prints an execution summary and persists `run_summary.json` next to `metadata.json`. Running `inspect-build --summary` again can show the persisted execution trace alongside the build, source, artifact, and compare-readiness context.

## Multi-Preset Compare Handoff

Once multiple preset builds exist for the same source model, Forge can show them as one experiment group and prepare the compare handoff.

```bash
python -m inferedgeforge.cli run-benchmark builds/test__jetson__tensorrt/metadata.json

python -m inferedgeforge.cli run-benchmark builds/test__rk3588__rknn/metadata.json

python -m inferedgeforge.cli list-builds --dir builds

python -m inferedgeforge.cli show-compare-candidates --dir builds --model models/test.onnx

python -m inferedgeforge.cli show-compare-command --dir builds --model models/test.onnx --left tensorrt/jetson_fp16 --right rknn/rk3588_fp16
```

`list-builds` groups build outputs by source model. `show-compare-candidates` shows which preset variants are ready because a `run_summary.json` exists and which ones are still pending. `show-compare-command` previews an InferEdgeLab `compare` command only when both selected variants have persisted `structured_result_path` values. The final compare execution and interpretation still happen in InferEdgeLab.

## Notes

- Backend toolchain availability is environment-dependent; these examples show the current Forge workflow shape.
- `build --dry-run` previews expected output paths and handoff structure without creating files.
- Real builds use predictable output paths such as `builds/<model>__<target>__<backend>/`.
- Benchmark-ready compare candidates depend on persisted `run_summary.json` files.
- Compare command preview depends on `structured_result_path` values reported by downstream profiling.
- Forge prepares artifacts, metadata, handoff mappings, execution traces, and compare command previews; Lab remains responsible for profiling, comparison, and validation analysis.
