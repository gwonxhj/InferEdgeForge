# InferEdgeForge Quickstart

This quickstart shows how to try the current InferEdgeForge workflow in a few minutes: install in editable mode, create a simple ONNX input, preview a build plan, run a build, inspect output metadata, execute a metadata-driven benchmark handoff, and review compare-ready build variants.

## Prerequisites

- Python 3.10+
- A local checkout of the InferEdgeForge repository
- Backend-specific toolchains (for example RKNN) are environment-dependent and may require extra setup
- For local workflow validation, the current TensorRT placeholder path is sufficient to test metadata generation and handoff commands

## 1. Install InferEdgeForge

```bash
python -m pip install -e .
python -m inferedgeforge.cli --help
```

This installs Forge in editable mode and confirms the CLI is available.

## 2. Create a Simple Test Model

```bash
mkdir -p models
echo "dummy onnx content" > models/test.onnx
```

This dummy file is only for validating the current build/metadata/handoff flow, not for real inference.

## 3. Run a Dry-Run Build Preview

```bash
python -m inferedgeforge.cli build \
  --model models/test.onnx \
  --preset tensorrt/jetson_fp16 \
  --output builds \
  --dry-run
```

This preview creates no files. It is useful for checking preset selection, expected output paths, and the planned InferEdgeLab handoff structure before a real build.

## 4. Run a Build

```bash
python -m inferedgeforge.cli build \
  --model models/test.onnx \
  --preset tensorrt/jetson_fp16 \
  --output builds
```

Example output:

```text
Build completed
Preset   : tensorrt/jetson_fp16
Backend  : tensorrt
Target   : jetson
Artifacts: 1
Metadata : builds/test__jetson__tensorrt/metadata.json
```

## 5. Inspect the Build Output

```bash
python -m inferedgeforge.cli show-lab-profile-input \
  builds/test__jetson__tensorrt/metadata.json
```

`show-lab-profile-input` prints the downstream JSON mapping for InferEdgeLab profile input.

```bash
python -m inferedgeforge.cli show-lab-profile-command \
  builds/test__jetson__tensorrt/metadata.json
```

`show-lab-profile-command` prints the runnable InferEdgeLab profile command derived from the same metadata.

```bash
python -m inferedgeforge.cli inspect-build --summary \
  builds/test__jetson__tensorrt/metadata.json
```

`inspect-build --summary` prints a human-readable status view covering build context, preset intent, source and artifact traceability, handoff readiness, persisted run summary visibility, and the next step. Use `inspect-build` without `--summary` when you want the full JSON inspection payload.

The dry-run preview is for plan validation only. The real build produces artifacts and `metadata.json`, including source model SHA-256, artifact SHA-256, and a preset snapshot for later review.

## 6. Run the Downstream Benchmark

```bash
python -m inferedgeforge.cli run-benchmark \
  builds/test__jetson__tensorrt/metadata.json
```

This executes the InferEdgeLab profile command derived from `metadata.json`. It is the execution step after preview, build, inspection, and handoff review.

Forge still prepares and invokes the handoff command from stored metadata; Lab performs the actual profiling. After execution, Forge saves `run_summary.json` next to `metadata.json`, and later `inspect-build` calls can include that persisted execution trace.

## 7. Build Another Preset Variant

To make the experiment view useful, create another build for the same source model with a different preset.

```bash
python -m inferedgeforge.cli build \
  --model models/test.onnx \
  --preset rknn/rk3588_fp16 \
  --output builds
```

When the selected backend environment is available, benchmark that artifact as well:

```bash
python -m inferedgeforge.cli run-benchmark \
  builds/test__rk3588__rknn/metadata.json
```

This keeps each preset variant tied to its own `metadata.json` and optional `run_summary.json`.

## 8. List Builds by Source Model

```bash
python -m inferedgeforge.cli list-builds --dir builds
```

`list-builds` scans build directories for `metadata.json` files and groups them by source model path. It is useful once you have multiple preset experiments for the same ONNX input.

## 9. Discover Compare Candidates

```bash
python -m inferedgeforge.cli show-compare-candidates \
  --dir builds \
  --model models/test.onnx
```

`show-compare-candidates` separates ready builds from pending builds. A ready build has a persisted `run_summary.json`; a pending build still needs `run-benchmark` before it can participate in the compare handoff.

## 10. Preview a Compare Command

```bash
python -m inferedgeforge.cli show-compare-command \
  --dir builds \
  --model models/test.onnx \
  --left tensorrt/jetson_fp16 \
  --right rknn/rk3588_fp16
```

`show-compare-command` previews an InferEdgeLab `compare` command when both selected ready builds have `structured_result_path` values in their persisted run summaries. Forge does not compute compare results; it only prepares the command preview so the downstream Lab workflow can run with the right structured result files.

## 11. Handoff to InferEdgeLab

InferEdgeForge does not score deployment quality or decide whether a build is better. Its role is to create artifacts, metadata, fingerprints, preset snapshots, and handoff-ready runtime mappings. InferEdgeLab is the downstream tool that performs runtime profiling, comparison, and deployment trade-off analysis.

```bash
python -m inferedgelab.cli profile models/test.onnx --engine tensorrt --engine-path builds/test__jetson__tensorrt/model.engine --device-name jetson --precision fp16
```

This command is generated from Forge metadata and represents the downstream validation step. Forge can show it explicitly with `show-lab-profile-command` or execute it through `run-benchmark`.

## Next Steps

- Replace the dummy ONNX file with a real model.
- Run builds in an environment compatible with the selected backend toolchain.
- Build more than one preset variant for the same source model when you want a compare workflow.
- Re-run `inspect-build --summary` after `run-benchmark` to review the persisted execution trace.
- Use `show-compare-candidates` and `show-compare-command` to bridge ready Forge outputs into InferEdgeLab comparison.
