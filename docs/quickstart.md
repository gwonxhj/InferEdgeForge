# InferEdgeForge Quickstart

This quickstart shows how to try the current InferEdgeForge workflow in a few minutes: install in editable mode, create a simple ONNX input, preview a build plan, run a build, inspect output metadata, prepare an InferEdgeLab handoff, and optionally execute the downstream benchmark command from metadata.

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

## 7. Handoff to InferEdgeLab

InferEdgeForge does not score deployment quality or decide whether a build is better. Its role is to create artifacts, metadata, fingerprints, preset snapshots, and handoff-ready runtime mappings. InferEdgeLab is the downstream tool that performs runtime profiling, comparison, and deployment trade-off analysis.

```bash
python -m inferedgelab.cli profile models/test.onnx --engine tensorrt --engine-path builds/test__jetson__tensorrt/model.engine --device-name jetson --precision fp16
```

This command is generated from Forge metadata and represents the downstream validation step. Forge can show it explicitly with `show-lab-profile-command` or execute it through `run-benchmark`.

## Next Steps

- Replace the dummy ONNX file with a real model.
- Run builds in an environment compatible with the selected backend toolchain.
- Use Forge metadata and artifacts as the bridge into InferEdgeLab validation.
- Re-run `inspect-build --summary` after `run-benchmark` to review the persisted execution trace.
