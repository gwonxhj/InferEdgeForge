# InferEdgeForge Quickstart

This quickstart shows how to try the current InferEdgeForge workflow in a few minutes: install in editable mode, create a simple ONNX input, preview a build plan, run a build, inspect output metadata, and prepare a handoff into InferEdgeLab.

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

This preview creates no files and is useful for validating preset selection and the expected InferEdgeLab handoff structure before a real build.

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

`show-lab-profile-command` prints a runnable InferEdgeLab profile command.

```bash
python -m inferedgeforge.cli inspect-build \
  builds/test__jetson__tensorrt/metadata.json
```

`inspect-build` prints a consolidated summary of build details and handoff-ready information. The dry-run preview is for plan validation only, while the real build produces metadata and artifacts for inspection.

## 6. Run the Downstream Benchmark

```bash
python -m inferedgeforge.cli run-benchmark \
  builds/test__jetson__tensorrt/metadata.json
```

This executes the InferEdgeLab profile command derived from metadata. It is the execution step after preview, build, and inspection, while Forge still prepares the handoff structure and Lab performs the profiling.

## 7. Handoff to InferEdgeLab

InferEdgeForge does not benchmark runtime performance or score deployment quality. Its role is to prepare structured build outputs and validation-ready handoff information. InferEdgeLab is the downstream tool that performs runtime profiling, comparison, and deployment trade-off analysis.

```bash
python -m inferedgelab.cli profile models/test.onnx --engine tensorrt --engine-path builds/test__jetson__tensorrt/model.engine --device-name jetson --precision fp16
```

This command is generated from Forge metadata and is intended as the next validation step.

## Next Steps

- Replace the dummy ONNX file with a real model
- Use a backend-compatible environment for RKNN or TensorRT builds
- Pair Forge outputs with InferEdgeLab for validation and comparison
