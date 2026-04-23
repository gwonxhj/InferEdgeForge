# InferEdgeForge

**A CLI-first optimization pipeline that turns ONNX models into reproducible deployment artifacts for edge inference targets.**

## Why This Project Exists

Deploying an ONNX model to an edge device usually requires more than model conversion. Teams also need target-specific build settings, reproducible output naming, traceable metadata, and a reliable way to hand generated artifacts into downstream validation workflows.

InferEdgeForge exists to make that build step explicit and repeatable. Its job is not to claim that every optimized artifact is automatically the right deployment choice. Its job is to produce disciplined, inspectable deployment artifacts that can be compared and validated later.

## What InferEdgeForge Solves

InferEdgeForge is intended to provide a consistent build pipeline for edge inference packaging:

- Start from a source ONNX model.
- Apply a named preset that captures build intent.
- Route the build through a target-specific backend.
- Generate deployment artifacts with predictable naming.
- Emit structured metadata alongside each artifact.
- Prepare outputs so they can be handed off to validation workflows.

This keeps optimization logic, build configuration, and output records in one place rather than scattering them across ad hoc scripts.

## Relationship to InferEdgeLab

InferEdgeForge and InferEdgeLab are paired projects with different responsibilities:

- **InferEdgeForge builds deployment artifacts.**
- **InferEdgeLab validates whether those artifacts are better deployment decisions.**

In practical terms, InferEdgeForge is the production side of the pipeline. It takes an ONNX model and produces target-ready artifacts plus metadata describing how they were built.

InferEdgeLab is the analysis side. It consumes those artifacts and metadata to evaluate runtime behavior, compare alternatives, and study deployment trade-offs such as performance, footprint, compatibility, and reproducibility.

This separation is deliberate. Build generation and deployment evaluation are related, but they should not be collapsed into one tool.

## Example Output

This section shows representative outputs produced by InferEdgeForge after a build.

### 1. Build Metadata (`metadata.json`)

This is a simplified example of the structured metadata emitted after build.

```json
{
  "build": {
    "preset_name": "tensorrt/jetson_fp16",
    "backend": "tensorrt",
    "target": "jetson"
  },
  "artifacts": [
    {
      "path": "builds/test__jetson__tensorrt/model.engine",
      "format": "engine"
    }
  ],
  "lab_compat": {
    "runtime": {
      "engine": "tensorrt",
      "device": "jetson",
      "precision": "fp16"
    }
  }
}
```

### 2. Lab Profile Input

This is the output of `python -m inferedgeforge.cli show-lab-profile-input builds/test__jetson__tensorrt/metadata.json`.

```json
{
  "engine": "tensorrt",
  "device": "jetson",
  "precision": "fp16",
  "engine_path": "builds/test__jetson__tensorrt/model.engine",
  "runtime_artifact_path": "builds/test__jetson__tensorrt/model.engine"
}
```

### 3. Lab Profile Command

This is the output of `python -m inferedgeforge.cli show-lab-profile-command builds/test__jetson__tensorrt/metadata.json`.

```bash
python -m inferedgelab.cli profile models/test.onnx --engine tensorrt --engine-path builds/test__jetson__tensorrt/model.engine --device-name jetson --precision fp16
```

This command can be executed directly to run validation in InferEdgeLab.

## Architecture Snapshot

InferEdgeForge is being structured as a build system with a small number of stable concepts:

- **Input model**: a source ONNX model provided by the user.
- **Preset**: a reusable build profile that expresses optimization intent.
- **Builder**: a backend-specific implementation such as RKNN or TensorRT.
- **Artifact output**: the generated deployment files for a chosen edge inference target.
- **Metadata output**: structured JSON describing the build inputs, selected preset, target/backend, and produced files.
- **Validation handoff**: a clean boundary where InferEdgeLab can consume artifacts without depending on internal build logic.

## Planned CLI Workflow

The initial workflow is intentionally narrow:

1. Select an ONNX model.
2. Choose a preset.
3. Choose a target/backend.
4. Run `build`.
5. Produce deployment artifacts and structured metadata.
6. Hand the result into InferEdgeLab for validation and comparison.

Representative command shape:

```bash
inferedgeforge build \
  --model ./models/resnet50.onnx \
  --preset rknn/rk3588_fp16 \
  --output ./artifacts
```

Additional read-only commands such as `list-presets` and `show-preset` are planned to support preset discovery before builds are executed.

## Installation and Execution

Use an editable install during development:

```bash
python -m pip install -e .
```

If you do not want to install the console script yet, you can run the CLI module directly:

```bash
python -m inferedgeforge.cli --help
```

Backend-specific toolchains are still environment-dependent. For example, RKNN builds require a compatible Linux environment with the appropriate RKNN toolkit installed. The project remains installable even when those backend toolchains are not available.

## Initial Scope (MVP)

The MVP is focused on establishing a reliable build contract rather than broad backend coverage.

- A CLI entry point centered on `build`
- Preset discovery from versioned preset definitions
- Structured metadata generation for every build
- Initial builder abstraction for multiple backends
- First concrete backend path for RKNN
- Early TensorRT pipeline design and staged implementation
- Output layout suitable for validation handoff into InferEdgeLab

Non-goals for the MVP:

- Benchmark claims without a validation workflow
- Broad device coverage on day one
- Mixing build generation with runtime scoring or deployment decision logic

## Repository Structure

Current repository shape:

```text
inferedgeforge/
  cli.py
  builders/
    base.py
    rknn.py
    tensorrt.py
  schemas/
presets/
  rknn/
  tensorrt/
docs/
tests/
README.md
Roadmap.md
pyproject.toml
```

Intended responsibilities:

- `inferedgeforge/cli.py`: CLI entry point and command orchestration
- `inferedgeforge/builders/`: backend-specific build implementations
- `inferedgeforge/schemas/`: structured definitions for metadata and related contracts
- `presets/`: reusable build presets organized by backend or target family
- `docs/`: design notes and interface decisions
- `tests/`: verification of CLI behavior, preset loading, and build metadata contracts

## Roadmap Summary

The project roadmap is staged around predictable build capability:

1. Establish the project foundation and core interfaces.
2. Introduce the preset and metadata system.
3. Implement the first real RKNN build path.
4. Add TensorRT build support.
5. Formalize validation handoff into InferEdgeLab.
6. Improve developer experience, packaging, and test coverage.

See [Roadmap.md](/Users/GwonHyeokJun/InferEdgeForge/Roadmap.md) for the phase-by-phase plan.

## Status

InferEdgeForge is in its initial repository setup stage. The project direction, terminology, and architecture boundaries are being defined before implementation expands.

The repository should currently be read as an intentional foundation for a serious build system, not as a claim that the full optimization pipeline is already complete.
