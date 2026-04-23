# InferEdgeForge

**A CLI-first build orchestration pipeline that turns ONNX models into traceable edge deployment artifacts, structured metadata, experiment-level build views, and InferEdgeLab-ready handoff records.**

## Why This Project Exists

Deploying an ONNX model to an edge device usually requires more than model conversion. Teams also need target-specific build settings, reproducible output naming, traceable metadata, and a reliable way to hand generated artifacts into downstream validation workflows.

InferEdgeForge exists to make that build step explicit, repeatable, and inspectable. Its job is not to claim that every optimized artifact is automatically the right deployment choice. Its job is to produce deployment artifacts with enough build context, fingerprints, preset intent, and handoff metadata that the result can be reviewed and validated later.

As multiple preset variants accumulate for the same source model, Forge also helps organize those builds into a comparison-ready workflow. It can list builds by source model, identify which variants already have benchmark traces, and preview an InferEdgeLab compare command when two structured results are available.

## What InferEdgeForge Solves

InferEdgeForge is intended to provide a consistent build pipeline for edge inference packaging:

- Start from a source ONNX model.
- Apply a named preset that captures build intent.
- Route the build through a target-specific backend.
- Generate deployment artifacts with predictable naming.
- Emit structured metadata alongside each artifact.
- Prepare outputs so they can be handed off to validation workflows.
- Execute the downstream InferEdgeLab profile step from stored handoff metadata when requested.
- Persist execution summaries so downstream benchmark runs can be traced back to the build context that produced them.
- List multiple builds for the same source model as an experiment view across presets.
- Identify compare-ready variants and preview a Lab compare command from persisted result paths.

This keeps optimization logic, build configuration, output records, and downstream execution context in one place rather than scattering them across ad hoc scripts. A Forge build leaves more than an artifact: it leaves `metadata.json` with source and artifact fingerprints, a preset snapshot, Lab handoff mapping, and, after `run-benchmark`, a persisted `run_summary.json`.

That means a project can move from single-artifact build output to a small, traceable experiment loop: build several preset variants, benchmark the ones that are ready, then hand the resulting structured outputs to InferEdgeLab for comparison.

## Relationship to InferEdgeLab

InferEdgeForge and InferEdgeLab are paired projects with different responsibilities:

- **InferEdgeForge builds deployment artifacts.**
- **InferEdgeLab validates whether those artifacts are better deployment decisions.**

In practical terms, InferEdgeForge is the build-generation side of the pipeline. It takes an ONNX model and produces backend-oriented artifacts plus metadata describing how they were built.

InferEdgeLab is the analysis side. It consumes those artifacts and metadata to evaluate runtime behavior, compare alternatives, and study deployment trade-offs such as performance, footprint, compatibility, and reproducibility.

Forge can prepare the compare handoff by showing ready variants and previewing an `inferedgelab compare` command when persisted structured result paths are available. Lab still owns the actual compare execution and result interpretation. This separation is deliberate: build generation, execution trace capture, and deployment evaluation are related, but they should not be collapsed into one tool.

## Example Output

This section shows representative outputs produced by InferEdgeForge after a build and optional downstream execution.

### 1. Build Metadata (`metadata.json`)

This is a simplified example of the structured metadata emitted after build. The real file includes additional fields such as schema version, build IDs, timestamps, and Lab-compatible runtime mapping.

```json
{
  "build": {
    "build_id": "test-tensorrt-jetson_fp16-20260423T120000Z",
    "preset_name": "tensorrt/jetson_fp16",
    "backend": "tensorrt",
    "target": "jetson"
  },
  "source_model": {
    "path": "models/test.onnx",
    "format": "onnx",
    "sha256": "..."
  },
  "artifacts": [
    {
      "path": "builds/test__jetson__tensorrt/model.engine",
      "format": "engine",
      "role": "deployment_model",
      "sha256": "..."
    }
  ],
  "preset_snapshot": {
    "name": "tensorrt/jetson_fp16",
    "backend": "tensorrt",
    "target": "jetson",
    "build_options": {
      "precision": "fp16",
      "workspace_mb": 2048
    }
  },
  "lab_compat": {
    "runtime": {
      "engine": "tensorrt",
      "device": "jetson",
      "precision": "fp16"
    }
  }
}
```

The metadata records both build identity and build intent. If a preset changes later, the snapshot in `metadata.json` still preserves the effective preset values used for that build.

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

### 4. Persisted Run Summary (`run_summary.json`)

After `run-benchmark`, Forge persists a summary next to the build metadata.

```json
{
  "command": "python -m inferedgelab.cli profile models/test.onnx --engine tensorrt ...",
  "returncode": 0,
  "structured_result_path": "results/test.json",
  "summary_file_path": "builds/test__jetson__tensorrt/run_summary.json",
  "status": "completed",
  "build_id": "test-tensorrt-jetson_fp16-20260423T120000Z",
  "preset_name": "tensorrt/jetson_fp16",
  "backend": "tensorrt",
  "target": "jetson",
  "source_model": {
    "path": "models/test.onnx",
    "format": "onnx",
    "sha256": "..."
  },
  "primary_artifact": {
    "path": "builds/test__jetson__tensorrt/model.engine",
    "format": "engine",
    "role": "deployment_model",
    "sha256": "..."
  }
}
```

This makes the downstream execution trace self-contained enough to understand which build, preset, source model, and artifact were used.

### 5. Compare Command Preview

After two variants for the same source model have benchmark summaries with structured result paths, Forge can preview the Lab compare command.

```bash
python -m inferedgeforge.cli show-compare-command \
  --dir builds \
  --model models/test.onnx \
  --left tensorrt/jetson_fp16 \
  --right rknn/rk3588_fp16
```

Example output:

```text
Compare Command Preview
-----------------------
Model       : models/test.onnx
Left Preset : tensorrt/jetson_fp16
Right Preset: rknn/rk3588_fp16

python -m inferedgelab.cli compare results/tensorrt.json results/rknn.json
```

Forge does not run or interpret the compare step here. It only exposes the command preview from existing `run_summary.json` data.

## Architecture Snapshot

InferEdgeForge is being structured as a build system with a small number of stable concepts:

- **Input model**: a source ONNX model provided by the user.
- **Preset**: a reusable build profile that expresses optimization intent.
- **Builder**: a backend-specific implementation such as RKNN or TensorRT.
- **Artifact output**: the generated deployment files for a chosen edge inference target.
- **Metadata output**: structured JSON describing the build inputs, preset snapshot, source fingerprint, target/backend, produced files, artifact fingerprints, and handoff mapping.
- **Validation handoff**: a clean boundary where InferEdgeLab can consume artifacts without depending on internal build logic.
- **Execution summary**: optional persisted `run_summary.json` from `run-benchmark`, linked back to the build and artifact context.
- **Experiment view**: source-model grouping across multiple preset builds.
- **Compare preview**: read-only discovery of compare-ready variants and the Lab compare command when structured result paths are present.

## Current CLI Workflow

The current workflow is intentionally narrow, build-centered, and traceable, then extends into compare-ready handoff:

1. Select an ONNX model.
2. Choose a preset.
3. Choose a target/backend.
4. Run `build --dry-run` to preview the expected build plan.
5. Run `build` to produce deployment artifacts and structured metadata.
6. Run `inspect-build --summary` for a human-readable status view, or `inspect-build` for full JSON.
7. Use `show-lab-profile-input` or `show-lab-profile-command` to expose the handoff mapping explicitly.
8. Run the downstream InferEdgeLab profile step with `run-benchmark` when the environment is ready.
9. Use `list-builds` to view multiple builds grouped by source model.
10. Use `show-compare-candidates` to distinguish ready and pending variants.
11. Use `show-compare-command` to preview the Lab compare command when two ready variants have structured result paths.

Representative build command:

```bash
inferedgeforge build \
  --model ./models/resnet50.onnx \
  --preset rknn/rk3588_fp16 \
  --output ./artifacts
```

Use `build --dry-run` to preview the build plan without executing the backend builder.

```bash
python -m inferedgeforge.cli build \
  --model models/test.onnx \
  --preset tensorrt/jetson_fp16 \
  --output builds \
  --dry-run
```

The output is JSON-only and includes predictable artifact, metadata, and run summary path previews plus an InferEdgeLab handoff preview. It does not create files or execute the backend builder.

Use read-only inspection commands to review generated metadata and handoff information:

```bash
python -m inferedgeforge.cli inspect-build --summary \
  builds/test__jetson__tensorrt/metadata.json

python -m inferedgeforge.cli inspect-build \
  builds/test__jetson__tensorrt/metadata.json

python -m inferedgeforge.cli show-lab-profile-input \
  builds/test__jetson__tensorrt/metadata.json

python -m inferedgeforge.cli show-lab-profile-command \
  builds/test__jetson__tensorrt/metadata.json
```

`inspect-build --summary` gives a concise human-readable view of build identity, preset, source model fingerprint, artifact fingerprint, run status, and the next step. The default `inspect-build` command preserves the full JSON inspection view. If a persisted `run_summary.json` exists in the build directory, inspection includes that downstream execution context.

Use `run-benchmark` to execute the downstream InferEdgeLab profile flow from `metadata.json`.

```bash
python -m inferedgeforge.cli run-benchmark \
  builds/test__jetson__tensorrt/metadata.json
```

This command uses the stored handoff metadata to execute the downstream Lab profile step, then prints and saves a Forge-side execution summary as `run_summary.json`. The summary includes command output paths plus build, preset, source model, and primary artifact context.

Use experiment and compare handoff commands after creating and benchmarking multiple preset variants for the same source model:

```bash
python -m inferedgeforge.cli list-builds --dir builds

python -m inferedgeforge.cli show-compare-candidates --dir builds \
  --model models/test.onnx

python -m inferedgeforge.cli show-compare-command --dir builds \
  --model models/test.onnx \
  --left tensorrt/jetson_fp16 \
  --right rknn/rk3588_fp16
```

`list-builds` gives an experiment-level view grouped by source model. `show-compare-candidates` separates benchmark-ready variants from pending ones. `show-compare-command` previews an InferEdgeLab `compare` command only when both selected variants have persisted `structured_result_path` values.

Preset discovery is available through `list-presets` and `show-preset` before builds are executed.

## Installation and Execution

New here? See [docs/quickstart.md](docs/quickstart.md) for a practical end-to-end walkthrough covering editable install, simple test model creation, build execution, output inspection, and InferEdgeLab handoff.

See [examples/README.md](examples/README.md) for minimal local build flow, metadata inspection, InferEdgeLab handoff, and preset-oriented compare workflow examples.

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
- Minimal TensorRT artifact path for local workflow validation
- Output layout suitable for validation handoff into InferEdgeLab
- Read-only inspection commands for metadata and Lab handoff information
- Source model and artifact SHA-256 fingerprints in metadata
- Preset snapshots persisted with build metadata
- Metadata-based downstream execution through `run-benchmark`
- Persisted execution summaries visible through `inspect-build`
- Experiment-level build listing with `list-builds`
- Compare candidate discovery with `show-compare-candidates`
- Compare command preview with `show-compare-command`

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
4. Add TensorRT-oriented build support.
5. Formalize validation handoff into InferEdgeLab.
6. Improve developer experience, packaging, and test coverage.

See [Roadmap.md](Roadmap.md) for the phase-by-phase plan.

## Status

InferEdgeForge has an initial MVP workflow in place. It can discover presets, preview build plans, generate build artifacts and metadata, record source model and artifact fingerprints, persist preset snapshots, expose InferEdgeLab handoff mappings, execute the downstream profile command through `run-benchmark`, persist enriched run summaries, and include those summaries in `inspect-build` output.

The current inspection and experiment flow supports full JSON review, a human-readable `inspect-build --summary` view, source-model build grouping with `list-builds`, compare-ready discovery with `show-compare-candidates`, and compare command preview with `show-compare-command`.

The repository should still be read as a focused foundation rather than a claim that every backend pipeline is production-complete. Backend-specific toolchains, real TensorRT engine generation, broader device coverage, and validation analysis remain staged work.
