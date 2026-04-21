# InferEdgeForge CLI Design

## Purpose

This document defines the initial CLI surface for InferEdgeForge before the full implementation is in place. The command set is intentionally narrow so the project can establish a stable build contract early.

The first implementation priority is `build`. Supporting commands for preset discovery should follow once the preset system is in place.

## Design Goals

- Keep the CLI centered on deployment artifact production
- Make preset usage explicit rather than hidden behind implicit defaults
- Produce machine-readable metadata for every build
- Keep backend-specific complexity behind a stable command surface
- Support clean validation handoff into InferEdgeLab

## Command Overview

### `build`

**Purpose**

Run the optimization and artifact production workflow for a source ONNX model using a selected preset and target/backend combination.

This is the first implementation priority because it defines the primary contract of InferEdgeForge.

**Example command**

```bash
inferedgeforge build \
  --model ./models/resnet50.onnx \
  --preset rknn/rk3588_fp16 \
  --target rk3588 \
  --backend rknn \
  --output ./artifacts
```

**Expected output direction**

- Generate deployment artifacts under the requested output directory
- Generate a structured JSON metadata file alongside the artifacts
- Print a concise build summary to stdout
- Return a non-zero exit code on validation or build failure

### `list-presets`

**Purpose**

List the presets currently available to the CLI so users can discover supported build profiles before running a build.

**Example command**

```bash
inferedgeforge list-presets
```

**Expected output direction**

- Print preset identifiers in a human-readable list
- Optionally include backend and target hints for each preset
- Avoid verbose implementation details in the default output

### `show-preset`

**Purpose**

Display the details of a single preset so users can inspect its intended target/backend configuration before building.

**Example command**

```bash
inferedgeforge show-preset rknn/rk3588_fp16
```

**Expected output direction**

- Print a readable summary of the preset definition
- Show the preset name, backend, target, and key build parameters
- Fail clearly if the preset identifier does not exist

## Proposed CLI Behavior

### Build Inputs

The `build` command should accept, at minimum:

- `--model` for the input ONNX model
- `--preset` for preset selection
- `--target` for explicit deployment target selection
- `--backend` for backend selection
- `--output` for artifact output location

The exact argument set can evolve, but the CLI should keep the relationship between preset, target, and backend visible rather than implicit.

### Build Outputs

The output of a successful build should include:

- deployment artifacts produced by the selected builder
- a structured JSON metadata file stored with the build output
- enough console output to confirm what was built and where it was written

The metadata JSON should be treated as a required build output because it is needed for traceability and validation handoff.

### Metadata Expectations

Artifact metadata should be stored in a structured JSON file. At minimum, it should record:

- source ONNX model reference
- selected preset
- selected target
- selected backend
- generated artifact filenames
- output paths
- build identifiers needed for reproducibility

This metadata is expected to become the handoff contract consumed by InferEdgeLab.

## Error Handling Direction

The initial CLI should prefer explicit failures over ambiguous fallback behavior.

Examples:

- missing ONNX input should fail before build execution
- unknown preset should fail before backend invocation
- incompatible target/backend combinations should fail clearly
- backend build errors should surface with enough context for diagnosis

## Priority Order

Implementation should proceed in this order:

1. `build`
2. `list-presets`
3. `show-preset`

This order matches the core purpose of the project: producing deployment artifacts first, then improving discoverability and inspection around the preset system.
