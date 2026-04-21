# InferEdgeForge Architecture

## Purpose

InferEdgeForge is designed as a build and optimization system for edge inference deployments. Its responsibility is to accept a source ONNX model, apply a preset, run a target-specific builder, and produce reproducible deployment artifacts plus structured metadata.

It does not decide whether one artifact is a better deployment choice than another. That comparison step belongs to InferEdgeLab.

## Core Flow

The initial architecture centers on a small, explicit pipeline:

1. Accept an **input ONNX model**.
2. Resolve a **preset selection** that describes the intended build profile.
3. Resolve a **target/backend selection** such as RKNN or TensorRT.
4. Execute the appropriate **builder** implementation.
5. Generate deployment **artifacts** in a predictable output layout.
6. Generate structured **metadata** in JSON form.
7. Prepare a **validation handoff** package that InferEdgeLab can consume.

## Text Diagram

```text
[ONNX model]
      |
      v
[Preset selection]
      |
      v
[Target/backend resolution]
      |
      v
[Builder implementation]
      |
      +--> [Deployment artifacts]
      |
      +--> [Build metadata JSON]
      |
      v
[Validation handoff to InferEdgeLab]
```

## Architectural Elements

### Input: ONNX Model

The ONNX model is the canonical source input for the system. InferEdgeForge should treat this model as the input artifact from which target-specific deployment outputs are derived.

The build metadata should preserve the identity of this input so downstream systems can trace every generated artifact back to its original model source.

### Presets

Presets capture reusable build intent. A preset is expected to define the configuration needed to produce a deployment artifact for a specific target or backend profile without requiring users to restate every option on the command line.

Examples of responsibilities for presets:

- identify the intended backend family
- capture target-specific optimization settings
- define reproducible build defaults
- provide a stable name that can be referenced from the CLI

Presets are configuration assets. They are not builders.

### Target and Backend Selection

Target selection determines the deployment environment being prepared for. Backend selection determines which build implementation is responsible for producing artifacts.

These concepts are related but not identical:

- A **target** refers to the deployment destination or hardware family.
- A **backend** refers to the build system used to produce the artifact.

This distinction matters because multiple targets may share a backend, and backend-specific build logic should remain isolated from generic CLI orchestration.

### Builders

Builders are the execution layer of InferEdgeForge. Each builder is responsible for translating normalized build intent into backend-specific artifact generation steps.

Expected builder responsibilities:

- validate required inputs for the backend
- translate preset values into backend-specific configuration
- invoke the backend build process
- report generated files and build details back to the common metadata layer

The builder interface should stay narrow so additional backends can be added without reshaping the CLI contract.

### Artifact Generation

Artifact generation is the primary output of the system. The exact file types will vary by backend, but the system should treat them as deployment artifacts with a clear output directory and consistent naming rules.

Artifact generation should be reproducible in structure even when the underlying backend output formats differ.

### Metadata Generation

Each build should emit a structured JSON metadata file stored with the generated artifacts. This metadata is a first-class output of the build, not a side effect.

At minimum, the metadata should capture:

- source ONNX model reference
- preset name
- target name
- backend name
- build timestamp or build identifier
- generated artifact filenames and paths
- relevant build parameters required for traceability

This metadata layer is what makes the system suitable for downstream validation and comparison.

### Artifact Naming

Artifact naming should be deterministic and readable. The naming convention should make it possible to identify what was built without opening the files.

A practical naming scheme should encode:

- model identifier
- preset identifier
- target or backend identifier
- version or build stamp where needed

The exact format can evolve, but naming should be treated as part of the public build contract because downstream tools may depend on it.

### Validation Handoff

Validation handoff is the boundary between InferEdgeForge and InferEdgeLab. InferEdgeForge should output artifacts and metadata in a shape that can be consumed directly by InferEdgeLab without requiring manual cleanup or interpretation.

That handoff should remain simple:

- InferEdgeForge produces
- InferEdgeLab evaluates

This separation prevents benchmarking, comparison logic, and deployment decision analysis from leaking into the artifact production layer.

## Design Implications

- The `build` command is the primary architectural entry point.
- Presets and metadata are shared infrastructure across all builders.
- Builders should be replaceable modules behind a stable CLI surface.
- Output layout and metadata quality are part of the system design, not cosmetic details.
- Validation compatibility should be considered during build output design, not added later as an afterthought.
