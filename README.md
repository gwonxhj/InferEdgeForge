# InferEdgeForge

**A reproducible inference experiment system for edge deployment.**

InferEdgeForge is a CLI-first system for turning ONNX models into deployment artifacts, traceable build records, benchmark handoff inputs, and compare-ready experiment outputs. It is not just a model conversion script. It is an attempt to make edge inference packaging reproducible, inspectable, and reviewable as an engineering workflow.

## Problem Statement

Moving from an ONNX model to an actual edge deployment artifact is usually messy.

- Build intent often lives in shell history, ad hoc notes, or one-off scripts.
- Artifacts can be regenerated, but it is unclear whether they came from the same recipe.
- Benchmark outputs are hard to trust when they are disconnected from the build context that produced them.
- Teams may keep the artifact, but lose the preset, source fingerprint, output lineage, and handoff state needed to review it later.

In practice, that means ONNX to edge deployment is often treated like a conversion step instead of an experiment system. Once multiple variants exist, traceability breaks down quickly.

## Why Existing Workflow Breaks

A typical flow looks simple:

1. Export an ONNX model.
2. Run a backend-specific build command.
3. Save the artifact somewhere.
4. Benchmark it later, often in a separate tool.

The problem is not that these steps are impossible. The problem is that they are weakly connected.

- Reproducibility is fragile because the exact preset/build intent may not be preserved.
- Traceability is fragile because artifact lineage, source hashes, and output context can be lost.
- Benchmark interpretation is fragile because results may not be tied back to a specific build state.
- Comparison readiness is fragile because multiple variants may exist, but there is no structured way to know which ones are ready for compare.

## What This Project Actually Builds

InferEdgeForge builds **a reproducible inference experiment system for edge deployment**.

That system has a narrow but important responsibility:

- generate deployment artifacts from ONNX using named presets
- preserve build intent and artifact lineage as structured records
- prepare downstream benchmark and compare handoff into InferEdgeLab
- keep build, benchmark trace, and experiment state connected enough to review later

The output is not only an artifact. It is an artifact plus the context required to reason about that artifact as part of a deployment experiment.

## Key Capabilities

InferEdgeForge currently provides these implemented capabilities:

- `preset`-based build abstraction across backends and targets
- structured `metadata.json` and `manifest.json` outputs
- source model and artifact SHA-256 fingerprints
- preset snapshots persisted with build records
- InferEdgeLab handoff generation through profile input and command views
- `run-benchmark` execution from stored handoff metadata
- persisted `run_summary.json` for downstream traceability
- experiment-level build listing across preset variants
- compare-ready candidate discovery
- compare command preview from persisted structured result paths
- downstream accuracy evidence attachment support via InferEdgeLab enrich flows
- `rebuild-from-manifest` support, with Jetson rebuild validation recorded as functionally reproducible

## Real Validation (Jetson)

This project is not documented only as a design idea. It has a recorded Jetson validation pass in [docs/jetson_validation.md](docs/jetson_validation.md).

Verified on Jetson Orin-class hardware:

- TensorRT FP16 build succeeded for YOLOv8n
- TensorRT FP32 build succeeded for YOLOv8n
- both variants were benchmarked successfully
- compare handoff was executed and recorded
- the compare result was documented as **latency-only**
- an accuracy-aware workflow smoke test was also recorded
- `rebuild-from-manifest` regenerated a runnable and benchmarkable TensorRT engine in a separate output root

What is important here is the level of validation:

- the artifact build path was exercised
- the benchmark handoff path was exercised
- the compare-ready workflow was exercised
- the rebuild path was exercised

What is equally important is what is **not** being claimed:

- this is not a statement that all TensorRT environments are production-ready
- this is not a statement that the Jetson compare is already accuracy-aware
- this is not a statement of bitwise TensorRT artifact reproducibility

The current recorded Jetson compare is intentionally scoped as a latency-oriented FP16 vs FP32 result. Accuracy-aware compare wiring has been smoke-tested with existing InferEdgeLab payload examples, but task-matched TensorRT accuracy evidence has not yet been attached to the official Jetson FP16/FP32 result.

## How The System Works

InferEdgeForge and InferEdgeLab are paired, but they do different jobs.

**InferEdgeForge**

- builds deployment artifacts
- records build identity and preset intent
- preserves source and artifact fingerprints
- prepares benchmark and compare handoff state

**InferEdgeLab**

- profiles runtime behavior
- compares structured results
- attaches downstream accuracy evidence when available
- interprets trade-offs during validation

This separation matters. Build generation and deployment evaluation are related, but they should not be collapsed into one opaque tool.

## Representative Workflow

The current workflow is intentionally build-centered and traceable.

```bash
python -m inferedgeforge.cli build \
  --model models/test.onnx \
  --preset tensorrt/jetson_fp16 \
  --output builds \
  --dry-run

python -m inferedgeforge.cli build \
  --model models/test.onnx \
  --preset tensorrt/jetson_fp16 \
  --output builds

python -m inferedgeforge.cli inspect-build --summary \
  builds/test__jetson__tensorrt__jetson_fp16/metadata.json

python -m inferedgeforge.cli run-benchmark \
  builds/test__jetson__tensorrt__jetson_fp16/metadata.json

python -m inferedgeforge.cli list-builds --dir builds

python -m inferedgeforge.cli show-compare-candidates \
  --dir builds \
  --model models/test.onnx

python -m inferedgeforge.cli show-compare-command \
  --dir builds \
  --model models/test.onnx \
  --left tensorrt/jetson_fp16 \
  --right tensorrt/jetson_fp32
```

Representative outputs produced by the system:

- `metadata.json`
- `manifest.json`
- deployment artifact such as `model.engine` or `model.rknn`
- `run_summary.json`
- compare-ready discovery views across builds

## Evidence Produced Per Build

Each build can leave a reviewable record instead of only a binary artifact.

- `metadata.json`: build identity, source model context, preset snapshot, handoff mapping
- `manifest.json`: reproducibility-oriented snapshot of the build recipe and artifact context
- artifact SHA-256: fingerprint of the produced deployment artifact
- source SHA-256: fingerprint of the ONNX input
- `run_summary.json`: persisted downstream execution trace after `run-benchmark`

That is the core difference between this project and a plain conversion wrapper. The system preserves enough state to support later inspection, rebuild, and comparison.

## What This Means

This project is meant to demonstrate more than CLI implementation.

It demonstrates the design of a system that supports:

- **experiment traceability**: multiple preset variants can be grouped and reviewed as one experiment surface
- **reproducibility**: build intent, manifests, source fingerprints, and rebuild flows are preserved as explicit records
- **deployment decision support**: compare-ready handoff and benchmark traces help downstream analysis happen with context intact

That is why this project should be read as an experiment workflow system rather than just a model build utility.

## Limitations

The current system is intentionally honest about its boundaries.

- TensorRT engine hashes are not guaranteed to be bitwise stable across rebuilds
- Jetson rebuild validation currently supports functional reproducibility, not bitwise identity
- Jetson FP16 vs FP32 compare is still documented as latency-only
- accuracy evidence depends on external evaluation results or downstream InferEdgeLab enrich flows
- some environment details in the Jetson validation record remain `TBD`
- backend toolchains remain environment-dependent
- broader device coverage is still open work

## Documentation Map

- [docs/quickstart.md](docs/quickstart.md): practical end-to-end quickstart
- [docs/handoff.md](docs/handoff.md): Forge to Lab handoff contract
- [docs/jetson_validation.md](docs/jetson_validation.md): recorded Jetson validation evidence
- [examples/README.md](examples/README.md): examples index
- [Roadmap.md](Roadmap.md): implementation status and next steps

## Status

InferEdgeForge already has:

- preset-based build orchestration
- structured metadata and manifest output
- SHA-based traceability for source models and artifacts
- Jetson TensorRT real engine generation via `trtexec`
- benchmark handoff and persisted execution summaries
- compare-ready experiment views
- documented Jetson FP16/FP32 validation evidence
- documented manifest-based rebuild validation

It should still be read as a focused and validated foundation, not as a claim that every backend path or deployment workflow is complete.
