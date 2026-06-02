# InferEdgeForge

Build provenance and artifact handoff layer  
(ONNX model -> TensorRT/RKNN artifacts · metadata · manifest · Runtime/Lab handoff)

Language: English | [한국어](README.ko.md)

**GitHub description:** Build provenance and handoff layer for converting ONNX models into edge deployment artifacts.

## Summary

- Build/provenance layer for the InferEdge validation pipeline
- Converts ONNX models into TensorRT/RKNN-oriented edge artifacts
- Records source model hash, artifact hash, preset, target, precision, and shape metadata
- Produces Runtime/Lab handoff records for downstream validation
- Supplies evidence for deployment review; InferEdgeLab owns the final decision

## Role Boundary At A Glance

| Area | Forge owns | Forge does not own |
|---|---|---|
| Build/provenance | Generates deployment artifacts, `metadata.json`, `manifest.json`, source/artifact hashes, preset snapshots, and handoff records | Execute inference, benchmark latency, compare candidates, or make deployment decisions |
| Runtime/Lab handoff | Provides Runtime/Lab-compatible artifact identity, shape, backend, target, precision, and worker/runtime summary context | Mutate Runtime `result.json`, Lab compare output, or Lab `deployment_decision` contracts |
| Manifest validation | Runs build-free sanity checks for required handoff fields and reproducibility metadata | Promote artifacts to production, run workers, manage queues, or decide release readiness |
| Agent manifest extension | Creates a separate `agent_manifest.json` contract without replacing existing `metadata.json` / `manifest.json` | Become an Orchestrator scheduler, runtime control plane, or AIGuard diagnosis owner |
| Portfolio scope | Keeps artifact builds traceable so downstream validation can explain evidence | Become production SaaS, auth/billing/upload service, cloud dashboard, or public artifact registry |

## What Makes InferEdgeForge Different?

InferEdgeForge is not just a model conversion script.

It is a reproducible artifact provenance layer that:

- preserves build intent as structured metadata
- keeps model artifacts tied to their source fingerprints
- makes benchmark and compare handoff traceable
- helps reviewers understand whether an edge artifact is rebuildable and valid

## InferEdge Pipeline Role

InferEdgeForge is the build/provenance layer of the larger InferEdge validation pipeline:

```text
ONNX model
-> InferEdgeForge build
-> metadata / manifest / worker runtime summary
-> InferEdgeRuntime validation / result export
-> InferEdgeLab compare / API / job workflow / deployment_decision
-> optional InferEdgeAIGuard provenance diagnosis
-> deploy / review / blocked decision

Experiment hygiene / comparability layer:
InferEdgeEnv -> v0.1.5 v1-complete local-first run evidence registry / comparability checker
```

In that pipeline, Forge is responsible for converting an ONNX model into edge deployment artifacts such as TensorRT engines or RKNN artifacts, while preserving the provenance needed by Runtime, Lab, and AIGuard.

Implemented today:

- `metadata.json` and `manifest.json` record build identity, source model hash, artifact hash, preset snapshot, backend, target, precision, and shape context
- `worker/runtime summary` projects build provenance into fields that Lab worker requests and Runtime invocation boundaries can consume
- compare and runtime handoff metadata keeps artifact lineage connected to downstream validation

Planned later:

- future Lab worker-triggered build execution
- worker/queue handoff hardening after metadata/manifest contracts stay stable
- stricter artifact promotion sanity checks around manifest validation

Reliable Edge Agent Runtime extension:

- Forge defines a standalone [`agent_manifest.json` contract](docs/agent_manifest_contract.md) ([한국어: agent_manifest.json 계약 quick guide](docs/agent_manifest_contract.ko.md)) for future multi-agent workload handoff.
- This contract is intentionally separate from existing `manifest.json` / `metadata.json` so the Core 4 validation pipeline remains backward-compatible.
- It records agent identity, type, priority, latency budget, fallback policy, runtime artifact mapping, telemetry contract version, and Lab compatibility metadata.

Forge does not decide whether a model should be deployed. InferEdgeLab owns the final `deployment_decision`; Forge supplies the reproducible build evidence that makes that decision reviewable.

Portfolio boundary: InferEdgeLab is the validation / decision layer. InferEdgeEnv is the v0.1.5 v1-complete experiment hygiene / comparability layer; it records whether benchmark evidence can be trusted and compared without owning Forge build provenance or Lab deployment decisions.

## Problem Statement

Most edge inference workflows can generate artifacts, but cannot explain or reproduce them reliably.

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
- prepare downstream Runtime/Lab handoff context for validation
- keep build, benchmark trace, and experiment state connected enough to review later

The output is not only an artifact. It is an artifact plus the context required to reason about that artifact as part of a deployment experiment.

## Key Capabilities

InferEdgeForge currently provides these implemented capabilities:

- `preset`-based build abstraction across backends and targets
- structured `metadata.json` and `manifest.json` outputs
- source model and artifact SHA-256 fingerprints
- preset snapshots persisted with build records
- InferEdgeLab handoff generation through profile input and command views
- `run-benchmark` handoff helper for invoking downstream validation commands from stored metadata
- persisted `run_summary.json` for downstream traceability
- experiment-level build listing across preset variants
- compare-ready candidate discovery
- compare command preview from persisted structured result paths
- manifest sanity validation before Runtime/Lab handoff
- standalone `agent_manifest.json` contract creation and validation for the Reliable Edge Agent Runtime extension
- downstream accuracy evidence attachment support via InferEdgeLab enrich flows
- `rebuild-from-manifest` support, with Jetson rebuild validation recorded as functionally reproducible

## Real Validation (Jetson)

This project is not documented only as a design idea. It has a recorded Jetson validation pass in [docs/jetson_validation.md](docs/jetson_validation.md).

Verified on Jetson Orin-class hardware for the baseline COCO YOLOv8n TensorRT flow:

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

Downstream Runtime/Lab validation has also recorded Jetson Evidence Track results for the Forge-generated TensorRT FP16 artifact:

- TensorRT FP16 25W: mean `10.066401 ms`, p95 `15.476641 ms`, p99 `15.548438 ms`, FPS `99.340373`
- TensorRT FP16 15W: mean `10.799106 ms`, p95 `15.438690 ms`, p99 `15.529218 ms`, FPS `92.600262`

Forge remains the build/provenance layer. Runtime owns these execution measurements, and Lab owns the comparison/deployment decision interpretation.

The repository also records an official accuracy-aware TensorRT validation for a Haeundae custom YOLOv8n model:

- source model: `models/onnx/yolov8n_haeundae.onnx`
- source SHA-256: `c99a5563c0c00859d39e2d2c4afc5de7646b96a320ba7e9493d8cc367427d9a5`
- validation dataset: 1,657 detection samples, 1 class, RGB 640x640
- TensorRT FP16 and FP32 engines were built, fingerprinted, benchmarked, and handed off for downstream InferEdgeLab validation
- matching detection accuracy payloads were attached to the FP16 and FP32 latency results
- enriched compare output judged FP32 as `tradeoff_slower`, with latency regressions and neutral accuracy

In that Haeundae validation context, FP16 is the selected TensorRT precision. FP32 increased mean latency from `8.8819 ms` to `10.2869 ms` and P99 latency from `13.7437 ms` to `18.1921 ms`, while accuracy deltas were effectively neutral (`map50 +0.04pp`, `map50_95 +0.01pp`, `f1_score +0.02pp`).

What is equally important is what is **not** being claimed:

- this is not a statement that all TensorRT environments are production-ready
- this is not a statement that the baseline COCO YOLOv8n Jetson compare is accuracy-aware
- this is not a statement of bitwise TensorRT artifact reproducibility
- this is not a claim that the Haeundae FP16 selection generalizes beyond the recorded Jetson Orin environment, Haeundae validation dataset, and custom YOLOv8n model

The baseline COCO YOLOv8n Jetson compare is intentionally scoped as a latency-oriented FP16 vs FP32 result. Accuracy-aware compare wiring has been smoke-tested with existing InferEdgeLab payload examples, but task-matched TensorRT accuracy evidence has not been attached to that baseline COCO FP16/FP32 result.

The Haeundae custom model result is recorded separately as the accuracy-aware validation record. The earlier COCO latency-only result and the RKNN payload smoke test are not mixed into that decision.

## How The System Works

InferEdgeForge and InferEdgeLab are paired, but they do different jobs.

**InferEdgeForge**

- builds deployment artifacts
- records build identity and preset intent
- preserves source and artifact fingerprints
- prepares runtime, benchmark, and compare handoff state

**InferEdgeLab**

- analyzes Runtime result JSON
- compares structured results
- attaches downstream accuracy evidence when available
- interprets trade-offs during validation

This separation matters. Build generation and deployment evaluation are related, but they should not be collapsed into one opaque tool.

Forge intentionally does not ship the `inferedgelab` CLI. Commands such as `evaluate-detection`, `enrich-pair`, and `compare` belong to the InferEdgeLab repository/package; Forge only previews or invokes those downstream handoff commands when Lab is available in the environment.

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

python -m inferedgeforge.cli validate-manifest \
  --build-dir builds/test__jetson__tensorrt__jetson_fp16
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
- worker/runtime summary: compact metadata/manifest projection for Lab worker requests and Runtime invocation planning
- artifact SHA-256: fingerprint of the produced deployment artifact
- source SHA-256: fingerprint of the ONNX input
- `run_summary.json`: persisted downstream execution trace after `run-benchmark`
- `validate-manifest`: build-free sanity check for required Runtime/Lab handoff fields

That is the core difference between this project and a plain conversion wrapper. The system preserves enough state to support later inspection, rebuild, and comparison.

## What This Means

This project is meant to demonstrate more than CLI implementation.

This project is not about running models faster.
It is about making inference experiments inspectable and reproducible, so that deployment decisions can be justified rather than guessed.

It demonstrates the design of a system that supports:

- **experiment traceability**: multiple preset variants can be grouped and reviewed as one experiment surface
- **reproducibility**: build intent, manifests, source fingerprints, and rebuild flows are preserved as explicit records
- **deployment decision support**: compare-ready handoff and benchmark traces help downstream analysis happen with context intact

That is why this project should be read as an experiment workflow system rather than just a model build utility.

## Limitations

The current system is intentionally honest about its boundaries.

- TensorRT engine hashes are not guaranteed to be bitwise stable across rebuilds
- Jetson rebuild validation currently supports functional reproducibility, not bitwise identity
- baseline COCO YOLOv8n Jetson FP16 vs FP32 compare is still documented as latency-only
- Haeundae custom YOLOv8n accuracy-aware validation is scoped to its recorded model, dataset, thresholds, and Jetson environment
- accuracy evidence depends on external evaluation results or downstream InferEdgeLab enrich flows
- some environment details in the Jetson validation record remain `TBD`
- backend toolchains remain environment-dependent
- broader device coverage is still open work

## Documentation Map

- [docs/quickstart.md](docs/quickstart.md): practical end-to-end quickstart
- [docs/handoff.md](docs/handoff.md): Forge to Lab handoff contract
- [docs/runtime_handoff_contract.md](docs/runtime_handoff_contract.md): Forge to Runtime artifact and manifest handoff contract
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
- documented Haeundae custom YOLOv8n TensorRT accuracy-aware validation evidence
- documented manifest-based rebuild validation

It should still be read as a focused and validated foundation, not as a claim that every backend path or deployment workflow is complete.
