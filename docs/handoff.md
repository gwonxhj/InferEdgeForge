# InferEdgeForge to InferEdgeLab Handoff

InferEdgeForge and InferEdgeLab are designed to work as separate stages in the same deployment workflow. InferEdgeForge produces deployment artifacts and structured metadata. InferEdgeLab consumes those outputs to perform runtime validation, comparison, and downstream deployment analysis.

## Responsibility Split

### InferEdgeForge

- Builds deployment artifacts from source ONNX models
- Applies presets and backend-specific build intent
- Emits structured metadata
- Prepares validation-ready handoff information

### InferEdgeLab

- Profiles runtime behavior
- Compares structured results
- Evaluates deployment trade-offs
- Interprets latency, accuracy, and deployment risk in downstream validation

## Handoff Contract

The main handoff object is `metadata.json`.

At a practical level, this metadata connects the information needed to move from build generation into validation review. It records build identity, source model context, artifact records, preset/build context, handoff status, and a Lab-compatible runtime mapping when a downstream profile command can be prepared directly.

When `run-benchmark` has been executed, Forge also persists `run_summary.json` in the build directory. That file records the downstream execution trace, including result paths when InferEdgeLab reports them. `metadata.json` remains the entry point for inspection, and `inspect-build` can include the persisted downstream execution context when that summary is present.

Across multiple builds, Forge can use existing `metadata.json` and `run_summary.json` files to group variants by source model, identify compare-ready builds, and preview an InferEdgeLab `compare` command when two structured result paths are available.

The contract is intentionally simple: Forge is responsible for producing inspectable outputs and structured context around them, while Lab is responsible for consuming that context during validation.

## Handoff Workflow

1. Run `build --dry-run` to preview the expected build plan without creating files.
2. Run `build` to generate actual artifacts and `metadata.json`.
3. Run `inspect-build --summary` or `inspect-build` to review the build, traceability, handoff, and run-summary context.
4. Run `show-lab-profile-input` to view the structured Lab mapping.
5. Run `show-lab-profile-command` to expose the runnable downstream command.
6. Run `run-benchmark` to execute that downstream command from stored handoff metadata.
7. Run `list-builds` to review build variants grouped by source model.
8. Run `show-compare-candidates` to separate benchmark-ready variants from pending ones.
9. Run `show-compare-command` to preview the InferEdgeLab compare command when two ready variants have persisted structured result paths.
10. Execute the compare command in InferEdgeLab and interpret results there.

## Example Commands

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

python -m inferedgeforge.cli inspect-build \
  builds/test__jetson__tensorrt/metadata.json

python -m inferedgeforge.cli inspect-build --summary \
  builds/test__jetson__tensorrt/metadata.json

python -m inferedgeforge.cli show-lab-profile-input \
  builds/test__jetson__tensorrt/metadata.json

python -m inferedgeforge.cli show-lab-profile-command \
  builds/test__jetson__tensorrt/metadata.json

python -m inferedgeforge.cli run-benchmark \
  builds/test__jetson__tensorrt/metadata.json

python -m inferedgeforge.cli list-builds --dir builds

python -m inferedgeforge.cli show-compare-candidates \
  --dir builds \
  --model models/test.onnx

python -m inferedgeforge.cli show-compare-command \
  --dir builds \
  --model models/test.onnx \
  --left tensorrt/jetson_fp16 \
  --right rknn/rk3588_fp16
```

Generated downstream example:

```bash
python -m inferedgelab.cli profile models/test.onnx --engine tensorrt --engine-path builds/test__jetson__tensorrt/model.engine --device-name jetson --precision fp16
```

## Important Notes

- Forge does not benchmark deployment quality.
- Lab does not build deployment artifacts.
- `build --dry-run` previews expected outputs without creating files.
- A real build produces artifacts, metadata, source/artifact fingerprints, and preset snapshots for downstream inspection.
- `show-lab-profile-*` commands expose the handoff mapping without executing Lab.
- `run-benchmark` executes the stored handoff command and persists a Forge-side execution trace.
- `list-builds` and `show-compare-candidates` organize existing build directories; they do not create new metadata.
- `show-compare-command` previews a Lab compare command only when existing run summaries include structured result paths.
- Forge does not compute or interpret compare results.
- The separation is intentional so build generation and validation analysis can remain independent.
