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

When `run-benchmark` has been executed, Forge also persists `run_summary.json` in the build directory. `metadata.json` remains the entry point for inspection, and `inspect-build` can include the persisted downstream execution context when that summary is present.

The contract is intentionally simple: Forge is responsible for producing inspectable outputs and structured context around them, while Lab is responsible for consuming that context during validation.

## Handoff Workflow

1. Run `build --dry-run` to preview the expected build plan without creating files.
2. Run `build` to generate actual artifacts and `metadata.json`.
3. Run `inspect-build --summary` or `inspect-build` to review the build, traceability, handoff, and run-summary context.
4. Run `show-lab-profile-input` to view the structured Lab mapping.
5. Run `show-lab-profile-command` to expose the runnable downstream command.
6. Run `run-benchmark` to execute that downstream command from stored handoff metadata.

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
- The separation is intentional so build generation and validation analysis can remain independent.
