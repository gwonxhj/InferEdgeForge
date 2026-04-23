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

At a practical level, this metadata is intended to carry the information needed to move from build generation into validation review. That includes build identity, source model information, artifact records, handoff status, and an optional Lab-compatible runtime mapping when a downstream profile command can be prepared directly.

The contract is intentionally simple: Forge is responsible for producing inspectable outputs and structured context around them, while Lab is responsible for consuming that context during validation.

## Handoff Workflow

1. Run `build --dry-run` to preview the expected build plan.
2. Run `build` to generate actual artifacts and metadata.
3. Run `inspect-build` to review the consolidated summary.
4. Run `show-lab-profile-input` to view the structured Lab mapping.
5. Run `show-lab-profile-command` to expose the runnable next-step command.
6. Run `run-benchmark` to execute that downstream command through Forge.

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
- A real build produces artifacts and metadata for downstream inspection.
- The separation is intentional so build generation and validation analysis can remain independent.
