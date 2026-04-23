# InferEdgeForge Examples

This directory contains minimal example flows for understanding how InferEdgeForge build outputs are prepared and handed off into InferEdgeLab.

## Current Example Flow

The current local flow is intentionally small:

- Create a simple test ONNX file.
- Run a build with a preset.
- Inspect the generated metadata.
- Derive Lab profile input from the build metadata.
- Derive a runnable Lab profile command for downstream validation.
- Execute the downstream validation step from handoff-ready metadata.

### Minimal Local Build Flow

```bash
mkdir -p models
echo "dummy onnx content" > models/test.onnx

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

This final step turns the handoff-ready metadata into an executed downstream validation step.

## What These Examples Demonstrate

- Forge creates deployment artifacts and structured metadata.
- Forge does not benchmark or score deployment quality.
- Forge prepares handoff-ready information for InferEdgeLab.
- InferEdgeLab is the downstream validation tool.

## Planned Example Expansion

- RKNN-oriented build examples.
- TensorRT-oriented build examples.
- Metadata inspection examples.
- Forge -> Lab handoff examples.
