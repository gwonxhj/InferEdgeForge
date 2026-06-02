# Agent Manifest Contract

Language: English | [한국어](agent_manifest_contract.ko.md)

`agent_manifest.json` is the first Forge-side contract for the Reliable Edge
Agent Runtime extension. It is intentionally separate from the existing
`manifest.json` contract so the Core 4 validation pipeline remains
backward-compatible.

## Scope

Included:

- agent workload identity
- scheduling priority and latency budget hints
- runtime artifact handoff context
- fallback policy metadata
- telemetry contract version
- optional guard policy pointer
- Lab compatibility mapping

Not included:

- production SaaS orchestration
- cloud deployment control
- general-purpose LLM agent framework
- replacement of `metadata.json` or `manifest.json`

## Flow

```text
Forge manifest.json
-> Forge agent_manifest.json
-> Runtime optional agent result block
-> Orchestrator orchestration_summary.json
-> AIGuard runtime reliability evidence
-> Lab Agent Runtime Reliability Report
```

## Required Fields

| Field | Type | Purpose |
|---|---|---|
| `schema_version` | string | Must be `inferedge-agent-manifest-v1`. |
| `agent_id` | string | Stable agent workload identifier. |
| `agent_type` | string | One of `vision`, `voice`, `safety`, `utility`. |
| `priority` | integer | Scheduler priority from 0 to 100. |
| `latency_budget_ms` | integer | Expected task latency budget. |
| `deadline_ms` | integer | End-to-end task deadline. |
| `input_type` | string | Input shape/category such as `frame` or `text`. |
| `output_type` | string | Output category such as `detections` or `command_result`. |
| `required_backend` | string | Runtime backend expected for this workload. |
| `device_target` | string | Device target such as `jetson` or `cpu`. |
| `precision` | string | Precision such as `fp16`, `fp32`, or `int8`. |
| `runtime_artifact_path` | string | Artifact path consumed by Runtime. |
| `fallback_policy` | object | Runtime policy hint for overload/degraded conditions. |
| `telemetry_contract_version` | string | Telemetry schema family for Orchestrator/Lab handoff. |
| `lab_compat` | object | Minimal mapping for Lab report/deployment decision ingestion. |

Optional:

- `tool_schema_path`
- `guard_policy`

## Fallback Policy

`fallback_policy.mode` must be one of:

- `none`
- `drop_stale`
- `degrade_backend`
- `skip_low_priority`
- `notify_only`

These are policy hints. Orchestrator is responsible for recording actual policy
decisions in `orchestration_summary.json`.

## Lab Compatibility Mapping

`lab_compat` must include:

| Field | Purpose |
|---|---|
| `source_manifest_path` | Link to the original Forge `manifest.json`. |
| `runtime_artifact_path` | Runtime artifact path. Must match top-level `runtime_artifact_path`. |
| `required_backend` | Backend expected by this agent. Must match top-level `required_backend`. |
| `device_target` | Device target. Must match top-level `device_target`. |
| `precision` | Precision. Must match top-level `precision`. |

## Example

See [`tests/fixtures/agent_manifest_vision.json`](../tests/fixtures/agent_manifest_vision.json).

```json
{
  "schema_version": "inferedge-agent-manifest-v1",
  "agent_id": "vision_detector",
  "agent_type": "vision",
  "priority": 90,
  "latency_budget_ms": 33,
  "deadline_ms": 40,
  "input_type": "frame",
  "output_type": "detections",
  "required_backend": "tensorrt",
  "device_target": "jetson",
  "precision": "fp16",
  "runtime_artifact_path": "builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine",
  "fallback_policy": {
    "mode": "drop_stale"
  },
  "telemetry_contract_version": "inferedge-agent-telemetry-v1",
  "guard_policy": {
    "policy_id": "vision_runtime_reliability_v1",
    "required": false
  },
  "lab_compat": {
    "source_manifest_path": "tests/fixtures/runtime_handoff_manifest.json",
    "runtime_artifact_path": "builds/yolov8n__jetson__tensorrt__jetson_fp16/model.engine",
    "required_backend": "tensorrt",
    "device_target": "jetson",
    "precision": "fp16"
  }
}
```

## CLI

Create an agent manifest from an existing Forge manifest:

```bash
python -m inferedgeforge.cli create-agent-manifest \
  --manifest tests/fixtures/runtime_handoff_manifest.json \
  --agent-id vision_detector \
  --agent-type vision \
  --priority 90 \
  --latency-budget-ms 33 \
  --deadline-ms 40 \
  --input-type frame \
  --output-type detections \
  --fallback-mode drop_stale \
  --output agent_manifest.json
```

Validate an agent manifest:

```bash
python -m inferedgeforge.cli validate-agent-manifest \
  --manifest tests/fixtures/agent_manifest_vision.json
```

## Contract Guardrails

- Do not replace `manifest.json` with `agent_manifest.json`.
- Keep agent fields backward-compatible and separate until Runtime/Lab support is
  explicitly added.
- Lab remains the final deployment decision owner.
- Orchestrator records scheduling/policy evidence.
- AIGuard provides deterministic diagnosis evidence.
