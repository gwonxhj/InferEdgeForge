# Agent Manifest Contract 한국어 Quick Guide

언어: [English](agent_manifest_contract.md) | 한국어

이 문서는 한국어 빠른 안내서입니다. 대표/canonical 문서는
[Agent Manifest Contract](agent_manifest_contract.md)입니다.

`agent_manifest.json`은 Reliable Edge Agent Runtime 확장을 위한 Forge-side
handoff contract입니다. 기존 `metadata.json` / `manifest.json`을 대체하지
않고, Core 4 validation pipeline의 backward compatibility를 보존하는 별도
계약으로 유지합니다.

## 핵심 역할

Forge는 agent workload가 어떤 artifact와 runtime expectation을 갖는지 기록합니다.
하지만 Forge는 scheduling, runtime control, diagnosis, deployment decision을
소유하지 않습니다.

```text
Forge manifest.json
-> Forge agent_manifest.json
-> Runtime optional agent result block
-> Orchestrator orchestration_summary.json
-> AIGuard deterministic runtime reliability evidence
-> Lab Agent Runtime Reliability Report
-> Lab-owned deployment decision
```

## 포함하는 것

| 영역 | 의미 |
|---|---|
| agent identity | `agent_id`, `agent_type`으로 workload를 식별 |
| scheduling hints | `priority`, `latency_budget_ms`, `deadline_ms` |
| runtime handoff | backend, device target, precision, runtime artifact path |
| fallback metadata | `drop_stale`, `degrade_backend`, `skip_low_priority` 같은 policy hint |
| telemetry contract | Orchestrator/Lab handoff를 위한 telemetry schema family |
| Lab compatibility | Lab report/deployment decision ingestion에 필요한 최소 mapping |

## 포함하지 않는 것

- production SaaS orchestration
- cloud deployment control
- general-purpose LLM agent framework
- Orchestrator scheduler 구현
- AIGuard diagnosis ownership
- Lab-owned deployment decision 대체
- 기존 `metadata.json` 또는 `manifest.json` 대체

## 경계

`agent_manifest.json`은 policy hint와 handoff metadata를 담는 Forge artifact입니다.
실제 queue/deadline/fallback decision evidence는 Orchestrator가 기록합니다.
AIGuard는 deterministic diagnosis evidence를 제공하고, 최종 판단은 Lab-owned
deployment decision으로 유지됩니다.

이 문서는 Runtime Operation / Agent Runtime 확장을 production control plane이나
AI OS로 표현하지 않기 위한 계약 경계 문서입니다.

## Jetson 필요 여부

이 문서를 읽거나 링크를 검증하는 작업에는 Jetson 기기가 필요 없습니다. 실제
Jetson artifact build, device-local execution, live telemetry replay를 새로 수집할
때만 Jetson 기기가 필요합니다.
