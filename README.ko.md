# InferEdgeForge

Build provenance and artifact handoff layer  
(ONNX model → TensorRT/RKNN artifacts · metadata · manifest · Runtime/Lab handoff)

언어: [English](README.md) | 한국어

## 요약

- InferEdge validation pipeline의 build/provenance layer입니다.
- ONNX 모델을 TensorRT/RKNN 계열 edge artifact로 변환합니다.
- source model hash, artifact hash, preset, target, precision, shape metadata를 기록합니다.
- Runtime/Lab이 사용할 handoff record를 생성합니다.
- deployment review에 필요한 build evidence를 제공하며, 최종 decision은 InferEdgeLab이 소유합니다.

## 역할 경계 한눈에 보기

| 영역 | Forge가 담당하는 일 | Forge가 담당하지 않는 일 |
|---|---|---|
| Build/provenance | deployment artifact, `metadata.json`, `manifest.json`, source/artifact hash, preset snapshot, handoff record를 생성합니다. | inference 실행, latency benchmark, candidate 비교, deployment decision을 수행하지 않습니다. |
| Runtime/Lab handoff | Runtime/Lab이 사용할 artifact identity, shape, backend, target, precision, worker/runtime summary context를 제공합니다. | Runtime `result.json`, Lab compare output, Lab `deployment_decision` contract를 임의 변경하지 않습니다. |
| Manifest validation | build 실행 없이 handoff 필드와 reproducibility metadata를 sanity check합니다. | artifact를 production promotion하거나 worker/queue를 운영하거나 release readiness를 결정하지 않습니다. |
| Agent manifest extension | 기존 `metadata.json` / `manifest.json`을 대체하지 않는 별도 `agent_manifest.json` contract를 생성합니다. | Orchestrator scheduler, runtime control plane, AIGuard diagnosis owner가 되지 않습니다. |
| Portfolio scope | downstream validation이 evidence를 설명할 수 있도록 artifact build traceability를 보존합니다. | production SaaS, auth/billing/upload service, cloud dashboard, public artifact registry가 되지 않습니다. |

## InferEdgeForge의 차별점

InferEdgeForge는 단순한 model conversion script가 아닙니다.

이 레포는 edge artifact를 재현 가능하고 검토 가능한 단위로 만들기 위해:

- build intent를 structured metadata로 보존하고
- model artifact를 source fingerprint와 연결하며
- benchmark/compare handoff를 추적 가능하게 만들고
- reviewer가 artifact의 rebuild 가능성과 validation 상태를 확인할 수 있게 합니다.

InferEdge는 ONNX 모델을 edge deployment artifact로 변환하고, C++ Runtime 실행 결과와 Lab 분석/deployment decision, optional AIGuard diagnosis evidence까지 연결하는 end-to-end Edge AI inference validation pipeline입니다.

```text
ONNX model
-> InferEdgeForge build/provenance
-> InferEdge-Runtime C++ execution/result export
-> InferEdgeLab compare/API/job/deployment_decision
-> optional InferEdgeAIGuard provenance diagnosis

Experiment hygiene / comparability layer:
InferEdgeEnv -> v0.1.5 v1-complete local-first run evidence registry / comparability checker
```

## 이 레포의 역할

- ONNX 모델을 TensorRT engine, RKNN artifact 등 edge deployment artifact로 변환합니다.
- build 과정의 source model hash, artifact hash, backend, target, precision, shape, preset/build id를 기록합니다.
- `metadata.json`, `manifest.json`, `run_summary.json`, `worker_runtime_summary`를 통해 Runtime/Lab/AIGuard가 추적 가능한 provenance contract를 제공합니다.
- `validate-manifest`로 build 실행 없이 Runtime/Lab handoff에 필요한 manifest 필드를 sanity check할 수 있습니다.
- Reliable Edge Agent Runtime 확장을 위해 독립 `agent_manifest.json` contract를 생성/검증할 수 있습니다.
- Forge는 deployment decision owner가 아닙니다. 최종 decision은 InferEdgeLab이 소유합니다.

## 주요 산출물

- `metadata.json`: backend, target, precision, shape, artifact path, Lab/Runtime handoff context
- `manifest.json`: build id, timestamp, source model hash, artifact hash, preset snapshot, tool version
- `agent_manifest.json`: agent id/type, priority, latency budget, fallback policy, runtime artifact mapping, Lab compatibility metadata
- `run_summary.json`: downstream benchmark/handoff trace
- `worker_runtime_summary`: Lab worker request와 Runtime invocation에 넘기기 쉬운 summary contract

## 빠른 실행

테스트:

```bash
poetry run python3 -m pytest -q
# 또는
python3 -m pytest -q
```

기본 사용 흐름은 preset 기반 build와 inspect/list 명령을 중심으로 합니다. 자세한 CLI 예시는 영어 README와 `docs/`를 참고하세요.

## 다른 InferEdge 레포와의 관계

- **InferEdge-Runtime:** Forge가 만든 artifact와 manifest를 받아 실행/profiling result JSON을 생성합니다.
- **InferEdgeLab:** Runtime result와 Forge provenance를 분석해 report/API/job/deployment decision을 생성합니다.
- **InferEdgeAIGuard:** Forge provenance와 Runtime provenance가 서로 맞는지 rule/evidence 기반으로 진단합니다.
- **InferEdgeEnv:** `v0.1.5` v1-complete 상태의 experiment hygiene / comparability layer로, benchmark run evidence를 local artifact와 SQLite registry로 고정하고 비교 가능성을 판정합니다.

포트폴리오 경계: InferEdgeLab은 validation / decision layer이고, InferEdgeEnv는 `v0.1.5` v1-complete experiment hygiene / comparability layer입니다. Forge는 build provenance를 소유하고, Env는 benchmark evidence가 신뢰 가능하고 비교 가능한 형태인지 관리합니다.

Forge는 `inferedgelab` CLI를 함께 배포하지 않습니다. `evaluate-detection`, `enrich-pair`, `compare` 같은 분석 명령은 InferEdgeLab 레포/패키지의 책임이며, Forge는 Lab이 설치된 환경에서 사용할 downstream handoff command를 preview하거나 실행하는 경계까지만 담당합니다.

## Reliable Edge Agent Runtime 확장

- Forge는 향후 multi-agent workload handoff를 위해 독립 [`agent_manifest.json` contract](docs/agent_manifest_contract.ko.md)를 정의합니다. 대표/canonical 문서는 [English contract](docs/agent_manifest_contract.md)입니다.
- 이 contract는 기존 `manifest.json` / `metadata.json`과 분리되어 Core 4 validation pipeline의 backward compatibility를 보존합니다.
- agent identity, type, priority, latency budget, fallback policy, runtime artifact mapping, telemetry contract version, Lab compatibility metadata를 기록합니다.

## 현재 범위와 future work

현재는 build/provenance/handoff contract를 안정화한 단계입니다. TensorRT/RKNN artifact build와 provenance 기록을 중심으로 하며, production SaaS worker 자동 실행은 포함하지 않습니다.

Jetson validation은 Forge가 생성한 TensorRT artifact가 downstream Runtime/Lab evidence로 이어지는지 확인하는 방향으로 유지합니다.
현재 Runtime/Lab 쪽에는 Forge-generated TensorRT FP16 artifact 기반 25W evidence(mean `10.066401 ms`, p99 `15.548438 ms`, FPS `99.340373`)와 15W power-mode evidence(mean `10.799106 ms`, p99 `15.529218 ms`, FPS `92.600262`)가 기록되어 있습니다.
Forge는 이 수치를 직접 판단하지 않고, build provenance와 artifact handoff를 제공합니다.

Future work:

- future Lab worker-triggered build execution
- metadata/manifest contract 안정화 이후 worker/queue handoff hardening
- manifest validation 기반 artifact promotion sanity check 강화
- 필요가 명확해진 뒤의 target/backend preset 확장
