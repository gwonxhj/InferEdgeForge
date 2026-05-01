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
```

## 이 레포의 역할

- ONNX 모델을 TensorRT engine, RKNN artifact 등 edge deployment artifact로 변환합니다.
- build 과정의 source model hash, artifact hash, backend, target, precision, shape, preset/build id를 기록합니다.
- `metadata.json`, `manifest.json`, `run_summary.json`, `worker_runtime_summary`를 통해 Runtime/Lab/AIGuard가 추적 가능한 provenance contract를 제공합니다.
- Forge는 deployment decision owner가 아닙니다. 최종 decision은 InferEdgeLab이 소유합니다.

## 주요 산출물

- `metadata.json`: backend, target, precision, shape, artifact path, Lab/Runtime handoff context
- `manifest.json`: build id, timestamp, source model hash, artifact hash, preset snapshot, tool version
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

## 현재 범위와 future work

현재는 build/provenance/handoff contract를 안정화한 단계입니다. TensorRT/RKNN artifact build와 provenance 기록을 중심으로 하며, production SaaS worker 자동 실행은 포함하지 않습니다.

Future work:

- Lab job에서 Forge build를 자동 실행하는 production worker
- queue/DB 기반 artifact promotion workflow
- 더 넓은 target/backend preset 확장
- production deployment control
