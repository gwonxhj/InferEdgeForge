from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_readmes_expose_forge_role_boundaries():
    readme = _read("README.md")
    readme_ko = _read("README.ko.md")

    assert "Language: English | [한국어](README.ko.md)" in readme
    assert "언어: [English](README.md) | 한국어" in readme_ko

    for required in [
        "## Role Boundary At A Glance",
        "`metadata.json`, `manifest.json`, source/artifact hashes",
        "Execute inference, benchmark latency, compare candidates",
        "Mutate Runtime `result.json`, Lab compare output",
        "future Lab worker-triggered build execution",
        "Become production SaaS, auth/billing/upload service, cloud dashboard",
    ]:
        assert required in readme

    assert "automatic SaaS worker execution" not in readme

    for required in [
        "## 역할 경계 한눈에 보기",
        "`metadata.json`, `manifest.json`, source/artifact hash",
        "inference 실행, latency benchmark, candidate 비교",
        "Runtime `result.json`, Lab compare output",
        "future Lab worker-triggered build execution",
        "production SaaS, auth/billing/upload service, cloud dashboard",
    ]:
        assert required in readme_ko


def test_agent_manifest_contract_has_korean_quick_guide_and_boundaries():
    readme = _read("README.md")
    readme_ko = _read("README.ko.md")
    contract = _read("docs/agent_manifest_contract.md")
    contract_ko = _read("docs/agent_manifest_contract.ko.md")

    assert "Language: English | [한국어](agent_manifest_contract.ko.md)" in contract
    assert "언어: [English](agent_manifest_contract.md) | 한국어" in contract_ko
    assert "[Agent Manifest Contract](agent_manifest_contract.md)" in contract_ko
    assert "대표/canonical 문서" in contract_ko

    assert (
        "[`agent_manifest.json` contract](docs/agent_manifest_contract.md)"
        in readme
    )
    assert (
        "[한국어: agent_manifest.json 계약 quick guide]"
        "(docs/agent_manifest_contract.ko.md)"
        in readme
    )
    assert (
        "[`agent_manifest.json` contract](docs/agent_manifest_contract.ko.md)"
        in readme_ko
    )
    assert "[English contract](docs/agent_manifest_contract.md)" in readme_ko

    for required in [
        "`agent_manifest.json`",
        "`metadata.json` / `manifest.json`",
        "production SaaS orchestration",
        "cloud deployment control",
        "general-purpose LLM agent framework",
        "Orchestrator scheduler",
        "AIGuard diagnosis ownership",
        "Lab-owned deployment decision",
        "production control plane",
        "AI OS",
        "Jetson 필요 여부",
    ]:
        assert required in contract_ko
