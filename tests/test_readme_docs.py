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
