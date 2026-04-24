"""Engine registry helpers."""

from __future__ import annotations


def create_engine(engine_name: str):
    """Create a runtime engine instance.

    This lightweight fallback tries to load a local registry function if present.
    """

    try:
        from inferedgelab.engines.registry import create_engine as _create_engine
    except ImportError as exc:  # pragma: no cover - integration path
        raise RuntimeError(
            "Engine registry is unavailable. Provide inferedgelab.engines.registry.create_engine."
        ) from exc
    return _create_engine(engine_name)
