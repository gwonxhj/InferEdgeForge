"""Builder exports and backend resolution."""

from __future__ import annotations

from inferedgeforge.builders.base import BaseBuilder, BuildRequest, BuildResult
from inferedgeforge.builders.rknn import RKNNBuilder
from inferedgeforge.builders.tensorrt import TensorRTBuilder


def get_builder(backend: str) -> BaseBuilder:
    if backend == "rknn":
        return RKNNBuilder()
    if backend == "tensorrt":
        return TensorRTBuilder()
    raise ValueError(f"Unsupported backend '{backend}'.")


__all__ = [
    "BaseBuilder",
    "BuildRequest",
    "BuildResult",
    "RKNNBuilder",
    "TensorRTBuilder",
    "get_builder",
]
