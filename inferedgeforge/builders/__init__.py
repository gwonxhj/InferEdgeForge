"""Builder exports and backend resolution."""

from __future__ import annotations

from inferedgeforge.builders.base import BaseBuilder, BuildRequest, BuildResult
from inferedgeforge.builders.rknn import RKNNBuilder
from inferedgeforge.builders.tensorrt import TensorRTBuilder


SUPPORTED_BUILDERS = ("rknn", "tensorrt")


def get_builder(backend: str) -> BaseBuilder:
    if backend == "rknn":
        return RKNNBuilder()
    if backend == "tensorrt":
        return TensorRTBuilder()
    supported = ", ".join(SUPPORTED_BUILDERS)
    raise ValueError(f"Unsupported backend '{backend}'. Supported backends: {supported}.")


__all__ = [
    "BaseBuilder",
    "BuildRequest",
    "BuildResult",
    "RKNNBuilder",
    "TensorRTBuilder",
    "get_builder",
]
