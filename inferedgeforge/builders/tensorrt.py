"""TensorRT builder placeholder."""

from __future__ import annotations

from inferedgeforge.builders.base import BaseBuilder, BuildRequest, BuildResult


class TensorRTBuilder(BaseBuilder):
    backend_name = "tensorrt"

    def build(self, request: BuildRequest) -> BuildResult:
        raise NotImplementedError("TensorRT build pipeline is not implemented yet.")
