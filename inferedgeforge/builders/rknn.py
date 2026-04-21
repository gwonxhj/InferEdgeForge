"""RKNN builder placeholder."""

from __future__ import annotations

from inferedgeforge.builders.base import BaseBuilder, BuildRequest, BuildResult


class RKNNBuilder(BaseBuilder):
    backend_name = "rknn"

    def build(self, request: BuildRequest) -> BuildResult:
        raise NotImplementedError("RKNN build pipeline is not implemented yet.")
