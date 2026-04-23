"""RKNN builder implementation."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from inferedgeforge.builders.base import BaseBuilder, BuildRequest, BuildResult


def _load_rknn_class() -> type[Any]:
    try:
        module = importlib.import_module("rknn.api")
        return module.RKNN
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "RKNN toolkit is unavailable in this environment. Install and run "
            "InferEdgeForge RKNN builds in a compatible Linux environment with "
            "rknn-toolkit2."
        ) from exc


def _require_success(stage: str, result: object) -> None:
    if result in (None, 0):
        return
    raise RuntimeError(f"Failed to {stage} for RKNN build.")


class RKNNBuilder(BaseBuilder):
    backend_name = "rknn"

    def build(self, request: BuildRequest) -> BuildResult:
        model_path = Path(request.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        rknn_cls = _load_rknn_class()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / "model.rknn"

        build_options = request.preset.build_options
        config_kwargs: dict[str, object] = {}
        target_platform = build_options.get("target_platform", request.preset.target)
        if isinstance(target_platform, str) and target_platform:
            config_kwargs["target_platform"] = target_platform
        if "optimization_level" in build_options:
            config_kwargs["optimization_level"] = build_options["optimization_level"]

        do_quantization = bool(build_options.get("quantization", False))

        rknn = rknn_cls()
        try:
            try:
                _require_success("configure RKNN", rknn.config(**config_kwargs))
            except Exception as exc:
                raise RuntimeError("Failed to configure RKNN.") from exc

            try:
                _require_success("load ONNX into RKNN", rknn.load_onnx(model=str(model_path)))
            except Exception as exc:
                raise RuntimeError("Failed to load ONNX into RKNN.") from exc

            try:
                _require_success("build RKNN artifact", rknn.build(do_quantization=do_quantization))
            except Exception as exc:
                raise RuntimeError("Failed to build RKNN artifact.") from exc

            try:
                _require_success("export RKNN artifact", rknn.export_rknn(str(artifact_path)))
            except Exception as exc:
                raise RuntimeError("Failed to export RKNN artifact.") from exc
        finally:
            release = getattr(rknn, "release", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass

        return BuildResult(
            backend=self.backend_name,
            target=request.preset.target,
            artifact_paths=[str(artifact_path)],
        )
