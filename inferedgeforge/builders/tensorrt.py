"""TensorRT builder backed by trtexec."""

from __future__ import annotations

from pathlib import Path
import subprocess

from inferedgeforge.builders.base import BaseBuilder, BuildRequest, BuildResult


class TensorRTBuilder(BaseBuilder):
    backend_name = "tensorrt"

    def _precision_flags(self, request: BuildRequest) -> list[str]:
        precision = request.preset.build_options.get("precision")
        if isinstance(precision, str) and precision.lower() == "fp16":
            return ["--fp16"]
        return []

    def _workspace_flags(self, request: BuildRequest) -> list[str]:
        workspace_mb = request.preset.build_options.get("workspace_mb")
        if isinstance(workspace_mb, int) and workspace_mb > 0:
            return [f"--workspace={workspace_mb}"]
        return []

    def _build_trtexec_command(self, request: BuildRequest, artifact_path: Path) -> list[str]:
        command = [
            "trtexec",
            f"--onnx={request.model_path}",
            f"--saveEngine={artifact_path}",
        ]
        command.extend(self._precision_flags(request))
        command.extend(self._workspace_flags(request))
        return command

    def _run_trtexec(self, command: list[str]) -> None:
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "trtexec is required for TensorRT builds but was not found in PATH. "
                "Run this preset on a Jetson/TensorRT environment with trtexec installed."
            ) from exc

        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            message = (
                "TensorRT engine build failed via trtexec. "
                "Run this preset on a Jetson/TensorRT environment with trtexec installed."
            )
            if details:
                message = f"{message}\n{details}"
            raise RuntimeError(message)

    def build(self, request: BuildRequest) -> BuildResult:
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = output_dir / "model.engine"
        command = self._build_trtexec_command(request, artifact_path)
        self._run_trtexec(command)

        if not artifact_path.is_file():
            raise RuntimeError(
                "trtexec completed without producing model.engine. "
                "Run this preset on a Jetson/TensorRT environment with trtexec installed."
            )

        return BuildResult(
            backend=self.backend_name,
            target=request.preset.target,
            artifact_paths=[str(artifact_path)],
        )
