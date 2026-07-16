"""Standalone count/spectral surface-MLE replay subprocess adapter."""

from __future__ import annotations

from pathlib import Path

from orchestrator.contracts import MeasurementLogInfo

from .base import AdapterExecution, AdapterSettings, EstimatorPin, run_adapter_process

DEFAULT_MLE_COMMAND = (
    "uv",
    "run",
    "--project",
    "{repository}",
    "estimate-radiation-mle",
    "{mode}",
    "--run-dir",
    "{log_dir}",
    "--mle-config",
    "{config}",
    "--output-dir",
    "{output_dir}",
    "--overwrite",
    "--cpu",
    "--json",
)


class MLECLIAdapter:
    """Invoke the standalone MLE through its public CLI only."""

    def __init__(self, pin: EstimatorPin, settings: AdapterSettings) -> None:
        if pin.name != "surface_mle":
            raise ValueError("MLECLIAdapter requires the surface_mle pin.")
        self.pin = pin
        self.settings = settings

    def run(
        self,
        measurement_log: MeasurementLogInfo,
        *,
        mode: str,
        config_path: str | Path,
        output_dir: str | Path,
        execution_dir: str | Path,
    ) -> AdapterExecution:
        command_mode = {"count": "replay", "spectral": "fit-spectrum"}.get(mode)
        if command_mode is None:
            raise ValueError(f"Unsupported MLE mode: {mode!r}")
        config = Path(config_path).resolve()
        output = Path(output_dir).resolve()
        return run_adapter_process(
            estimator=f"surface_mle:{mode}",
            pin=self.pin,
            settings=self.settings,
            command_values={
                "repository": self.settings.repository_path,
                "log_dir": measurement_log.root,
                "config": config,
                "output_dir": output,
                "seed": 0,
                "profile": "not-applicable",
                "mode": command_mode,
            },
            measurement_log_sha256=measurement_log.measurement_log_sha256,
            config_path=config,
            output_dir=output,
            execution_dir=Path(execution_dir),
        )
