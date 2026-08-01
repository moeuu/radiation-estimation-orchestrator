"""Warm-start standalone surface-MLE subprocess adapter."""

from __future__ import annotations

from pathlib import Path

from orchestrator.contracts import MeasurementLogInfo

from .base import AdapterExecution, AdapterSettings, EstimatorPin, run_adapter_process

DEFAULT_WARM_MLE_COMMAND = (
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
    "--initial-estimate",
    "{initial_estimate}",
    "--output-dir",
    "{output_dir}",
    "--overwrite",
    "--cpu",
    "--json",
)


class WarmMLECLIAdapter:
    """Invoke a complete-prefix MLE using a prior report only as initialization."""

    def __init__(self, pin: EstimatorPin, settings: AdapterSettings) -> None:
        if pin.name != "surface_mle":
            raise ValueError("WarmMLECLIAdapter requires the surface_mle pin.")
        if settings.command_template != DEFAULT_WARM_MLE_COMMAND:
            raise ValueError("Warm MLE production command may not be overridden.")
        self.pin = pin
        self.settings = settings

    def run(
        self,
        measurement_log: MeasurementLogInfo,
        *,
        mode: str,
        config_path: str | Path,
        initial_estimate_dir: str | Path,
        output_dir: str | Path,
        execution_dir: str | Path,
    ) -> AdapterExecution:
        """Run one warm all-history fit on the supplied exact prefix."""
        if (
            measurement_log.schema_version
            != self.pin.expected_measurement_log_schema_version
        ):
            raise ValueError(
                "Warm MLE pin expects MeasurementLog schema version "
                f"{self.pin.expected_measurement_log_schema_version}, got "
                f"{measurement_log.schema_version}."
            )
        command_mode = {"count": "replay", "spectral": "fit-spectrum"}.get(mode)
        if command_mode is None:
            raise ValueError(f"Unsupported MLE mode: {mode!r}")
        initial = Path(initial_estimate_dir).resolve()
        if initial.is_symlink() or not initial.is_dir():
            raise FileNotFoundError(f"Warm-start report directory is invalid: {initial}")
        config = Path(config_path).resolve()
        output = Path(output_dir).resolve()
        return run_adapter_process(
            estimator=f"surface_mle:{mode}:warm",
            pin=self.pin,
            settings=self.settings,
            command_values={
                "repository": self.settings.repository_path,
                "log_dir": measurement_log.root,
                "config": config,
                "initial_estimate": initial,
                "output_dir": output,
                "mode": command_mode,
                "seed": 0,
                "relocation_seed": 0,
                "stop_after": measurement_log.record_count,
                "profile": "not-applicable",
                "directive_schedule": "not-applicable",
            },
            measurement_log_sha256=measurement_log.measurement_log_sha256,
            config_path=config,
            output_dir=output,
            execution_dir=Path(execution_dir),
        )


__all__ = ["DEFAULT_WARM_MLE_COMMAND", "WarmMLECLIAdapter"]
