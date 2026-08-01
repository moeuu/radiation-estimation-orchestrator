"""Pure-PF sequential replay subprocess adapter."""

from __future__ import annotations

from pathlib import Path

from orchestrator.contracts import MeasurementLogInfo

from .base import AdapterExecution, AdapterSettings, EstimatorPin, run_adapter_process

DEFAULT_PF_COMMAND = (
    "uv",
    "run",
    "--project",
    "{repository}",
    "python",
    "-m",
    "pf.replay",
    "--measurement-log",
    "{log_dir}",
    "--config",
    "{config}",
    "--profile",
    "{profile}",
    "--output-dir",
    "{output_dir}",
    "--seed",
    "{seed}",
)


class PFCLIAdapter:
    """Invoke pure PF through its public CLI without importing estimator source."""

    def __init__(self, pin: EstimatorPin, settings: AdapterSettings) -> None:
        if pin.name != "particle_filter":
            raise ValueError("PFCLIAdapter requires the particle_filter pin.")
        self.pin = pin
        self.settings = settings

    def run(
        self,
        measurement_log: MeasurementLogInfo,
        *,
        config_path: str | Path,
        output_dir: str | Path,
        execution_dir: str | Path,
        seed: int,
        profile: str = "pf_strict",
    ) -> AdapterExecution:
        if (
            measurement_log.schema_version
            != self.pin.expected_measurement_log_schema_version
        ):
            raise ValueError(
                "PF pin expects MeasurementLog schema version "
                f"{self.pin.expected_measurement_log_schema_version}, got "
                f"{measurement_log.schema_version}."
            )
        if profile not in {"pf_strict", "pf_profiled", "pf_online_profiled"}:
            raise ValueError(f"Unsupported pure-PF profile: {profile!r}")
        config = Path(config_path).resolve()
        output = Path(output_dir).resolve()
        return run_adapter_process(
            estimator=f"particle_filter:{profile}",
            pin=self.pin,
            settings=self.settings,
            command_values={
                "repository": self.settings.repository_path,
                "log_dir": measurement_log.root,
                "config": config,
                "output_dir": output,
                "seed": int(seed),
                "profile": profile,
                "mode": "count",
            },
            measurement_log_sha256=measurement_log.measurement_log_sha256,
            config_path=config,
            output_dir=output,
            execution_dir=Path(execution_dir),
        )
