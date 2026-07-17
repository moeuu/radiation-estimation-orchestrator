"""Opt-in target-preserving PF hybrid replay subprocess adapter."""

from __future__ import annotations

from pathlib import Path

from orchestrator.contracts import MeasurementLogInfo

from .base import AdapterExecution, AdapterSettings, EstimatorPin, run_adapter_process

DEFAULT_HYBRID_PF_COMMAND = (
    "uv",
    "run",
    "--project",
    "{repository}",
    "python",
    "-m",
    "pf.hybrid_replay",
    "--measurement-log",
    "{log_dir}",
    "--config",
    "{config}",
    "--directive-schedule",
    "{directive_schedule}",
    "--profile",
    "{profile}",
    "--output-dir",
    "{output_dir}",
    "--seed",
    "{seed}",
    "--relocation-seed",
    "{relocation_seed}",
    "--stop-after",
    "{stop_after}",
)


class HybridPFCLIAdapter:
    """Invoke the generic PF external-control boundary without importing PF code."""

    def __init__(self, pin: EstimatorPin, settings: AdapterSettings) -> None:
        if pin.name != "particle_filter":
            raise ValueError("HybridPFCLIAdapter requires the particle_filter pin.")
        if settings.command_template != DEFAULT_HYBRID_PF_COMMAND:
            raise ValueError("Hybrid PF production command may not be overridden.")
        self.pin = pin
        self.settings = settings

    def run(
        self,
        measurement_log: MeasurementLogInfo,
        *,
        config_path: str | Path,
        directive_schedule_path: str | Path,
        output_dir: str | Path,
        execution_dir: str | Path,
        seed: int,
        relocation_seed: int,
        stop_after: int,
        profile: str = "pf_strict",
    ) -> AdapterExecution:
        """Replay exactly ``stop_after`` rows and apply cutoff-bound directives."""
        schedule = Path(directive_schedule_path).resolve()
        if schedule.is_symlink() or not schedule.is_file():
            raise FileNotFoundError(f"Directive schedule is invalid: {schedule}")
        count = int(stop_after)
        if count < 1 or count > measurement_log.record_count:
            raise ValueError("stop_after must select a non-empty MeasurementLog prefix.")
        config = Path(config_path).resolve()
        output = Path(output_dir).resolve()
        return run_adapter_process(
            estimator="particle_filter:hybrid_relocation",
            pin=self.pin,
            settings=self.settings,
            command_values={
                "repository": self.settings.repository_path,
                "log_dir": measurement_log.root,
                "config": config,
                "directive_schedule": schedule,
                "output_dir": output,
                "seed": int(seed),
                "relocation_seed": int(relocation_seed),
                "stop_after": count,
                "profile": profile,
                "mode": "not-applicable",
                "initial_estimate": "not-applicable",
            },
            measurement_log_sha256=measurement_log.measurement_log_sha256,
            config_path=config,
            output_dir=output,
            execution_dir=Path(execution_dir),
        )


__all__ = ["DEFAULT_HYBRID_PF_COMMAND", "HybridPFCLIAdapter"]
