"""Recommendation-only hybrid DSS-PP subprocess adapter."""

from __future__ import annotations

from pathlib import Path

from orchestrator.contracts import MeasurementLogInfo

from .base import AdapterExecution, AdapterSettings, EstimatorPin, run_adapter_process

DEFAULT_HYBRID_PLANNING_COMMAND = (
    "uv",
    "run",
    "--project",
    "{repository}",
    "python",
    "-m",
    "pf.hybrid_planning",
    "--measurement-log",
    "{log_dir}",
    "--config",
    "{config}",
    "--planning-request",
    "{planning_request}",
    "--directive-schedule",
    "{directive_schedule}",
    "--profile",
    "{profile}",
    "--seed",
    "{seed}",
    "--relocation-seed",
    "{relocation_seed}",
    "--output",
    "{output_dir}/hybrid_planning_recommendation.json",
)


class HybridPlanningCLIAdapter:
    """Invoke the PF repository's non-actuating DSS-PP recommendation boundary."""

    def __init__(self, pin: EstimatorPin, settings: AdapterSettings) -> None:
        if pin.name != "particle_filter":
            raise ValueError("HybridPlanningCLIAdapter requires the particle_filter pin.")
        if settings.command_template != DEFAULT_HYBRID_PLANNING_COMMAND:
            raise ValueError("Hybrid planning production command may not be overridden.")
        self.pin = pin
        self.settings = settings

    def run(
        self,
        measurement_log: MeasurementLogInfo,
        *,
        config_path: str | Path,
        planning_request_path: str | Path,
        directive_schedule_path: str | Path,
        output_dir: str | Path,
        execution_dir: str | Path,
        seed: int,
        relocation_seed: int,
        profile: str,
    ) -> AdapterExecution:
        """Return one algorithmic recommendation without authorizing actuation."""
        request = Path(planning_request_path).resolve()
        schedule = Path(directive_schedule_path).resolve()
        for path, label in ((request, "planning request"), (schedule, "directive schedule")):
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(f"Hybrid {label} is invalid: {path}")
        config = Path(config_path).resolve()
        output = Path(output_dir).resolve()
        return run_adapter_process(
            estimator="particle_filter:hybrid_planning_recommendation",
            pin=self.pin,
            settings=self.settings,
            command_values={
                "repository": self.settings.repository_path,
                "log_dir": measurement_log.root,
                "config": config,
                "planning_request": request,
                "directive_schedule": schedule,
                "output_dir": output,
                "seed": int(seed),
                "relocation_seed": int(relocation_seed),
                "profile": profile,
                "stop_after": measurement_log.record_count,
                "mode": "not-applicable",
                "initial_estimate": "not-applicable",
                "snapshot": "not-applicable",
                "snapshot_estimate": "not-applicable",
            },
            measurement_log_sha256=measurement_log.measurement_log_sha256,
            config_path=config,
            output_dir=output,
            execution_dir=Path(execution_dir),
        )


__all__ = ["DEFAULT_HYBRID_PLANNING_COMMAND", "HybridPlanningCLIAdapter"]
