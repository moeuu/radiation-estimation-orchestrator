"""Frozen-snapshot future-only verification subprocess adapter."""

from __future__ import annotations

from pathlib import Path

from orchestrator.contracts import MeasurementLogInfo

from .base import AdapterExecution, AdapterSettings, EstimatorPin, run_adapter_process

DEFAULT_FUTURE_SCORE_COMMAND = (
    "uv",
    "run",
    "--project",
    "{repository}",
    "estimate-radiation-mle",
    "score-future",
    "--run-dir",
    "{log_dir}",
    "--mle-config",
    "{config}",
    "--snapshot-estimate",
    "{snapshot_estimate}",
    "--snapshot",
    "{snapshot}",
    "--output",
    "{output_dir}/future_candidate_scores.json",
    "--json",
)


class FutureScoreCLIAdapter:
    """Evaluate one frozen count-MLE snapshot on a later exact prefix."""

    def __init__(self, pin: EstimatorPin, settings: AdapterSettings) -> None:
        if pin.name != "surface_mle":
            raise ValueError("FutureScoreCLIAdapter requires the surface_mle pin.")
        if settings.command_template != DEFAULT_FUTURE_SCORE_COMMAND:
            raise ValueError("Future-score production command may not be overridden.")
        self.pin = pin
        self.settings = settings

    def run(
        self,
        measurement_log: MeasurementLogInfo,
        *,
        config_path: str | Path,
        snapshot_estimate_dir: str | Path,
        snapshot_path: str | Path,
        output_dir: str | Path,
        execution_dir: str | Path,
    ) -> AdapterExecution:
        """Score strictly post-cutoff rows without refitting snapshot parameters."""
        if (
            measurement_log.schema_version
            != self.pin.expected_measurement_log_schema_version
        ):
            raise ValueError(
                "Future-score MLE pin and MeasurementLog schema versions differ."
            )
        estimate = Path(snapshot_estimate_dir).resolve()
        snapshot = Path(snapshot_path).resolve()
        if estimate.is_symlink() or not estimate.is_dir():
            raise FileNotFoundError(f"Snapshot MLE report directory is invalid: {estimate}")
        if snapshot.is_symlink() or not snapshot.is_file():
            raise FileNotFoundError(f"MLESnapshot artifact is invalid: {snapshot}")
        config = Path(config_path).resolve()
        output = Path(output_dir).resolve()
        return run_adapter_process(
            estimator="surface_mle:count:future_score",
            pin=self.pin,
            settings=self.settings,
            command_values={
                "repository": self.settings.repository_path,
                "log_dir": measurement_log.root,
                "config": config,
                "snapshot_estimate": estimate,
                "snapshot": snapshot,
                "output_dir": output,
                "seed": 0,
                "relocation_seed": 0,
                "stop_after": measurement_log.record_count,
                "profile": "not-applicable",
                "mode": "not-applicable",
                "directive_schedule": "not-applicable",
                "initial_estimate": "not-applicable",
            },
            measurement_log_sha256=measurement_log.measurement_log_sha256,
            config_path=config,
            output_dir=output,
            execution_dir=Path(execution_dir),
        )


__all__ = ["DEFAULT_FUTURE_SCORE_COMMAND", "FutureScoreCLIAdapter"]
