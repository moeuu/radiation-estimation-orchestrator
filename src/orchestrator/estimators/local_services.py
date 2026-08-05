"""Adapter-shaped in-process services for the active hybrid-v2 controller.

These wrappers retain the existing execution-manifest structure while removing
all sibling estimator repositories and subprocess estimator dependencies.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from orchestrator.adapters.base import AdapterExecution
from orchestrator.contracts import (
    FutureSpectralScoreRequestInfo,
    MeasurementLogInfo,
    PFCheckpointInfo,
    PFRJDirectiveInfo,
    SpectralMLESnapshotInfo,
)
from orchestrator.errors import ContractError
from orchestrator.hashing import (
    directory_inventory,
    inventory_digest,
    sha256_file,
    write_json_atomic,
)

from .artifacts import repository_commit, repository_root, run_pf_checkpoint, run_spectral_mle
from .future_scoring import score_future_spectra
from .rj import apply_exact_rj


def _execute[T](
    *,
    estimator: str,
    measurement_log: MeasurementLogInfo,
    config_path: Path,
    output_directory: Path,
    execution_directory: Path,
    operation: Callable[[], T],
) -> tuple[T, AdapterExecution]:
    started = datetime.now(UTC)
    started_clock = time.perf_counter()
    result = operation()
    completed = datetime.now(UTC)
    execution_directory.mkdir(parents=True, exist_ok=False)
    stdout = execution_directory / "stdout.txt"
    stderr = execution_directory / "stderr.txt"
    stdout.write_text("in-process estimator completed\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    inventory = directory_inventory(output_directory)
    commit = repository_commit()
    execution = AdapterExecution(
        estimator=estimator,
        requested_revision=commit,
        observed_revision=commit,
        repository_path=repository_root().as_posix(),
        command=("in-process", estimator),
        started_at_utc=started.isoformat(),
        completed_at_utc=completed.isoformat(),
        exit_code=0,
        runtime_s=float(time.perf_counter() - started_clock),
        peak_memory_bytes=0,
        timed_out=False,
        measurement_log_sha256=measurement_log.measurement_log_sha256,
        config_sha256=sha256_file(config_path),
        stdout_path=stdout.as_posix(),
        stdout_sha256=sha256_file(stdout),
        stderr_path=stderr.as_posix(),
        stderr_sha256=sha256_file(stderr),
        output_inventory=inventory,
        output_sha256=inventory_digest(inventory),
        dirty_worktree={},
    )
    write_json_atomic(execution_directory / "adapter_execution.json", execution.to_dict())
    return result, execution


class LocalPFCheckpointService:
    def run(
        self,
        measurement_log: MeasurementLogInfo,
        *,
        config_path: str | Path,
        output_dir: str | Path,
        checkpoint_out: str | Path,
        execution_dir: str | Path,
        seed: int,
        stop_after: int,
        checkpoint_in: PFCheckpointInfo | None,
    ) -> AdapterExecution:
        output = Path(output_dir).resolve()
        expected_checkpoint = Path(checkpoint_out).resolve()
        if expected_checkpoint != output / "pf_checkpoint.json":
            raise ContractError("Local PF checkpoint output must live in its result bundle.")
        if int(stop_after) != measurement_log.record_count:
            raise ContractError("Local PF service requires an exact materialized prefix.")
        _, execution = _execute(
            estimator="local:particle_filter:checkpoint",
            measurement_log=measurement_log,
            config_path=Path(config_path).resolve(),
            output_directory=output,
            execution_directory=Path(execution_dir).resolve(),
            operation=lambda: run_pf_checkpoint(
                measurement_log,
                config_path=config_path,
                output_directory=output,
                random_seed=seed,
                checkpoint_in=checkpoint_in,
            ),
        )
        return execution


class LocalSpectralMLEService:
    def run(
        self,
        measurement_log: MeasurementLogInfo,
        *,
        mode: str,
        config_path: str | Path,
        output_dir: str | Path,
        execution_dir: str | Path,
    ) -> AdapterExecution:
        if mode != "spectral":
            raise ContractError("The active local hybrid supports raw-spectrum MLE only.")
        output = Path(output_dir).resolve()
        _, execution = _execute(
            estimator="local:surface_mle:spectral:cold",
            measurement_log=measurement_log,
            config_path=Path(config_path).resolve(),
            output_directory=output,
            execution_directory=Path(execution_dir).resolve(),
            operation=lambda: run_spectral_mle(
                measurement_log,
                config_path=config_path,
                output_directory=output,
                fit_kind="cold_start_all_history",
            ),
        )
        return execution


class LocalWarmSpectralMLEService:
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
        from orchestrator.contracts import validate_mle_result

        if mode != "spectral":
            raise ContractError("The active local hybrid supports raw-spectrum MLE only.")
        warm = validate_mle_result(initial_estimate_dir, expected_mode="spectral")
        output = Path(output_dir).resolve()
        _, execution = _execute(
            estimator="local:surface_mle:spectral:warm",
            measurement_log=measurement_log,
            config_path=Path(config_path).resolve(),
            output_directory=output,
            execution_directory=Path(execution_dir).resolve(),
            operation=lambda: run_spectral_mle(
                measurement_log,
                config_path=config_path,
                output_directory=output,
                warm_start_result=warm,
                fit_kind="warm_start_prefix",
            ),
        )
        return execution


class LocalFutureSpectralScoreService:
    def run(
        self,
        measurement_log: MeasurementLogInfo,
        *,
        config_path: str | Path,
        snapshot_estimate_dir: str | Path,
        snapshot: SpectralMLESnapshotInfo,
        score_request: FutureSpectralScoreRequestInfo,
        output_dir: str | Path,
        execution_dir: str | Path,
    ) -> AdapterExecution:
        from orchestrator.contracts import validate_mle_result

        estimate = validate_mle_result(snapshot_estimate_dir, expected_mode="spectral")
        output = Path(output_dir).resolve()
        output_file = output / "future_spectral_candidate_scores.json"

        def operation():
            output.mkdir(parents=True, exist_ok=False)
            return score_future_spectra(
                measurement_log,
                config_path=config_path,
                snapshot_result=estimate,
                snapshot=snapshot,
                request=score_request,
                output_path=output_file,
            )

        _, execution = _execute(
            estimator="local:surface_mle:future_spectral_score",
            measurement_log=measurement_log,
            config_path=Path(config_path).resolve(),
            output_directory=output,
            execution_directory=Path(execution_dir).resolve(),
            operation=operation,
        )
        return execution


class LocalExactRJService:
    def run(
        self,
        measurement_log: MeasurementLogInfo,
        *,
        config_path: str | Path,
        checkpoint_in: PFCheckpointInfo,
        directive: PFRJDirectiveInfo,
        checkpoint_out: str | Path,
        receipt_output: str | Path,
        output_dir: str | Path,
        execution_dir: str | Path,
        seed: int,
    ) -> AdapterExecution:
        del seed
        output = Path(output_dir).resolve()
        if Path(checkpoint_out).resolve() != output / "pf_checkpoint.json":
            raise ContractError("Local exact-RJ checkpoint must live in its result bundle.")
        if Path(receipt_output).resolve() != output / "pf_rj_receipt.json":
            raise ContractError("Local exact-RJ receipt must live in its result bundle.")
        _, execution = _execute(
            estimator="local:particle_filter:exact_rj",
            measurement_log=measurement_log,
            config_path=Path(config_path).resolve(),
            output_directory=output,
            execution_directory=Path(execution_dir).resolve(),
            operation=lambda: apply_exact_rj(
                measurement_log,
                config_path=config_path,
                checkpoint_in=checkpoint_in,
                directive=directive,
                output_directory=output,
            ),
        )
        return execution


def local_hybrid_services() -> dict[str, object]:
    return {
        "pf_checkpoint": LocalPFCheckpointService(),
        "pf_rj": LocalExactRJService(),
        "mle_cold": LocalSpectralMLEService(),
        "mle_warm": LocalWarmSpectralMLEService(),
        "future_score": LocalFutureSpectralScoreService(),
    }


__all__ = ["local_hybrid_services"]
