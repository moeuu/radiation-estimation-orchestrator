"""Checkpointed PF and spectral-MLE updates for a live MeasurementLog prefix."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from orchestrator.adapters.base import AdapterExecution, recover_adapter_execution
from orchestrator.contracts import (
    MLEResultInfo,
    PFCheckpointInfo,
    SpectralMLESnapshotInfo,
    validate_future_spectral_candidate_score_v2,
    validate_measurement_log,
    validate_mle_result,
    validate_pf_checkpoint_v1,
    validate_pf_result,
    validate_pf_rj_receipt_v1,
    validate_spectral_mle_snapshot_v3,
)
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.estimators.artifacts import repository_commit
from orchestrator.estimators.local_services import local_hybrid_services
from orchestrator.hashing import load_json, sha256_file, write_json_atomic
from orchestrator.hybrid.prefix import StationBoundarySchedule
from orchestrator.hybrid.prefix_log import measurement_records_sha256

from .mission import RealizedAction
from .pf_products import load_pf_spectral_predictions
from .predictive import (
    SpectralHybridScheduler,
    SpectralPredictiveMonitor,
)
from .rj import build_pf_rj_directive_v1
from .run_config import SpectralHybridMode, SpectralHybridRunConfig
from .score_request import build_future_spectral_score_request_v1
from .scoring import apply_future_spectral_scores, register_snapshot_candidates
from .snapshot import build_spectral_mle_snapshot_v3
from .verification import BlockVerificationQueue


class LiveHybridEstimatorUpdater:
    """Advance estimators only after the runtime has durably published a prefix."""

    def __init__(
        self,
        *,
        config: SpectralHybridRunConfig,
        work_directory: str | Path,
    ) -> None:
        self.config = config
        self.work = Path(work_directory).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work / "live_estimator_state.json"
        if config.mode is SpectralHybridMode.WITHIN_MODEL_RELOCATION:
            raise ContractError(
                "Live v2 runtime-only inference supports verification-only or exact-RJ."
            )
        self.adapters = local_hybrid_services()
        if self.state_path.exists():
            self.state = load_json(self.state_path)
            if self.state.get("schema_version") != 1:
                raise ContractError("Live estimator state has an unsupported schema.")
        else:
            self.state: dict[str, Any] = {
                "schema_version": 1,
                "last_step": -1,
                "measurement_log_path": None,
                "measurement_log_sha256": "0" * 64,
                "pf_checkpoint_path": None,
                "scheduler": SpectralHybridScheduler(
                    config.scheduler_policy
                ).to_state(),
                "verification_queue": BlockVerificationQueue(
                    config.verification_policy
                ).to_state(),
                "snapshots": [],
                "previous_mle_result_path": None,
                "previous_snapshot_path": None,
                "scored_steps_by_snapshot": {},
                "applied_rj_candidates": [],
                "executions": {},
            }
            self._save()

    def update_after_append(self, realized: RealizedAction) -> dict[str, object]:
        log = validate_measurement_log(realized.measurement_log_prefix_path)
        if log.schema_version != 2:
            raise ContractError("Live hybrid estimator updates require MeasurementLog v2.")
        if log.measurement_log_sha256 != realized.measurement_log_prefix_sha256:
            raise ContractError("Runtime receipt and durable MeasurementLog prefix hashes differ.")
        prior_step = int(self.state["last_step"])
        step_ids = tuple(int(record["step_id"]) for record in realized.records)
        expected_steps = tuple(range(prior_step + 1, prior_step + 1 + len(step_ids)))
        if step_ids != expected_steps or tuple(log.step_ids[-len(step_ids) :]) != step_ids:
            raise DataReuseError(
                "Live estimator update must append one exact causal shield program."
            )
        step = step_ids[-1]
        checkpoint = self._checkpoint()
        pf_name = f"pf-live-step-{step}"
        pf_output = self.work / "results" / pf_name
        checkpoint_path = pf_output / "pf_checkpoint.json"
        execution = self._execute_once(
            pf_name,
            output=pf_output,
            operation=partial(
                self.adapters["pf_checkpoint"].run,
                log,
                config_path=self.config.pf_config_path,
                output_dir=pf_output,
                checkpoint_out=checkpoint_path,
                execution_dir=self.work / "executions" / pf_name,
                seed=self.config.random_seed,
                stop_after=log.record_count,
                checkpoint_in=checkpoint,
            ),
        )
        self._record_execution(pf_name, execution)
        validate_pf_result(
            pf_output,
            expected_variant="pf_strict",
            expected_isotopes=log.isotopes,
            expected_log_sha256=log.measurement_log_sha256,
            expected_commit=repository_commit(),
            expected_config_sha256=sha256_file(self.config.pf_config_path),
            expected_resolved_config_sha256=(
                self.config.expected_pf_resolved_config_sha256
            ),
            expected_record_count=log.record_count,
            expected_step_ids=log.step_ids,
        )
        checkpoint = validate_pf_checkpoint_v1(
            checkpoint_path,
            expected_source_run_id=str(log.manifest["run_id"]),
            expected_prefix_measurement_log_sha256=log.measurement_log_sha256,
        )
        predictions = load_pf_spectral_predictions(
            pf_output / "pf_spectral_predictions.npz",
            measurement_log=log,
            record_count=log.record_count,
        )
        monitor = SpectralPredictiveMonitor()
        scheduler = SpectralHybridScheduler.from_state(self.state["scheduler"])
        trigger = None
        first_index = log.record_count - len(step_ids)
        final_signal = None
        for offset, record in enumerate(realized.records):
            metadata = record.get("metadata")
            if not isinstance(metadata, dict) or not isinstance(
                metadata.get("station_complete"), bool
            ):
                raise ContractError("Live runtime records must attest station_complete.")
            index = first_index + offset
            signal = monitor.record(
                step_id=step_ids[offset],
                station_id=int(record["station_id"]),
                prediction_data_cutoff_step=(
                    predictions.prediction_data_cutoff_steps[index]
                ),
                station_complete=bool(metadata["station_complete"]),
                observed_spectrum=log.arrays["spectrum_counts"][index],
                predicted_spectrum=predictions.predicted_spectra[index],
            )
            candidate_trigger = scheduler.consider(signal)
            if trigger is not None and candidate_trigger is not None:
                raise DataReuseError("One shield program produced multiple MLE triggers.")
            trigger = candidate_trigger or trigger
            final_signal = signal
        assert final_signal is not None
        self.state["scheduler"] = scheduler.to_state()
        queue = BlockVerificationQueue.from_state(self.state["verification_queue"])
        if final_signal.station_complete:
            self._score_existing_snapshots(log=log, queue=queue)
            if self.config.mode is SpectralHybridMode.EXACT_RJ:
                checkpoint = self._apply_verified_rj(
                    log=log,
                    queue=queue,
                    checkpoint=checkpoint,
                )
        created_snapshot = None
        if trigger is not None:
            created_snapshot = self._fit_snapshot(
                log=log,
                queue=queue,
                checkpoint=checkpoint,
            )
        self.state["last_step"] = step
        self.state["measurement_log_path"] = log.root.as_posix()
        self.state["measurement_log_sha256"] = log.measurement_log_sha256
        self.state["pf_checkpoint_path"] = checkpoint.path.as_posix()
        self.state["verification_queue"] = queue.to_state()
        self._save()
        return {
            "data_cutoff_step": step,
            "estimator_prefix_sha256": log.measurement_log_sha256,
            "pf_checkpoint_sha256": checkpoint.checkpoint_sha256,
            "spectral_snapshot_id": (
                None if created_snapshot is None else created_snapshot.payload["snapshot_id"]
            ),
        }

    def finalize_after_publish(self, published: dict[str, object]) -> dict[str, object]:
        path = published.get("path")
        if not isinstance(path, str):
            raise ContractError("Runtime publication lacks MeasurementLog path.")
        log = validate_measurement_log(path)
        declared_hash = published.get("measurement_log_sha256")
        if declared_hash is not None and declared_hash != log.measurement_log_sha256:
            raise ContractError("Published MeasurementLog hash differs from runtime receipt.")
        name = "final-live-cold-spectral-mle"
        output = self.work / "results" / name
        execution = self._execute_once(
            name,
            output=output,
            operation=partial(
                self.adapters["mle_cold"].run,
                log,
                mode="spectral",
                config_path=self.config.mle_spectral_config_path,
                output_dir=output,
                execution_dir=self.work / "executions" / name,
            ),
        )
        self._record_execution(name, execution)
        result = self._validate_mle(output, log)
        if result.diagnostics.get("converged") is not True:
            raise ContractError("Final live cold spectral MLE must converge.")
        self._save()
        return {
            "estimator": "cold_full_log_spectral_mle",
            "result_path": result.root.as_posix(),
            "result_sha256": result.result_sha256,
            "measurement_log_path": log.root.as_posix(),
            "measurement_log_sha256": log.measurement_log_sha256,
        }

    def planning_context(self) -> dict[str, object]:
        queue = BlockVerificationQueue.from_state(self.state["verification_queue"])
        snapshots = {
            str(snapshot.payload["snapshot_id"]): snapshot
            for snapshot in (
                validate_spectral_mle_snapshot_v3(path)
                for path in self.state["snapshots"]
            )
        }
        external = []
        for candidate in queue.candidates:
            planner_weight = queue.planner_weight(candidate)
            if planner_weight <= 0:
                continue
            snapshot = snapshots[candidate.snapshot_id]
            raw_candidates = snapshot.payload["candidates"]
            assert isinstance(raw_candidates, list)
            source = next(
                raw
                for raw in raw_candidates
                if isinstance(raw, dict) and raw["candidate_id"] == candidate.candidate_id
            )
            covariance = np.asarray(source["covariance_xyz"], dtype=float)
            external.append(
                {
                    "snapshot_id": candidate.snapshot_id,
                    "candidate_id": candidate.candidate_id,
                    "isotope": source["isotope"],
                    "position_xyz": source["centroid_xyz"],
                    "strength_cps_1m": source["integrated_strength_cps_1m"],
                    "spread_m": float(np.sqrt(max(0.0, np.trace(covariance)))),
                    "state": candidate.state.value,
                    "planner_weight": planner_weight,
                }
            )
        return {
            "measurement_log_path": self.state["measurement_log_path"],
            "measurement_log_sha256": self.state["measurement_log_sha256"],
            "pf_checkpoint_path": self.state["pf_checkpoint_path"],
            "external_candidates": external,
        }

    def _fit_snapshot(
        self,
        *,
        log,
        queue: BlockVerificationQueue,
        checkpoint: PFCheckpointInfo,
    ) -> SpectralMLESnapshotInfo:
        del checkpoint
        step = log.step_ids[-1]
        name = f"spectral-live-through-step-{step}"
        output = self.work / "results" / name
        previous_result_path = self.state["previous_mle_result_path"]
        if previous_result_path is None:
            operation = partial(
                self.adapters["mle_cold"].run,
                log,
                mode="spectral",
                config_path=self.config.mle_spectral_config_path,
                output_dir=output,
                execution_dir=self.work / "executions" / name,
            )
        else:
            operation = partial(
                self.adapters["mle_warm"].run,
                log,
                mode="spectral",
                config_path=self.config.mle_spectral_config_path,
                initial_estimate_dir=previous_result_path,
                output_dir=output,
                execution_dir=self.work / "executions" / name,
            )
        execution = self._execute_once(name, output=output, operation=operation)
        self._record_execution(name, execution)
        result = self._validate_mle(output, log)
        boundaries = StationBoundarySchedule.from_measurement_log(log)
        previous_snapshot_path = self.state["previous_snapshot_path"]
        previous_snapshot = (
            None
            if previous_snapshot_path is None
            else validate_spectral_mle_snapshot_v3(previous_snapshot_path)
        )
        snapshot = build_spectral_mle_snapshot_v3(
            output_path=self.work / "snapshots" / f"through-step-{step}.json",
            prefix_log=log,
            mle_result=result,
            station_boundaries_sha256=boundaries.schedule_sha256,
            covered_records_sha256=measurement_records_sha256(log),
            warm_start_snapshot=previous_snapshot,
        )
        register_snapshot_candidates(queue, snapshot)
        self.state["snapshots"].append(snapshot.path.as_posix())
        self.state["previous_mle_result_path"] = result.root.as_posix()
        self.state["previous_snapshot_path"] = snapshot.path.as_posix()
        self.state["scored_steps_by_snapshot"][str(snapshot.payload["snapshot_id"])] = []
        return snapshot

    def _score_existing_snapshots(
        self,
        *,
        log,
        queue: BlockVerificationQueue,
    ) -> None:
        for raw_path in tuple(self.state["snapshots"]):
            snapshot = validate_spectral_mle_snapshot_v3(raw_path)
            previous = tuple(
                int(value)
                for value in self.state["scored_steps_by_snapshot"].get(
                    str(snapshot.payload["snapshot_id"]), []
                )
            )
            available = [
                step
                for step in log.step_ids
                if step > snapshot.cutoff_step and step not in set(previous)
            ]
            if not available:
                continue
            name = f"live-score-{snapshot.payload['snapshot_id']}-step-{log.step_ids[-1]}"
            request = build_future_spectral_score_request_v1(
                output_path=self.work / "score_requests" / f"{name}.json",
                snapshot=snapshot,
                current_log=log,
                previously_scored_step_ids=previous,
            )
            snapshot_result_path = self._snapshot_result_path(snapshot)
            output = self.work / "results" / name
            execution = self._execute_once(
                name,
                output=output,
                operation=partial(
                    self.adapters["future_score"].run,
                    log,
                    config_path=self.config.mle_spectral_config_path,
                    snapshot_estimate_dir=snapshot_result_path,
                    snapshot=snapshot,
                    score_request=request,
                    output_dir=output,
                    execution_dir=self.work / "executions" / name,
                ),
            )
            self._record_execution(name, execution)
            score = validate_future_spectral_candidate_score_v2(
                output / "future_spectral_candidate_scores.json",
                expected_snapshot=snapshot,
                expected_request=request,
            )
            apply_future_spectral_scores(queue, snapshot=snapshot, score=score)
            self.state["scored_steps_by_snapshot"][
                str(snapshot.payload["snapshot_id"])
            ] = [*previous, *available]

    def _apply_verified_rj(
        self,
        *,
        log,
        queue: BlockVerificationQueue,
        checkpoint: PFCheckpointInfo,
    ) -> PFCheckpointInfo:
        from .verification import CandidateState

        snapshots = {
            str(snapshot.payload["snapshot_id"]): snapshot
            for snapshot in (
                validate_spectral_mle_snapshot_v3(path)
                for path in self.state["snapshots"]
            )
        }
        applied = set(
            str(value) for value in self.state.get("applied_rj_candidates", [])
        )
        for candidate in queue.candidates:
            key = f"{candidate.snapshot_id}.{candidate.candidate_id}"
            if candidate.state is not CandidateState.VERIFIED or key in applied:
                continue
            snapshot = snapshots[candidate.snapshot_id]
            name = f"live-exact-rj-{key}-step-{log.step_ids[-1]}"
            directive = build_pf_rj_directive_v1(
                output_path=self.work / "directives" / f"{name}.json",
                snapshot=snapshot,
                pf_checkpoint=checkpoint,
                verification_candidates=(candidate,),
                data_cutoff_station=log.station_ids[-1],
                prefix_measurement_log_sha256=log.measurement_log_sha256,
                covered_records_sha256=measurement_records_sha256(log),
                dimension_matching_transform="log_strength_auxiliary_v1",
            )
            output = self.work / "results" / name
            checkpoint_path = output / "pf_checkpoint.json"
            receipt_path = output / "pf_rj_receipt.json"
            execution = self._execute_once(
                name,
                output=output,
                operation=partial(
                    self.adapters["pf_rj"].run,
                    log,
                    config_path=self.config.pf_config_path,
                    checkpoint_in=checkpoint,
                    directive=directive,
                    checkpoint_out=checkpoint_path,
                    receipt_output=receipt_path,
                    output_dir=output,
                    execution_dir=self.work / "executions" / name,
                    seed=self.config.relocation_seed,
                ),
            )
            self._record_execution(name, execution)
            output_checkpoint = validate_pf_checkpoint_v1(
                checkpoint_path,
                expected_source_run_id=str(log.manifest["run_id"]),
                expected_prefix_measurement_log_sha256=(
                    log.measurement_log_sha256
                ),
            )
            validate_pf_rj_receipt_v1(
                receipt_path,
                expected_directive=directive,
                expected_output_checkpoint=output_checkpoint,
            )
            checkpoint = output_checkpoint
            applied.add(key)
        self.state["applied_rj_candidates"] = sorted(applied)
        return checkpoint

    def _checkpoint(self) -> PFCheckpointInfo | None:
        path = self.state.get("pf_checkpoint_path")
        return None if path is None else validate_pf_checkpoint_v1(path)

    def _validate_mle(self, output: Path, log) -> MLEResultInfo:
        return validate_mle_result(
            output,
            expected_mode="spectral",
            expected_isotopes=log.isotopes,
            expected_log_sha256=log.measurement_log_sha256,
            expected_commit=repository_commit(),
            expected_config_sha256=sha256_file(self.config.mle_spectral_config_path),
            expected_resolved_config_sha256=(
                self.config.expected_mle_resolved_config_sha256
            ),
        )

    def _snapshot_result_path(self, snapshot: SpectralMLESnapshotInfo) -> Path:
        for path in (self.work / "results").glob("spectral-live-through-step-*"):
            try:
                result = validate_mle_result(path, expected_mode="spectral")
            except ContractError:
                continue
            if result.result_sha256 == snapshot.payload["fit"]["mle_result_sha256"]:  # type: ignore[index]
                return path
        raise ContractError("Could not find the MLE result bound to spectral snapshot.")

    def _record_execution(self, name: str, execution: AdapterExecution) -> None:
        self.state["executions"][name] = execution.to_dict()

    def _execute_once(
        self,
        name: str,
        *,
        output: Path,
        operation: Callable[[], AdapterExecution],
    ) -> AdapterExecution:
        receipt = self.work / "executions" / name / "adapter_execution.json"
        if receipt.exists():
            return recover_adapter_execution(receipt, output_directory=output)
        # A process crash may occur after deterministic estimator artifacts were
        # published but before the execution receipt was fsynced.  Such a bundle
        # is unauthenticated, so quarantine-by-discard and recompute it from the
        # same immutable prefix instead of treating it as a completed call.
        execution_directory = receipt.parent
        for stale in (output, execution_directory):
            try:
                stale.resolve().relative_to(self.work)
            except ValueError as exc:  # pragma: no cover - construction invariant
                raise ContractError("Estimator recovery path escaped its work root.") from exc
            if stale.exists():
                shutil.rmtree(stale)
        return operation()

    def _save(self) -> None:
        write_json_atomic(self.state_path, self.state, overwrite=True)


__all__ = ["LiveHybridEstimatorUpdater"]
