"""Executable raw-spectrum causal offline PF+spectral-MLE hybrid v2."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from orchestrator.adapters.base import AdapterExecution
from orchestrator.contracts import (
    MeasurementLogInfo,
    MLEResultInfo,
    PFCheckpointInfo,
    SpectralMLESnapshotInfo,
    validate_future_spectral_candidate_score_v2,
    validate_measurement_log,
    validate_mle_result,
    validate_pf_checkpoint_v1,
    validate_pf_result,
    validate_pf_rj_receipt_v1,
)
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.estimators.artifacts import repository_commit
from orchestrator.estimators.local_services import local_hybrid_services
from orchestrator.hashing import sha256_file, write_json_atomic
from orchestrator.hybrid.prefix import StationBoundarySchedule
from orchestrator.hybrid.prefix_log import (
    build_and_materialize_measurement_prefix,
    materialize_station_marked_log,
    measurement_records_sha256,
)
from orchestrator.manifests import orchestrator_provenance

from .pf_products import load_pf_spectral_predictions
from .predictive import SpectralHybridScheduler, SpectralPredictiveMonitor
from .rj import build_pf_rj_directive_v1
from .run_config import SpectralHybridMode, SpectralHybridRunConfig
from .score_request import build_future_spectral_score_request_v1
from .scoring import apply_future_spectral_scores, register_snapshot_candidates
from .snapshot import build_spectral_mle_snapshot_v3
from .verification import BlockVerificationQueue


class SpectralOfflineHybridController:
    """Run checkpointed PF prefixes and full-spectrum MLE without observation reuse."""

    def __init__(self, config: SpectralHybridRunConfig) -> None:
        self.config = config

    def run(self) -> Path:
        config = self.config
        target = config.output_directory
        if target.exists():
            raise FileExistsError(f"Hybrid-v2 output already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{target.name}.spectral-v2-{os.getpid()}"
        if staging.exists():
            raise FileExistsError(f"Stale hybrid-v2 staging directory: {staging}")
        staging.mkdir()
        try:
            source_log = validate_measurement_log(config.measurement_log_path)
            if source_log.schema_version != 2:
                raise ContractError("Spectral offline hybrid requires MeasurementLog v2.")
            if config.mode is SpectralHybridMode.WITHIN_MODEL_RELOCATION:
                raise ContractError(
                    "The runtime-only architecture supports verification-only or exact-RJ; "
                    "legacy sibling-estimator relocation is not an active v2 mode."
                )
            boundaries = StationBoundarySchedule.create(
                source_run_id=str(source_log.manifest["run_id"]),
                station_end_steps=config.station_end_steps,
            )
            self._validate_boundaries(source_log, boundaries)
            marked_log, _ = materialize_station_marked_log(
                source_log,
                boundaries,
                staging / "measurement_log.station_marked",
            )
            adapters = local_hybrid_services()
            scheduler = SpectralHybridScheduler(config.scheduler_policy)
            monitor = SpectralPredictiveMonitor()
            queue = BlockVerificationQueue(config.verification_policy)
            executions: dict[str, AdapterExecution] = {}
            checkpoint: PFCheckpointInfo | None = None
            final_pf_log: MeasurementLogInfo | None = None
            previous_mle: MLEResultInfo | None = None
            previous_snapshot: SpectralMLESnapshotInfo | None = None
            snapshots: list[SpectralMLESnapshotInfo] = []
            snapshot_results: dict[str, MLEResultInfo] = {}
            processed_predictions = 0

            for station_id, cutoff_step in boundaries.station_end_steps:
                prefix, prefix_log = build_and_materialize_measurement_prefix(
                    marked_log,
                    cutoff_step=cutoff_step,
                    station_boundaries=boundaries,
                    station_complete_marker=True,
                    output_directory=staging / "prefixes" / f"through-step-{cutoff_step}",
                )
                pf_name = f"pf-through-step-{cutoff_step}"
                pf_output = staging / "results" / pf_name
                checkpoint_path = pf_output / "pf_checkpoint.json"
                execution = adapters["pf_checkpoint"].run(
                    prefix_log,
                    config_path=config.pf_config_path,
                    output_dir=pf_output,
                    checkpoint_out=checkpoint_path,
                    execution_dir=staging / "executions" / pf_name,
                    seed=config.random_seed,
                    stop_after=prefix_log.record_count,
                    checkpoint_in=checkpoint,
                )
                executions[pf_name] = execution
                validate_pf_result(
                    pf_output,
                    expected_variant="pf_strict",
                    expected_isotopes=prefix_log.isotopes,
                    expected_log_sha256=prefix_log.measurement_log_sha256,
                    expected_commit=repository_commit(),
                    expected_config_sha256=sha256_file(config.pf_config_path),
                    expected_resolved_config_sha256=(
                        config.expected_pf_resolved_config_sha256
                    ),
                    expected_record_count=prefix_log.record_count,
                    expected_step_ids=prefix_log.step_ids,
                )
                checkpoint = validate_pf_checkpoint_v1(
                    checkpoint_path,
                    expected_source_run_id=str(prefix_log.manifest["run_id"]),
                    expected_prefix_measurement_log_sha256=(
                        prefix_log.measurement_log_sha256
                    ),
                )
                final_pf_log = prefix_log
                predictions = load_pf_spectral_predictions(
                    pf_output / "pf_spectral_predictions.npz",
                    measurement_log=prefix_log,
                    record_count=prefix_log.record_count,
                )
                trigger = None
                for index in range(processed_predictions, prefix_log.record_count):
                    signal = monitor.record(
                        step_id=prefix_log.step_ids[index],
                        station_id=prefix_log.station_ids[index],
                        prediction_data_cutoff_step=(
                            predictions.prediction_data_cutoff_steps[index]
                        ),
                        station_complete=index + 1 == prefix_log.record_count,
                        observed_spectrum=prefix_log.arrays["spectrum_counts"][index],
                        predicted_spectrum=predictions.predicted_spectra[index],
                    )
                    candidate_trigger = scheduler.consider(signal)
                    if candidate_trigger is not None:
                        if trigger is not None:
                            raise DataReuseError("One station produced multiple spectral triggers.")
                        trigger = candidate_trigger
                processed_predictions = prefix_log.record_count
                if trigger is None:
                    continue
                if (
                    trigger.data_cutoff_step != cutoff_step
                    or trigger.data_cutoff_station != station_id
                ):
                    raise DataReuseError("Spectral trigger differs from the current cutoff.")
                mle_name = f"spectral-mle-through-step-{cutoff_step}"
                mle_output = staging / "results" / mle_name
                if previous_mle is None:
                    mle_execution = adapters["mle_cold"].run(
                        prefix_log,
                        mode="spectral",
                        config_path=config.mle_spectral_config_path,
                        output_dir=mle_output,
                        execution_dir=staging / "executions" / mle_name,
                    )
                else:
                    mle_execution = adapters["mle_warm"].run(
                        prefix_log,
                        mode="spectral",
                        config_path=config.mle_spectral_config_path,
                        initial_estimate_dir=previous_mle.root,
                        output_dir=mle_output,
                        execution_dir=staging / "executions" / mle_name,
                    )
                executions[mle_name] = mle_execution
                mle_result = self._validate_mle(
                    mle_output,
                    prefix_log=prefix_log,
                )
                snapshot = build_spectral_mle_snapshot_v3(
                    output_path=staging
                    / "snapshots"
                    / f"through-step-{cutoff_step}.json",
                    prefix_log=prefix_log,
                    mle_result=mle_result,
                    station_boundaries_sha256=(
                        prefix.covered_station_boundaries_sha256
                    ),
                    covered_records_sha256=prefix.covered_records_sha256,
                    warm_start_snapshot=previous_snapshot,
                )
                register_snapshot_candidates(queue, snapshot)
                snapshots.append(snapshot)
                snapshot_results[str(snapshot.payload["snapshot_id"])] = mle_result
                previous_mle = mle_result
                previous_snapshot = snapshot

            for snapshot in snapshots:
                if snapshot.cutoff_step >= marked_log.step_ids[-1]:
                    continue
                score_name = f"future-score-{snapshot.payload['snapshot_id']}"
                score_output = staging / "results" / score_name
                score_request = build_future_spectral_score_request_v1(
                    output_path=staging / "score_requests" / f"{score_name}.json",
                    snapshot=snapshot,
                    current_log=marked_log,
                )
                score_execution = adapters["future_score"].run(
                    marked_log,
                    config_path=config.mle_spectral_config_path,
                    snapshot_estimate_dir=snapshot_results[
                        str(snapshot.payload["snapshot_id"])
                    ].root,
                    snapshot=snapshot,
                    score_request=score_request,
                    output_dir=score_output,
                    execution_dir=staging / "executions" / score_name,
                )
                executions[score_name] = score_execution
                score = validate_future_spectral_candidate_score_v2(
                    score_output / "future_spectral_candidate_scores.json",
                    expected_snapshot=snapshot,
                    expected_request=score_request,
                )
                apply_future_spectral_scores(queue, snapshot=snapshot, score=score)

            if config.mode is SpectralHybridMode.EXACT_RJ:
                if checkpoint is None or final_pf_log is None:
                    raise ContractError("Exact RJ requires a final PF checkpoint.")
                for snapshot in snapshots:
                    verified = tuple(
                        candidate
                        for candidate in queue.candidates
                        if candidate.snapshot_id == snapshot.payload["snapshot_id"]
                        and candidate.state.value == "verified"
                    )
                    for candidate in verified:
                        checkpoint = self._apply_rj(
                            measurement_log=final_pf_log,
                            checkpoint=checkpoint,
                            snapshot=snapshot,
                            verification_candidate=candidate,
                            adapter=adapters["pf_rj"],
                            executions=executions,
                            staging=staging,
                        )

            final_name = "final-cold-spectral-mle"
            final_output = staging / "results" / final_name
            executions[final_name] = adapters["mle_cold"].run(
                marked_log,
                mode="spectral",
                config_path=config.mle_spectral_config_path,
                output_dir=final_output,
                execution_dir=staging / "executions" / final_name,
            )
            final_mle = self._validate_mle(
                final_output,
                prefix_log=marked_log,
            )
            if final_mle.diagnostics.get("converged") is not True:
                raise ContractError("Final cold spectral MLE must converge.")
            write_json_atomic(staging / "verification_queue.json", queue.to_state())
            write_json_atomic(
                staging / "hybrid_v2_manifest.json",
                {
                    "schema_version": 2,
                    "milestone": "pf_mle_hybrid_v2",
                    "hybrid_run_id": config.hybrid_run_id,
                    "mode": config.mode.value,
                    "measurement_log_sha256": marked_log.measurement_log_sha256,
                    "station_boundary_schedule_sha256": boundaries.schedule_sha256,
                    "spectral_snapshot_sha256": [
                        snapshot.snapshot_sha256 for snapshot in snapshots
                    ],
                    "final_authoritative_result": {
                        "estimator": "cold_full_log_spectral_mle",
                        "result_sha256": final_mle.result_sha256,
                    },
                    "executions": {
                        name: execution.to_dict()
                        for name, execution in sorted(executions.items())
                    },
                    "orchestrator_provenance": orchestrator_provenance(
                        Path(__file__).resolve().parents[3]
                    ),
                    "safety": {
                        "mle_feedback_uses_raw_spectra_only": True,
                        "pf_frontend": "in_repository_full_spectrum_pf_strict",
                        "count_mle_invoked": False,
                        "future_only_verification": True,
                        "direct_mle_weight_increment": False,
                        "hard_prune": False,
                    },
                },
            )
            os.replace(staging, target)
            return target
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _validate_mle(
        self,
        output: Path,
        *,
        prefix_log,
    ) -> MLEResultInfo:
        return validate_mle_result(
            output,
            expected_mode="spectral",
            expected_isotopes=prefix_log.isotopes,
            expected_log_sha256=prefix_log.measurement_log_sha256,
            expected_commit=repository_commit(),
            expected_config_sha256=sha256_file(self.config.mle_spectral_config_path),
            expected_resolved_config_sha256=(
                self.config.expected_mle_resolved_config_sha256
            ),
        )

    def _apply_rj(
        self,
        *,
        measurement_log,
        checkpoint: PFCheckpointInfo,
        snapshot: SpectralMLESnapshotInfo,
        verification_candidate,
        adapter,
        executions: dict[str, AdapterExecution],
        staging: Path,
    ) -> PFCheckpointInfo:
        candidate_id = verification_candidate.candidate_id
        name = f"exact-rj-{snapshot.payload['snapshot_id']}-{candidate_id}"
        directive = build_pf_rj_directive_v1(
            output_path=staging / "directives" / f"{name}.json",
            snapshot=snapshot,
            pf_checkpoint=checkpoint,
            verification_candidates=(verification_candidate,),
            data_cutoff_station=measurement_log.station_ids[-1],
            prefix_measurement_log_sha256=(
                measurement_log.measurement_log_sha256
            ),
            covered_records_sha256=measurement_records_sha256(measurement_log),
            dimension_matching_transform="log_strength_auxiliary_v1",
        )
        output = staging / "results" / name
        checkpoint_path = output / "pf_checkpoint.json"
        receipt_path = output / "pf_rj_receipt.json"
        executions[name] = adapter.run(
            measurement_log,
            config_path=self.config.pf_config_path,
            checkpoint_in=checkpoint,
            directive=directive,
            checkpoint_out=checkpoint_path,
            receipt_output=receipt_path,
            output_dir=output,
            execution_dir=staging / "executions" / name,
            seed=self.config.relocation_seed,
        )
        output_checkpoint = validate_pf_checkpoint_v1(
            checkpoint_path,
            expected_source_run_id=str(measurement_log.manifest["run_id"]),
            expected_prefix_measurement_log_sha256=(
                measurement_log.measurement_log_sha256
            ),
        )
        validate_pf_rj_receipt_v1(
            receipt_path,
            expected_directive=directive,
            expected_output_checkpoint=output_checkpoint,
        )
        return output_checkpoint

    @staticmethod
    def _validate_boundaries(
        log,
        boundaries: StationBoundarySchedule,
    ) -> None:
        expected = StationBoundarySchedule.from_measurement_log(log)
        if boundaries.station_end_steps != expected.station_end_steps:
            raise ContractError("Configured station boundaries differ from the complete log.")


__all__ = ["SpectralOfflineHybridController"]
