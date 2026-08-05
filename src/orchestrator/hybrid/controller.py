"""Executable causal offline controller for proposal-only PF+MLE replay."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from orchestrator.adapters import (
    FutureScoreCLIAdapter,
    HybridPFCLIAdapter,
    HybridPlanningCLIAdapter,
    MLECLIAdapter,
    WarmMLECLIAdapter,
    load_estimator_pins,
)
from orchestrator.adapters.base import (
    AdapterExecution,
    EstimatorPin,
    settings_from_dict,
)
from orchestrator.adapters.future_score_cli import DEFAULT_FUTURE_SCORE_COMMAND
from orchestrator.adapters.hybrid_mle_cli import DEFAULT_WARM_MLE_COMMAND
from orchestrator.adapters.hybrid_pf_cli import DEFAULT_HYBRID_PF_COMMAND
from orchestrator.adapters.hybrid_planning_cli import DEFAULT_HYBRID_PLANNING_COMMAND
from orchestrator.adapters.mle_cli import DEFAULT_MLE_COMMAND
from orchestrator.contracts import (
    FutureCandidateScoreInfo,
    HybridPlanningRecommendationInfo,
    MeasurementLogInfo,
    MLEResultInfo,
    MLESnapshotInfo,
    PFDirectiveInfo,
    PFDirectiveReceiptInfo,
    PFResultInfo,
    validate_future_candidate_score,
    validate_hybrid_ledger_summary,
    validate_hybrid_planning_recommendation,
    validate_hybrid_planning_request,
    validate_measurement_log,
    validate_mle_result,
    validate_mle_snapshot_v2,
    validate_pf_directive,
    validate_pf_directive_receipt,
    validate_pf_result,
)
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import (
    canonical_json_bytes,
    load_json,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)
from orchestrator.manifests import orchestrator_provenance

from .directives import (
    CandidateOutcome,
    DirectiveProposal,
    PFDirective,
    PFDirectiveReceipt,
    build_pf_directive,
)
from .ledger import ObservationUseLedger
from .predictive_monitor import PredictiveMonitor
from .prefix import StationBoundarySchedule
from .prefix_log import (
    build_and_materialize_measurement_prefix,
    materialize_station_marked_log,
    measurement_records_sha256,
)
from .reporting import build_hybrid_result, write_hybrid_result_bundle
from .run_config import HybridRunConfig
from .scheduler import HybridScheduler, HybridTrigger
from .snapshot import MLESnapshot, SnapshotCluster, SnapshotPrediction
from .verification_queue import CandidateState, VerificationQueue


class HybridController:
    """Run deterministic PF prefixes, causal MLE snapshots, and safe directives."""

    def __init__(self, config: HybridRunConfig) -> None:
        self.config = config

    def run(self) -> Path:
        """Execute inference only; this method never receives or opens truth."""
        config = self.config
        target = config.output_directory
        if target.exists():
            raise FileExistsError(f"Hybrid output already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{target.name}.hybrid-staging"
        if staging.exists():
            raise FileExistsError(f"Stale hybrid staging directory exists: {staging}")
        staging.mkdir()
        try:
            source_log = validate_measurement_log(config.measurement_log_path)
            if source_log.schema_version != 1:
                raise ContractError(
                    "Causal hybrid v1 requires the archived projected-count "
                    "MeasurementLog v1. Raw full-spectrum v2 must use a separately "
                    "versioned spectral-prefix hybrid contract; deriving legacy isotope "
                    "counts inside the orchestrator is forbidden."
                )
            boundaries = StationBoundarySchedule.create(
                source_run_id=str(source_log.manifest["run_id"]),
                station_end_steps=config.station_end_steps,
            )
            self._validate_explicit_boundaries(source_log, boundaries)
            marked_log, _ = materialize_station_marked_log(
                source_log,
                boundaries,
                staging / "measurement_log.station_marked",
            )
            pins = load_estimator_pins(config.pin_registry_path)
            (
                pf_adapter,
                cold_mle_adapter,
                warm_mle_adapter,
                future_score_adapter,
                planning_adapter,
            ) = self._build_adapters(pins)
            ledger = ObservationUseLedger(
                source_run_id=boundaries.source_run_id,
                station_boundary_schedule_sha256=boundaries.schedule_sha256,
            )
            queue = VerificationQueue(config.hybrid_policy)
            scheduler = HybridScheduler(config.hybrid_policy)
            monitor = PredictiveMonitor()
            directives: list[PFDirective] = []
            directive_infos: dict[str, PFDirectiveInfo] = {}
            receipt_hashes: dict[str, str] = {}
            receipt_infos: dict[str, PFDirectiveReceiptInfo] = {}
            executions: dict[str, AdapterExecution] = {}
            snapshots: list[MLESnapshot] = []
            snapshot_infos: list[MLESnapshotInfo] = []
            snapshot_paths: dict[str, Path] = {}
            snapshot_results: dict[str, MLEResultInfo] = {}
            proposal_bindings: dict[tuple[str, str], tuple[str, str]] = {}
            future_score_infos: list[FutureCandidateScoreInfo] = []
            future_corroboration_count = 0
            planning_recommendations: list[HybridPlanningRecommendationInfo] = []
            previous_mle_result: MLEResultInfo | None = None
            previous_snapshot: MLESnapshot | None = None
            previous_snapshot_info: MLESnapshotInfo | None = None
            processed_prediction_count = 0

            write_json_atomic(
                staging / "station_boundary_schedule.json",
                {
                    "schema_version": 1,
                    "source_run_id": boundaries.source_run_id,
                    "station_end_steps": [
                        {"station_id": station, "terminal_step_id": step}
                        for station, step in boundaries.station_end_steps
                    ],
                    "schedule_sha256": boundaries.schedule_sha256,
                },
            )
            write_json_atomic(
                staging / "resolved_hybrid_policy.json",
                config.hybrid_policy.to_dict(),
            )

            for station_id, cutoff_step in boundaries.station_end_steps:
                record_count = marked_log.step_ids.index(cutoff_step) + 1
                schedule_path = self._write_directive_schedule(
                    staging / "directive_schedules" / f"before-step-{cutoff_step}.json",
                    directives,
                )
                run_name = f"pf_prefix_step_{cutoff_step}"
                execution = pf_adapter.run(
                    marked_log,
                    config_path=config.pf_config_path,
                    directive_schedule_path=schedule_path,
                    output_dir=staging / "results" / run_name,
                    execution_dir=staging / "executions" / run_name,
                    seed=config.random_seed,
                    relocation_seed=config.relocation_seed,
                    stop_after=record_count,
                    profile=config.pf_profile,
                )
                executions[run_name] = execution
                pf_result = self._validate_pf_run(
                    marked_log,
                    pins["particle_filter"],
                    output=staging / "results" / run_name,
                    execution=execution,
                    record_count=record_count,
                )
                self._ingest_receipts(
                    pf_result.root,
                    directives={item.directive_id: item for item in directives},
                    directive_infos=directive_infos,
                    receipt_hashes=receipt_hashes,
                    receipt_infos=receipt_infos,
                    ledger=ledger,
                )
                prediction_rows = self._load_predictive_rows(
                    pf_result.root, marked_log, record_count
                )
                if len(prediction_rows) < processed_prediction_count:
                    raise DataReuseError("A later PF prefix lost prior predictive rows.")
                trigger: HybridTrigger | None = None
                for record_index in range(processed_prediction_count, record_count):
                    signal = self._predictive_signal(
                        marked_log,
                        prediction_rows[record_index],
                        record_index=record_index,
                        boundaries=boundaries,
                        monitor=monitor,
                    )
                    candidate_trigger = scheduler.consider(signal)
                    if candidate_trigger is not None:
                        if trigger is not None:
                            raise DataReuseError(
                                "One PF prefix produced multiple unhandled triggers."
                            )
                        trigger = candidate_trigger
                processed_prediction_count = record_count
                should_score = self._has_scoreable_candidates(
                    queue,
                    receipt_infos=receipt_infos,
                    cutoff_step=cutoff_step,
                )
                should_plan = cutoff_step in config.planning_requests
                if trigger is None and not should_score and not should_plan:
                    continue
                if trigger is not None and (
                    trigger.data_cutoff_step != cutoff_step
                    or trigger.data_cutoff_station != station_id
                ):
                    raise DataReuseError("Scheduler trigger differs from current station boundary.")

                prefix, prefix_log = build_and_materialize_measurement_prefix(
                    marked_log,
                    cutoff_step=cutoff_step,
                    station_boundaries=boundaries,
                    station_complete_marker=True,
                    output_directory=(staging / "prefixes" / f"through-step-{cutoff_step}"),
                )
                if should_score:
                    consumed, scores = self._score_future_candidates(
                        current_prefix=prefix_log,
                        current_covered_records_sha256=prefix.covered_records_sha256,
                        current_cutoff_step=cutoff_step,
                        adapter=future_score_adapter,
                        pin=pins["surface_mle"],
                        queue=queue,
                        ledger=ledger,
                        directives=directives,
                        receipt_infos=receipt_infos,
                        snapshot_infos=snapshot_infos,
                        snapshot_paths=snapshot_paths,
                        snapshot_results=snapshot_results,
                        proposal_bindings=proposal_bindings,
                        staging=staging,
                        executions=executions,
                    )
                    future_corroboration_count += consumed
                    future_score_infos.extend(scores)
                if trigger is None:
                    if should_plan:
                        planning_recommendations.append(
                            self._run_planning_recommendation(
                                log=marked_log,
                                station_id=station_id,
                                cutoff_step=cutoff_step,
                                record_count=record_count,
                                template=config.planning_requests[cutoff_step],
                                queue=queue,
                                directives=directives,
                                adapter=planning_adapter,
                                staging=staging,
                                executions=executions,
                            )
                        )
                    continue
                mle_name = f"mle_prefix_count_step_{cutoff_step}"
                mle_output = staging / "results" / mle_name
                if previous_mle_result is None:
                    mle_execution = cold_mle_adapter.run(
                        prefix_log,
                        mode="count",
                        config_path=config.mle_count_config_path,
                        output_dir=mle_output,
                        execution_dir=staging / "executions" / mle_name,
                    )
                else:
                    mle_execution = warm_mle_adapter.run(
                        prefix_log,
                        mode="count",
                        config_path=config.mle_count_config_path,
                        initial_estimate_dir=previous_mle_result.root,
                        output_dir=mle_output,
                        execution_dir=staging / "executions" / mle_name,
                    )
                executions[mle_name] = mle_execution
                mle_result = self._validate_mle_run(
                    prefix_log,
                    pins["surface_mle"],
                    output=mle_output,
                    execution=mle_execution,
                    mode="count",
                    expected_resolved_hash=config.expected_resolved_config_sha256["mle_count"],
                    expected_covered_records_sha256=prefix.covered_records_sha256,
                    expected_fit_kind=(
                        "cold_start_all_history"
                        if previous_mle_result is None
                        else "warm_start_all_history"
                    ),
                    expected_station_boundary_attestation=("covered_prefix_markers_v1"),
                    expected_previous_mle_result=previous_mle_result,
                )
                snapshot = self._build_snapshot(
                    trigger=trigger,
                    prefix=prefix,
                    result=mle_result,
                    previous_snapshot=previous_snapshot,
                    previous_result=previous_mle_result,
                    estimator_commit=pins["surface_mle"].revision,
                )
                snapshot_path = write_json_atomic(
                    staging / "snapshots" / f"{snapshot.snapshot_id}.json",
                    snapshot.to_dict(),
                )
                snapshot_info = validate_mle_snapshot_v2(
                    snapshot_path,
                    expected_covered_step_ids=prefix.covered_step_ids,
                    expected_source_run_id=prefix.source_run_id,
                    expected_prefix_log_sha256=prefix.prefix_measurement_log_sha256,
                    expected_covered_records_sha256=prefix.covered_records_sha256,
                    expected_covered_station_boundaries_sha256=(
                        prefix.covered_station_boundaries_sha256
                    ),
                    expected_previous_snapshot=previous_snapshot_info,
                    expected_previous_mle_result=previous_mle_result,
                )
                if snapshot_info.snapshot_sha256 != snapshot.sha256:
                    raise ContractError("MLESnapshot changed while being validated.")
                ledger.register_snapshot(snapshot)
                snapshots.append(snapshot)
                snapshot_infos.append(snapshot_info)
                snapshot_paths[snapshot.snapshot_id] = snapshot_path
                snapshot_results[snapshot.snapshot_id] = mle_result

                active_isotopes = self._active_pf_isotopes(pf_result)
                proposals = tuple(
                    DirectiveProposal.from_cluster(
                        cluster,
                        candidate_mean_xyz=mean_xyz,
                        proposal_kernel=config.proposal_kernel.for_candidate(
                            cluster.integrated_strength_cps_1m
                        ),
                    )
                    for cluster in snapshot.clusters
                    if cluster.isotope in active_isotopes
                    for mean_xyz in (self._pf_domain_candidate_mean(cluster, marked_log),)
                    if mean_xyz is not None
                )
                if proposals:
                    directive = build_pf_directive(
                        snapshot,
                        config=config.hybrid_policy,
                        pf_resolved_config_sha256=(
                            config.expected_resolved_config_sha256["pf_strict"]
                        ),
                        proposals=proposals,
                        provenance={
                            "hybrid_run_id": config.hybrid_run_id,
                            "proposal_role": "target_preserving_mcmc_only",
                            "source_mle_result_sha256": mle_result.result_sha256,
                        },
                    )
                    directive_path = write_json_atomic(
                        staging / "directives" / f"{directive.directive_id}.json",
                        directive.to_dict(),
                    )
                    directive_info = validate_pf_directive(
                        directive_path, expected_snapshot=snapshot_info
                    )
                    if directive_info.directive_sha256 != directive.sha256:
                        raise ContractError("PFDirective changed while being validated.")
                    ledger.issue_directive(directive)
                    queue.register(directive)
                    directives.append(directive)
                    directive_infos[directive.directive_id] = directive_info
                    for proposal in directive.proposals:
                        proposal_bindings[
                            (snapshot.snapshot_id, proposal.snapshot_candidate_id)
                        ] = (directive.directive_id, proposal.proposal_id)
                previous_mle_result = mle_result
                previous_snapshot = snapshot
                previous_snapshot_info = snapshot_info
                if should_plan:
                    planning_recommendations.append(
                        self._run_planning_recommendation(
                            log=marked_log,
                            station_id=station_id,
                            cutoff_step=cutoff_step,
                            record_count=record_count,
                            template=config.planning_requests[cutoff_step],
                            queue=queue,
                            directives=directives,
                            adapter=planning_adapter,
                            staging=staging,
                            executions=executions,
                        )
                    )

            final_schedule = self._write_directive_schedule(
                staging / "directive_schedules" / "final.json",
                directives,
            )
            final_pf_execution = pf_adapter.run(
                marked_log,
                config_path=config.pf_config_path,
                directive_schedule_path=final_schedule,
                output_dir=staging / "results" / "hybrid_pf_final",
                execution_dir=staging / "executions" / "hybrid_pf_final",
                seed=config.random_seed,
                relocation_seed=config.relocation_seed,
                stop_after=marked_log.record_count,
                profile=config.pf_profile,
            )
            executions["hybrid_pf_final"] = final_pf_execution
            final_pf = self._validate_pf_run(
                marked_log,
                pins["particle_filter"],
                output=staging / "results" / "hybrid_pf_final",
                execution=final_pf_execution,
                record_count=marked_log.record_count,
            )
            self._ingest_receipts(
                final_pf.root,
                directives={item.directive_id: item for item in directives},
                directive_infos=directive_infos,
                receipt_hashes=receipt_hashes,
                receipt_infos=receipt_infos,
                ledger=ledger,
            )
            missing_receipts = sorted(set(directive_infos) - set(receipt_hashes))
            if missing_receipts:
                raise DataReuseError(
                    f"Final PF replay did not receipt directives: {missing_receipts}."
                )

            final_count, count_execution = self._run_final_mle(
                marked_log,
                pins["surface_mle"],
                cold_mle_adapter,
                mode="count",
                output=staging / "results" / "mle_count_final",
                execution_dir=staging / "executions" / "mle_count_final",
            )
            executions["mle_count_final"] = count_execution
            final_spectral, spectral_execution = self._run_final_mle(
                marked_log,
                pins["surface_mle"],
                cold_mle_adapter,
                mode="spectral",
                output=staging / "results" / "mle_spectral_final",
                execution_dir=staging / "executions" / "mle_spectral_final",
            )
            executions["mle_spectral_final"] = spectral_execution

            ledger_path = write_json_atomic(
                staging / "observation_use_ledger.json", ledger.summary()
            )
            ledger_info = validate_hybrid_ledger_summary(ledger_path)
            queue_path = write_json_atomic(
                staging / "verification_queue.json",
                {
                    "schema_version": 1,
                    "scoring_status": "future_only_frozen_snapshot_scoring_complete",
                    "score_family": ("frozen_count_snapshot_cluster_log_predictive_ratio"),
                    "score_artifacts": [
                        {
                            "snapshot_id": info.payload["snapshot_id"],
                            "through_step_id": info.future_step_ids[-1],
                            "sha256": info.score_sha256,
                        }
                        for info in future_score_infos
                    ],
                    "all_history_mle_persistence_used_as_corroboration": False,
                    "candidates": [
                        {
                            "directive_id": candidate.directive_id,
                            "proposal_id": candidate.proposal_id,
                            "snapshot_candidate_id": candidate.snapshot_candidate_id,
                            "data_cutoff_step": candidate.data_cutoff_step,
                            "corroboration_min_step": candidate.corroboration_min_step,
                            "state": candidate.state.value,
                            "future_step_ids": list(candidate.future_step_ids),
                            "cumulative_log_predictive_likelihood_ratio": (
                                candidate.cumulative_log_predictive_likelihood_ratio
                            ),
                        }
                        for candidate in queue.candidates
                    ],
                },
            )
            verification_counts = {"pending": 0, "verified": 0, "quarantined": 0}
            for candidate in queue.candidates:
                verification_counts[candidate.state.value] += 1
            repository_root = Path(__file__).resolve().parents[3]
            provenance = orchestrator_provenance(repository_root)
            orchestrator_commit = provenance.get("commit")
            if not isinstance(orchestrator_commit, str) or not orchestrator_commit:
                raise ContractError("Could not resolve orchestrator commit for hybrid report.")
            execution_evidence_path = write_json_atomic(
                staging / "execution_evidence.json",
                {
                    "schema_version": 1,
                    "hybrid_run_id": config.hybrid_run_id,
                    "source_run_id": str(marked_log.manifest["run_id"]),
                    "source_measurement_log_sha256": source_log.measurement_log_sha256,
                    "inference_measurement_log_sha256": marked_log.measurement_log_sha256,
                    "hybrid_config_sha256": sha256_file(config.source_path),
                    "pin_registry_sha256": sha256_file(config.pin_registry_path),
                    "orchestrator": provenance,
                    "pins": {name: asdict(pin) for name, pin in sorted(pins.items())},
                    "executions": {
                        name: self._retarget_execution(execution, staging).to_dict()
                        for name, execution in sorted(executions.items())
                    },
                    "validated_artifacts": {
                        "final_pf_result_sha256": final_pf.result_sha256,
                        "final_count_mle_result_sha256": final_count.result_sha256,
                        "final_spectral_mle_result_sha256": final_spectral.result_sha256,
                        "ledger_sha256": ledger_info.summary_sha256,
                        "verification_queue_sha256": sha256_file(queue_path),
                        "future_candidate_score_sha256": [
                            info.score_sha256 for info in future_score_infos
                        ],
                        "planning_recommendation_sha256": [
                            info.recommendation_sha256 for info in planning_recommendations
                        ],
                    },
                },
            )
            execution_evidence_sha256 = sha256_file(execution_evidence_path)
            hybrid_payload = build_hybrid_result(
                hybrid_run_id=config.hybrid_run_id,
                hybrid_mode=config.hybrid_policy.mode,
                measurement_log=marked_log,
                source_measurement_log_sha256=source_log.measurement_log_sha256,
                source_measurement_log_record_count=source_log.record_count,
                station_boundary_schedule_sha256=boundaries.schedule_sha256,
                final_pf_result=final_pf,
                final_count_mle_result=final_count,
                final_spectral_mle_result=final_spectral,
                ledger=ledger_info,
                snapshots=tuple(snapshot_infos),
                directives=tuple(directive_infos.values()),
                receipts=tuple(receipt_infos.values()),
                future_candidate_scores=tuple(future_score_infos),
                planning_recommendations=tuple(planning_recommendations),
                verification_queue_sha256=sha256_file(queue_path),
                verification_counts=verification_counts,
                orchestrator_commit=orchestrator_commit,
                orchestrator_source_provenance=provenance,
                hybrid_config_sha256=sha256_file(config.source_path),
                pin_registry_sha256=sha256_file(config.pin_registry_path),
                execution_evidence_sha256=execution_evidence_sha256,
            )
            hybrid_result, _ = write_hybrid_result_bundle(
                staging,
                hybrid_payload,
                expected_measurement_log=marked_log,
                expected_source_measurement_log=source_log,
                expected_pf_result=final_pf,
                expected_final_count_mle_result=final_count,
                expected_final_spectral_mle_result=final_spectral,
                expected_ledger=ledger_info,
                expected_snapshots=tuple(snapshot_infos),
                expected_directives=tuple(directive_infos.values()),
                expected_receipts=tuple(receipt_infos.values()),
                expected_future_candidate_scores=tuple(future_score_infos),
                expected_planning_recommendations=tuple(planning_recommendations),
                expected_verification_queue_sha256=sha256_file(queue_path),
            )
            manifest = {
                "schema_version": 1,
                "hybrid_run_id": config.hybrid_run_id,
                "status": "inference_complete",
                "measurement_log": {
                    "source_sha256": source_log.measurement_log_sha256,
                    "station_marked_sha256": marked_log.measurement_log_sha256,
                    "source_run_id": str(marked_log.manifest["run_id"]),
                    "record_count": marked_log.record_count,
                    "station_boundary_schedule_sha256": boundaries.schedule_sha256,
                },
                "hybrid_policy": config.hybrid_policy.to_dict(),
                "execution_evidence_sha256": execution_evidence_sha256,
                "proposal_kernel": asdict(config.proposal_kernel),
                "pins": {name: asdict(pin) for name, pin in sorted(pins.items())},
                "snapshots": [
                    {
                        "snapshot_id": snapshot.snapshot_id,
                        "snapshot_sha256": snapshot.sha256,
                        "data_cutoff_step": snapshot.prefix.data_cutoff_step,
                    }
                    for snapshot in snapshots
                ],
                "directives": [
                    {
                        "directive_id": directive.directive_id,
                        "directive_sha256": directive.sha256,
                        "data_cutoff_step": directive.snapshot.prefix.data_cutoff_step,
                    }
                    for directive in directives
                ],
                "receipts": dict(sorted(receipt_hashes.items())),
                "ledger_sha256": ledger_info.summary_sha256,
                "verification_queue_sha256": sha256_file(queue_path),
                "future_candidate_scores": [
                    {
                        "snapshot_id": info.payload["snapshot_id"],
                        "future_step_ids": list(info.future_step_ids),
                        "sha256": info.score_sha256,
                    }
                    for info in future_score_infos
                ],
                "planning_recommendations": [
                    {
                        "recommendation_id": info.payload["recommendation_id"],
                        "data_cutoff_step": info.payload["causal_boundary"][  # type: ignore[index]
                            "data_cutoff_step"
                        ],
                        "sha256": info.recommendation_sha256,
                        "robot_actuation_authorized": False,
                    }
                    for info in planning_recommendations
                ],
                "final_outputs": {
                    "hybrid_pf": final_pf.result_sha256,
                    "count_mle": final_count.result_sha256,
                    "spectral_mle": final_spectral.result_sha256,
                    "hybrid_result": hybrid_result.result_sha256,
                    "authoritative_final_report": "spectral_mle",
                },
                "executions": {
                    name: self._retarget_execution(execution, staging).to_dict()
                    for name, execution in sorted(executions.items())
                },
                "validation_order": [
                    "all_pf_prefix_outputs",
                    "all_count_mle_snapshots",
                    "all_directives_and_receipts",
                    "final_hybrid_pf",
                    "final_cold_count_mle",
                    "final_cold_spectral_mle",
                    "observation_use_ledger",
                    "all_planning_recommendations",
                    "hybrid_result_contract",
                ],
                "truth_isolation": {
                    "truth_path_present_in_config": False,
                    "truth_opened_by_controller": False,
                    "truth_passed_to_estimators": False,
                    "evaluation_allowed_only_after_this_manifest": True,
                },
                "safety": {
                    "direct_mle_objective_reweight": False,
                    "hard_prune": False,
                    "external_cardinality_change": False,
                    "feedback_mechanism": "fixed_cardinality_target_preserving_mh",
                    "future_corroboration_consumed": future_corroboration_count > 0,
                    "future_corroboration_event_count": future_corroboration_count,
                    "algorithmic_planning_recommendation_count": len(planning_recommendations),
                    "planner_recommendations_authorize_actuation": False,
                    "live_robot_actuation_performed": False,
                },
            }
            write_json_atomic(staging / "hybrid_run_manifest.json", manifest)
            os.replace(staging, target)
            return target
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _build_adapters(
        self, pins: Mapping[str, EstimatorPin]
    ) -> tuple[
        HybridPFCLIAdapter,
        MLECLIAdapter,
        WarmMLECLIAdapter,
        FutureScoreCLIAdapter,
        HybridPlanningCLIAdapter,
    ]:
        config = self.config
        pf_repository = self._repository_path(config.pf_adapter, pins["particle_filter"])
        mle_repository = self._repository_path(config.mle_adapter, pins["surface_mle"])
        pf_policy = {**config.pf_adapter, "repository_path": pf_repository.as_posix()}
        mle_policy = {**config.mle_adapter, "repository_path": mle_repository.as_posix()}
        pf_settings = settings_from_dict(
            pf_policy,
            default_repository=pf_repository,
            default_command=DEFAULT_HYBRID_PF_COMMAND,
        )
        cold_settings = settings_from_dict(
            mle_policy,
            default_repository=mle_repository,
            default_command=DEFAULT_MLE_COMMAND,
        )
        warm_settings = settings_from_dict(
            mle_policy,
            default_repository=mle_repository,
            default_command=DEFAULT_WARM_MLE_COMMAND,
        )
        future_score_settings = settings_from_dict(
            mle_policy,
            default_repository=mle_repository,
            default_command=DEFAULT_FUTURE_SCORE_COMMAND,
        )
        planning_settings = settings_from_dict(
            pf_policy,
            default_repository=pf_repository,
            default_command=DEFAULT_HYBRID_PLANNING_COMMAND,
        )
        return (
            HybridPFCLIAdapter(pins["particle_filter"], pf_settings),
            MLECLIAdapter(pins["surface_mle"], cold_settings),
            WarmMLECLIAdapter(pins["surface_mle"], warm_settings),
            FutureScoreCLIAdapter(pins["surface_mle"], future_score_settings),
            HybridPlanningCLIAdapter(pins["particle_filter"], planning_settings),
        )

    def _repository_path(self, adapter: Mapping[str, object], pin: EstimatorPin) -> Path:
        configured = adapter.get("repository_path")
        if configured is not None:
            supplied = Path(str(configured))
            return (
                (self.config.source_path.parent / supplied).resolve()
                if not supplied.is_absolute()
                else supplied.resolve()
            )
        if pin.local_path_hint is None:
            raise ContractError(f"No local repository path for {pin.name}.")
        hint = Path(pin.local_path_hint)
        return (
            (self.config.pin_registry_path.parent / hint).resolve()
            if not hint.is_absolute()
            else hint.resolve()
        )

    def _run_planning_recommendation(
        self,
        *,
        log: MeasurementLogInfo,
        station_id: int,
        cutoff_step: int,
        record_count: int,
        template: Mapping[str, object],
        queue: VerificationQueue,
        directives: Sequence[PFDirective],
        adapter: HybridPlanningCLIAdapter,
        staging: Path,
        executions: dict[str, AdapterExecution],
    ) -> HybridPlanningRecommendationInfo:
        """Build and execute one collision-attested, non-actuating DSS-PP request."""
        index = record_count - 1
        if int(log.step_ids[index]) != cutoff_step or int(log.station_ids[index]) != station_id:
            raise DataReuseError("Planning request cutoff differs from the current PF prefix.")
        external_modes = self._planner_external_modes(queue, directives)

        request_body = {key: value for key, value in template.items() if key != "data_cutoff_step"}
        request_body.update(
            {
                "schema_version": 1,
                "source_run_id": str(log.manifest["run_id"]),
                "data_cutoff_step": cutoff_step,
                "data_cutoff_station": station_id,
                "covered_records_sha256": measurement_records_sha256(
                    log, record_count=record_count
                ),
                "pf_resolved_config_sha256": (
                    self.config.expected_resolved_config_sha256["pf_strict"]
                ),
                "current_pose_xyz": np.asarray(
                    log.arrays["detector_pose_xyz"][index], dtype=float
                ).tolist(),
                "current_pair_id": int(log.arrays["fe_orientation_index"][index]) * 8
                + int(log.arrays["pb_orientation_index"][index]),
                "visited_poses_xyz": np.asarray(
                    log.arrays["detector_pose_xyz"][:record_count], dtype=float
                ).tolist(),
                "external_modes": external_modes,
            }
        )
        request_identity = {
            key: value for key, value in request_body.items() if key != "request_id"
        }
        request_body["request_id"] = (
            "hybrid-planning-" + sha256_bytes(canonical_json_bytes(request_identity))[:20]
        )
        request_path = write_json_atomic(
            staging / "planning_requests" / f"through-step-{cutoff_step}.json",
            request_body,
        )
        request_info = validate_hybrid_planning_request(request_path)
        schedule_path = self._write_directive_schedule(
            staging / "directive_schedules" / f"planning-step-{cutoff_step}.json",
            directives,
        )
        run_name = f"hybrid_planning_step_{cutoff_step}"
        output = staging / "planning_recommendations" / f"through-step-{cutoff_step}"
        execution = adapter.run(
            log,
            config_path=self.config.pf_config_path,
            planning_request_path=request_path,
            directive_schedule_path=schedule_path,
            output_dir=output,
            execution_dir=staging / "executions" / run_name,
            seed=self.config.random_seed,
            relocation_seed=self.config.relocation_seed,
            profile=self.config.pf_profile,
        )
        executions[run_name] = execution
        recommendation_path = output / "hybrid_planning_recommendation.json"
        recommendation = validate_hybrid_planning_recommendation(
            recommendation_path,
            expected_request=request_info,
        )
        if execution.output_inventory != {
            "hybrid_planning_recommendation.json": sha256_file(recommendation_path)
        }:
            raise ContractError("Hybrid planning output contains unexpected artifacts.")
        return recommendation

    @staticmethod
    def _planner_external_modes(
        queue: VerificationQueue,
        directives: Sequence[PFDirective],
    ) -> list[dict[str, object]]:
        """Build planner-only modes without letting quarantine alter active weights."""
        directive_by_id = {directive.directive_id: directive for directive in directives}
        candidate_states = {
            (candidate.directive_id, candidate.proposal_id): candidate
            for candidate in queue.candidates
        }
        strengths = [
            max(float(proposal.snapshot_strength_cps_1m_metadata), 1e-12)
            for directive in directives
            for proposal in directive.proposals
            if (
                (directive.directive_id, proposal.proposal_id) in candidate_states
                and candidate_states[(directive.directive_id, proposal.proposal_id)].state
                in {CandidateState.PENDING, CandidateState.VERIFIED}
            )
        ]
        strength_total = sum(strengths) or 1.0
        external_modes: list[dict[str, object]] = []
        for key in sorted(candidate_states):
            candidate = candidate_states[key]
            directive = directive_by_id[candidate.directive_id]
            proposal = next(
                item for item in directive.proposals if item.proposal_id == candidate.proposal_id
            )
            kernel = proposal.proposal_kernel
            sigma = (
                [0.25, 0.25, 0.25]
                if kernel is None
                else [float(value) for value in kernel["position_sigma_xyz_m"]]  # type: ignore[index]
            )
            strength = max(float(proposal.snapshot_strength_cps_1m_metadata), 1e-12)
            planner_weight = (
                max(strength / strength_total, 1e-12)
                if candidate.state in {CandidateState.PENDING, CandidateState.VERIFIED}
                else 1e-12
            )
            external_modes.append(
                {
                    "mode_id": proposal.proposal_id,
                    "isotope": proposal.isotope,
                    "position_xyz": list(proposal.candidate_mean_xyz),
                    "strength_cps_1m": strength,
                    "weight": planner_weight,
                    "spread_m": max(sigma),
                    "verification_state": candidate.state.value,
                    "source_snapshot_id": directive.snapshot.snapshot_id,
                }
            )
        return external_modes

    @staticmethod
    def _pf_domain_candidate_mean(
        cluster: SnapshotCluster,
        log: MeasurementLogInfo,
    ) -> tuple[float, float, float] | None:
        """Clip numerical surface-boundary noise and reject true domain mismatches."""
        environment = log.manifest.get("environment")
        if not isinstance(environment, dict):
            raise ContractError("MeasurementLog lacks environment bounds for PF proposals.")
        upper = np.asarray(
            [environment.get("size_x"), environment.get("size_y"), environment.get("size_z")],
            dtype=float,
        )
        position = np.asarray(cluster.centroid_xyz, dtype=float)
        if upper.shape != (3,) or not np.all(np.isfinite(upper)) or np.any(upper <= 0):
            raise ContractError("MeasurementLog environment bounds are invalid.")
        tolerance = 1e-9 * np.maximum(1.0, upper)
        if np.any(position < -tolerance) or np.any(position > upper + tolerance):
            return None
        clipped = np.clip(position, 0.0, upper)
        return tuple(float(value) for value in clipped)

    @staticmethod
    def _has_scoreable_candidates(
        queue: VerificationQueue,
        *,
        receipt_infos: Mapping[str, PFDirectiveReceiptInfo],
        cutoff_step: int,
    ) -> bool:
        """Return whether an applied directive has pending post-cutoff evidence."""
        for candidate in queue.candidates:
            receipt = receipt_infos.get(candidate.directive_id)
            if (
                candidate.state is CandidateState.PENDING
                and candidate.data_cutoff_step < cutoff_step
                and receipt is not None
                and receipt.payload["status"] == "applied"
            ):
                return True
        return False

    def _score_future_candidates(
        self,
        *,
        current_prefix: MeasurementLogInfo,
        current_covered_records_sha256: str,
        current_cutoff_step: int,
        adapter: FutureScoreCLIAdapter,
        pin: EstimatorPin,
        queue: VerificationQueue,
        ledger: ObservationUseLedger,
        directives: Sequence[PFDirective],
        receipt_infos: Mapping[str, PFDirectiveReceiptInfo],
        snapshot_infos: Sequence[MLESnapshotInfo],
        snapshot_paths: Mapping[str, Path],
        snapshot_results: Mapping[str, MLEResultInfo],
        proposal_bindings: Mapping[tuple[str, str], tuple[str, str]],
        staging: Path,
        executions: dict[str, AdapterExecution],
    ) -> tuple[int, tuple[FutureCandidateScoreInfo, ...]]:
        """Consume each unseen frozen-snapshot future score at most once."""
        info_by_snapshot = {str(info.payload["snapshot_id"]): info for info in snapshot_infos}
        directive_by_id = {directive.directive_id: directive for directive in directives}
        scoreable_snapshots: set[str] = set()
        for candidate in queue.candidates:
            receipt = receipt_infos.get(candidate.directive_id)
            directive = directive_by_id.get(candidate.directive_id)
            if (
                candidate.state is CandidateState.PENDING
                and candidate.data_cutoff_step < current_cutoff_step
                and receipt is not None
                and receipt.payload["status"] == "applied"
                and directive is not None
            ):
                scoreable_snapshots.add(directive.snapshot.snapshot_id)

        consumed = 0
        score_infos: list[FutureCandidateScoreInfo] = []
        for snapshot_id in sorted(
            scoreable_snapshots,
            key=lambda value: info_by_snapshot[value].cutoff_step,
        ):
            snapshot_info = info_by_snapshot[snapshot_id]
            snapshot_result = snapshot_results[snapshot_id]
            run_name = f"future_score_{snapshot_id}_through_step_{current_cutoff_step}"
            output = staging / "future_scores" / snapshot_id / f"through-step-{current_cutoff_step}"
            execution = adapter.run(
                current_prefix,
                config_path=self.config.mle_count_config_path,
                snapshot_estimate_dir=snapshot_result.root,
                snapshot_path=snapshot_paths[snapshot_id],
                output_dir=output,
                execution_dir=staging / "executions" / run_name,
            )
            executions[run_name] = execution
            score_path = output / "future_candidate_scores.json"
            score_info = validate_future_candidate_score(
                score_path,
                expected_snapshot=snapshot_info,
                expected_snapshot_mle_result=snapshot_result,
                expected_current_log=current_prefix,
                expected_current_covered_records_sha256=(current_covered_records_sha256),
            )
            if execution.output_inventory != {
                "future_candidate_scores.json": sha256_file(score_path)
            }:
                raise ContractError("Future-score output inventory contains unexpected artifacts.")
            if execution.output_sha256 != sha256_bytes(
                canonical_json_bytes(dict(execution.output_inventory))
            ):
                raise ContractError("Future-score execution inventory digest is invalid.")
            score_infos.append(score_info)

            rows_by_candidate = score_info.payload["candidates"]
            assert isinstance(rows_by_candidate, list)
            for scored in rows_by_candidate:
                assert isinstance(scored, dict)
                binding = proposal_bindings.get((snapshot_id, str(scored["snapshot_candidate_id"])))
                if binding is None:
                    continue
                directive_id, proposal_id = binding
                receipt = receipt_infos.get(directive_id)
                if receipt is None or receipt.payload["status"] != "applied":
                    continue
                per_step = scored["future_step_scores"]
                assert isinstance(per_step, list)
                for row in per_step:
                    assert isinstance(row, dict)
                    current = {
                        (item.directive_id, item.proposal_id): item for item in queue.candidates
                    }[(directive_id, proposal_id)]
                    if current.state is not CandidateState.PENDING:
                        break
                    step_id = int(row["step_id"])
                    if step_id in current.future_step_ids:
                        continue
                    updated = queue.corroborate(
                        directive_id=directive_id,
                        proposal_id=proposal_id,
                        step_id=step_id,
                        log_predictive_likelihood_ratio=float(
                            row["log_predictive_likelihood_ratio"]
                        ),
                    )
                    ledger.record_corroboration(
                        directive_id=directive_id,
                        proposal_id=proposal_id,
                        snapshot_id=snapshot_id,
                        snapshot_candidate_id=str(scored["snapshot_candidate_id"]),
                        step_id=step_id,
                        station_id=int(row["station_id"]),
                        log_predictive_likelihood_ratio=float(
                            row["log_predictive_likelihood_ratio"]
                        ),
                        future_score_sha256=score_info.score_sha256,
                        current_covered_records_sha256=(current_covered_records_sha256),
                        state=updated.state.value,
                    )
                    consumed += 1
        return consumed, tuple(score_infos)

    @staticmethod
    def _validate_explicit_boundaries(
        log: MeasurementLogInfo, schedule: StationBoundarySchedule
    ) -> None:
        declared = dict(schedule.station_end_steps)
        if set(log.station_ids) != set(declared):
            raise DataReuseError("Station schedule does not exactly cover MeasurementLog stations.")
        for index, (step, station) in enumerate(zip(log.step_ids, log.station_ids, strict=True)):
            terminal = declared[station]
            if step > terminal:
                raise DataReuseError("MeasurementLog station continues after its declared end.")
            if step == terminal and index + 1 < log.record_count:
                if log.station_ids[index + 1] == station:
                    raise DataReuseError("Declared terminal step is not the end of its station.")
        for station, terminal in schedule.station_end_steps:
            matching = [
                step
                for step, row_station in zip(log.step_ids, log.station_ids, strict=True)
                if row_station == station
            ]
            if not matching or matching[-1] != terminal:
                raise DataReuseError("Station schedule terminal step differs from the log.")
        if schedule.station_end_steps[-1][1] != log.step_ids[-1]:
            raise DataReuseError("Station schedule must cover the complete finalized log.")

    @staticmethod
    def _write_directive_schedule(path: Path, directives: Sequence[PFDirective]) -> Path:
        return write_json_atomic(
            path,
            {"schema_version": 1, "directives": [item.to_dict() for item in directives]},
        )

    def _validate_pf_run(
        self,
        log: MeasurementLogInfo,
        pin: EstimatorPin,
        *,
        output: Path,
        execution: AdapterExecution,
        record_count: int,
    ) -> PFResultInfo:
        result = validate_pf_result(
            output,
            expected_variant=self.config.pf_profile,
            expected_isotopes=log.isotopes,
            expected_log_sha256=log.measurement_log_sha256,
            expected_commit=pin.revision,
            expected_config_sha256=sha256_file(self.config.pf_config_path),
            expected_resolved_config_sha256=(
                self.config.expected_resolved_config_sha256["pf_strict"]
            ),
            expected_record_count=record_count,
            expected_step_ids=log.step_ids[:record_count],
        )
        if result.result_sha256 != execution.output_sha256:
            raise ContractError("PF result changed between execution and validation.")
        required = (
            "hybrid_diagnostics.json",
            "hybrid_pf_posterior.json",
            "pf_pre_update_predictive.jsonl",
            "external_directive_schedule.json",
            "external_relocation_receipts.json",
        )
        if any(not (output / name).is_file() for name in required):
            raise ContractError("Hybrid PF output lacks required control-boundary artifacts.")
        diagnostics = load_json(output / "hybrid_diagnostics.json")
        expected = {
            "arbitrary_particle_reweighting": False,
            "cardinality_changes_from_external_directives": False,
            "future_records_used_for_proposal_application": False,
            "proposal_role": "target_preserving_mcmc_only",
        }
        if any(diagnostics.get(name) != value for name, value in expected.items()):
            raise ContractError("Hybrid PF diagnostics violate proposal-only safety policy.")
        if (
            diagnostics.get("resolved_config_sha256")
            != (self.config.expected_resolved_config_sha256["pf_strict"])
        ):
            raise ContractError("Hybrid PF diagnostics resolved config hash differs.")
        return result

    def _validate_mle_run(
        self,
        log: MeasurementLogInfo,
        pin: EstimatorPin,
        *,
        output: Path,
        execution: AdapterExecution,
        mode: str,
        expected_resolved_hash: str,
        expected_covered_records_sha256: str,
        expected_fit_kind: str,
        expected_station_boundary_attestation: str,
        expected_previous_mle_result: MLEResultInfo | None,
    ) -> MLEResultInfo:
        config_path = (
            self.config.mle_count_config_path
            if mode == "count"
            else self.config.mle_spectral_config_path
        )
        result = validate_mle_result(
            output,
            expected_mode=mode,
            expected_isotopes=log.isotopes,
            expected_log_sha256=log.measurement_log_sha256,
            expected_commit=pin.revision,
            expected_config_sha256=sha256_file(config_path),
            expected_resolved_config_sha256=expected_resolved_hash,
        )
        if result.result_sha256 != execution.output_sha256:
            raise ContractError("MLE result changed between execution and validation.")
        nested = result.diagnostics.get("diagnostics")
        if not isinstance(nested, dict):
            raise ContractError("MLE result lacks nested diagnostics.")
        lineage = nested.get("causal_lineage")
        if not isinstance(lineage, dict):
            raise ContractError("MLE result lacks causal_lineage.")
        expected = {
            "covered_step_ids": list(log.step_ids),
            "data_cutoff_step": log.step_ids[-1],
            "data_cutoff_station": log.station_ids[-1],
            "record_count": log.record_count,
            "covered_records_sha256": expected_covered_records_sha256,
            "fit_kind": expected_fit_kind,
            "station_boundary_attestation": expected_station_boundary_attestation,
        }
        if any(lineage.get(name) != value for name, value in expected.items()):
            raise DataReuseError("MLE causal lineage differs from its exact controller prefix.")
        if expected_fit_kind == "warm_start_all_history":
            warm = lineage.get("warm_start")
            if not isinstance(warm, dict) or expected_previous_mle_result is None:
                raise DataReuseError("Warm MLE fit lacks prior-prefix lineage.")
            previous_nested = expected_previous_mle_result.diagnostics["diagnostics"]
            assert isinstance(previous_nested, dict)
            previous_lineage = previous_nested.get("causal_lineage")
            previous_provenance = previous_nested.get("provenance")
            if not isinstance(previous_lineage, dict) or not isinstance(previous_provenance, dict):
                raise DataReuseError("Prior MLE result lacks causal lineage.")
            expected_warm = {
                "report_sha256": expected_previous_mle_result.result_sha256,
                "estimate_sha256": sha256_file(
                    expected_previous_mle_result.root / "mle_estimate.npz"
                ),
                "diagnostics_sha256": sha256_file(
                    expected_previous_mle_result.root / "mle_diagnostics.json"
                ),
                "data_cutoff_step": previous_lineage["data_cutoff_step"],
                "data_cutoff_station": previous_lineage["data_cutoff_station"],
                "record_count": previous_lineage["record_count"],
                "covered_records_sha256": previous_lineage["covered_records_sha256"],
                "measurement_log_sha256": previous_provenance["measurement_log_sha256"],
            }
            if any(warm.get(name) != value for name, value in expected_warm.items()):
                raise DataReuseError("Warm MLE ancestry differs from the prior snapshot fit.")
        elif lineage.get("warm_start") is not None:
            raise DataReuseError("Cold MLE fit unexpectedly used a warm-start artifact.")
        elif expected_previous_mle_result is not None:
            raise DataReuseError("Cold MLE validation received an unexpected prior result.")
        return result

    def _run_final_mle(
        self,
        log: MeasurementLogInfo,
        pin: EstimatorPin,
        adapter: MLECLIAdapter,
        *,
        mode: str,
        output: Path,
        execution_dir: Path,
    ) -> tuple[MLEResultInfo, AdapterExecution]:
        config_path = (
            self.config.mle_count_config_path
            if mode == "count"
            else self.config.mle_spectral_config_path
        )
        execution = adapter.run(
            log,
            mode=mode,
            config_path=config_path,
            output_dir=output,
            execution_dir=execution_dir,
        )
        result = self._validate_mle_run(
            log,
            pin,
            output=output,
            execution=execution,
            mode=mode,
            expected_resolved_hash=self.config.expected_resolved_config_sha256[f"mle_{mode}"],
            expected_covered_records_sha256=measurement_records_sha256(log),
            expected_fit_kind="cold_start_all_history",
            expected_station_boundary_attestation="finalized_measurement_log",
            expected_previous_mle_result=None,
        )
        return result, execution

    @staticmethod
    def _load_predictive_rows(
        output: Path, log: MeasurementLogInfo, record_count: int
    ) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for line_number, line in enumerate(
            (output / "pf_pre_update_predictive.jsonl").read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"Invalid PF predictive JSONL row {line_number}.") from exc
            if not isinstance(payload, dict):
                raise ContractError("PF predictive rows must be JSON objects.")
            rows.append(payload)
        if len(rows) != record_count:
            raise ContractError("PF predictive row count differs from replay prefix.")
        for index, row in enumerate(rows):
            if (
                row.get("record_index") != index
                or row.get("step_id") != log.step_ids[index]
                or row.get("station_id") != log.station_ids[index]
                or row.get("computed_before_observation_update") is not True
                or row.get("uses_observed_isotope_counts") is not False
            ):
                raise DataReuseError("PF predictive row is not bound to its causal record.")
        return tuple(rows)

    @staticmethod
    def _predictive_signal(
        log: MeasurementLogInfo,
        row: Mapping[str, object],
        *,
        record_index: int,
        boundaries: StationBoundarySchedule,
        monitor: PredictiveMonitor,
    ):
        arrays = log.arrays
        isotopes = log.isotopes
        if not bool(arrays["isotope_counts_record_present"][record_index]):
            raise ContractError("Hybrid scheduling requires causal isotope counts per record.")
        observed = {
            isotope: float(arrays["isotope_counts"][record_index, isotope_index])
            for isotope_index, isotope in enumerate(isotopes)
            if bool(arrays["isotope_counts_present"][record_index, isotope_index])
        }
        raw_predictions = row.get("isotopes")
        if not isinstance(raw_predictions, dict):
            raise ContractError("PF predictive row lacks isotope predictions.")
        predicted: dict[str, float] = {}
        for isotope in observed:
            item = raw_predictions.get(isotope)
            if not isinstance(item, dict) or "mean_counts" not in item:
                raise ContractError(f"PF prediction lacks isotope {isotope}.")
            predicted[isotope] = float(item["mean_counts"])
        step = log.step_ids[record_index]
        station = log.station_ids[record_index]
        complete = boundaries.asserts_complete(station_id=station, step_id=step)
        previous_step = -1 if record_index == 0 else log.step_ids[record_index - 1]
        return monitor.record(
            step_id=step,
            station_id=station,
            prediction_data_cutoff_step=previous_step,
            station_complete_marker=complete,
            observed_counts=observed,
            predicted_counts=predicted,
        )

    @staticmethod
    def _active_pf_isotopes(result: PFResultInfo) -> set[str]:
        isotopes = result.posterior.get("isotopes")
        if not isinstance(isotopes, dict):
            raise ContractError("PF posterior lacks isotope payload.")
        active: set[str] = set()
        for isotope, raw in isotopes.items():
            if not isinstance(raw, dict):
                raise ContractError("PF isotope posterior must be an object.")
            distribution = raw.get("cardinality_distribution")
            if not isinstance(distribution, dict):
                raise ContractError("PF isotope posterior lacks cardinality distribution.")
            mass = sum(float(value) for key, value in distribution.items() if int(key) > 0)
            if mass > 0:
                active.add(str(isotope))
        return active

    @staticmethod
    def _build_snapshot(
        *,
        trigger: HybridTrigger,
        prefix: Any,
        result: MLEResultInfo,
        previous_snapshot: MLESnapshot | None,
        previous_result: MLEResultInfo | None,
        estimator_commit: str,
    ) -> MLESnapshot:
        clusters: list[SnapshotCluster] = []
        for raw in result.hotspot_clusters:
            identity = {
                "covered_records_sha256": prefix.covered_records_sha256,
                "cluster_id": int(raw["cluster_id"]),
                "isotope": str(raw["isotope"]),
                "centroid_xyz": list(raw["centroid_xyz"]),
                "patch_ids": list(raw["patch_ids"]),
            }
            candidate_id = f"candidate-{sha256_bytes(canonical_json_bytes(identity))[:20]}"
            clusters.append(
                SnapshotCluster(
                    snapshot_candidate_id=candidate_id,
                    cluster_id=int(raw["cluster_id"]),
                    isotope=str(raw["isotope"]),
                    centroid_xyz=tuple(float(value) for value in raw["centroid_xyz"]),
                    integrated_strength_cps_1m=float(raw["integrated_strength_cps_1m"]),
                    surface_kinds=tuple(str(value) for value in raw["surface_kinds"]),
                    patch_ids=tuple(int(value) for value in raw["patch_ids"]),
                )
            )
        predicted = np.asarray(result.arrays.get("predicted_isotope_counts"), dtype=float)
        isotope_names = tuple(str(value) for value in result.arrays["isotope_names"])
        expected_shape = (len(prefix.covered_step_ids), len(isotope_names))
        if predicted.shape != expected_shape or not np.all(np.isfinite(predicted)):
            raise ContractError("MLE predicted isotope counts do not cover the exact prefix.")
        predictions = tuple(
            SnapshotPrediction(
                step_id=step,
                isotope_counts={
                    isotope: float(predicted[row_index, isotope_index])
                    for isotope_index, isotope in enumerate(isotope_names)
                },
            )
            for row_index, step in enumerate(prefix.covered_step_ids)
        )
        nested = result.diagnostics["diagnostics"]
        assert isinstance(nested, dict)
        mle_provenance = nested.get("provenance")
        if not isinstance(mle_provenance, dict):
            raise ContractError("Validated count MLE lacks snapshot provenance.")
        if mle_provenance.get("estimator_commit") != estimator_commit:
            raise ContractError("Snapshot provider commit differs from the pinned MLE.")
        return MLESnapshot.create(
            trigger=trigger,
            prefix=prefix,
            estimator_variant="count",
            mle_result_sha256=result.result_sha256,
            clusters=clusters,
            predictions=predictions,
            fit_diagnostics={
                "converged": bool(result.diagnostics["converged"]),
                "objective_value": float(result.diagnostics["objective_value"]),
                "poisson_deviance": float(result.diagnostics["poisson_deviance"]),
                "causal_lineage": nested["causal_lineage"],
            },
            provenance={
                **mle_provenance,
                "snapshot_provider": "validated_count_mle_result",
                "source_mle_result_sha256": result.result_sha256,
            },
            warm_start_snapshot_id=(
                None if previous_snapshot is None else previous_snapshot.snapshot_id
            ),
            warm_start_mle_result_sha256=(
                None if previous_result is None else previous_result.result_sha256
            ),
        )

    @staticmethod
    def _receipt_from_payload(
        payload: Mapping[str, object], directive: PFDirective
    ) -> PFDirectiveReceipt:
        raw_outcomes = payload["candidate_outcomes"]
        assert isinstance(raw_outcomes, list)
        outcomes = tuple(
            CandidateOutcome(
                proposal_id=str(item["proposal_id"]),
                outcome=str(item["outcome"]),
                mh_attempt_count=int(item["mh_attempt_count"]),
                mh_accepted_count=int(item["mh_accepted_count"]),
                mh_rejected_count=int(item["mh_rejected_count"]),
                not_sampled_count=int(item["not_sampled_count"]),
                eligible_particle_count=int(item["eligible_particle_count"]),
                mh_log_acceptance_ratio=(
                    None
                    if item["mh_log_acceptance_ratio"] is None
                    else float(item["mh_log_acceptance_ratio"])
                ),
                mh_log_uniform_draw=(
                    None
                    if item["mh_log_uniform_draw"] is None
                    else float(item["mh_log_uniform_draw"])
                ),
            )
            for item in raw_outcomes
        )
        provenance = payload["provenance"]
        assert isinstance(provenance, dict)
        return PFDirectiveReceipt(
            receipt_id=str(payload["receipt_id"]),
            directive=directive,
            consumer_variant=str(payload["consumer_variant"]),
            status=str(payload["status"]),
            pf_state_sha256_before=str(payload["pf_state_sha256_before"]),
            pf_state_sha256_after=str(payload["pf_state_sha256_after"]),
            outcomes=outcomes,
            provenance=MappingProxyType(dict(provenance)),
        )

    def _ingest_receipts(
        self,
        output: Path,
        *,
        directives: Mapping[str, PFDirective],
        directive_infos: Mapping[str, PFDirectiveInfo],
        receipt_hashes: dict[str, str],
        receipt_infos: dict[str, PFDirectiveReceiptInfo],
        ledger: ObservationUseLedger,
    ) -> None:
        receipt_dir = output / "pf_directive_receipts"
        if not receipt_dir.is_dir():
            raise ContractError("Hybrid PF output lacks receipt directory.")
        for path in sorted(receipt_dir.glob("*.json")):
            payload = load_json(path)
            directive_id = str(payload.get("directive_id", ""))
            if directive_id not in directives or directive_id not in directive_infos:
                raise DataReuseError("PF emitted a receipt for an unknown directive.")
            info = validate_pf_directive_receipt(
                path, expected_directive=directive_infos[directive_id]
            )
            previous = receipt_hashes.get(directive_id)
            if previous is not None:
                if previous != info.receipt_sha256:
                    raise DataReuseError("Causal PF rerun changed an earlier directive receipt.")
                continue
            receipt = self._receipt_from_payload(info.payload, directives[directive_id])
            if receipt.to_dict() != dict(info.payload):
                raise ContractError("Typed PF receipt differs from validated PF artifact.")
            ledger.record_receipt(receipt)
            receipt_hashes[directive_id] = info.receipt_sha256
            receipt_infos[directive_id] = info

    @staticmethod
    def _retarget_execution(execution: AdapterExecution, staging: Path) -> AdapterExecution:
        def relative(value: str) -> str:
            try:
                return Path(value).resolve().relative_to(staging.resolve()).as_posix()
            except ValueError as exc:
                raise ContractError(f"Execution log escaped hybrid staging: {value}") from exc

        return replace(
            execution,
            stdout_path=relative(execution.stdout_path),
            stderr_path=relative(execution.stderr_path),
        )


__all__ = ["HybridController"]
