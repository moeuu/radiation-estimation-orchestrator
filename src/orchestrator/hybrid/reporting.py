"""Canonical final reporting for the causal PF+MLE hybrid v1 milestone."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from orchestrator.contracts import (
    FutureCandidateScoreInfo,
    HybridLedgerSummaryInfo,
    HybridPlanningRecommendationInfo,
    HybridResultInfo,
    MeasurementLogInfo,
    MLEResultInfo,
    MLESnapshotInfo,
    PFDirectiveInfo,
    PFDirectiveReceiptInfo,
    PFResultInfo,
    validate_hybrid_result,
)
from orchestrator.errors import ContractError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes, sha256_file, write_json_atomic


def _sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _mode(value: object) -> str:
    normalized = str(getattr(value, "value", value))
    if normalized not in {"verification_only", "proposal_only_mh"}:
        raise ContractError(f"Unsupported hybrid reporting mode: {normalized!r}.")
    return normalized


def _mle_provenance(result: MLEResultInfo) -> dict[str, object]:
    nested = result.diagnostics["diagnostics"]
    assert isinstance(nested, dict)
    provenance = nested["provenance"]
    assert isinstance(provenance, dict)
    return provenance


def _require_final_mle(
    result: MLEResultInfo,
    *,
    mode: str,
    measurement_log: MeasurementLogInfo,
) -> None:
    if result.mode != mode:
        raise ContractError(f"Final {mode} role received {result.mode!r} MLE output.")
    nested = result.diagnostics["diagnostics"]
    assert isinstance(nested, dict)
    lineage = nested.get("causal_lineage")
    if not isinstance(lineage, dict):
        raise ContractError("Final MLE output lacks causal_lineage evidence.")
    covered = tuple(int(value) for value in lineage.get("covered_step_ids", ()))
    if (
        lineage.get("fit_kind") != "cold_start_all_history"
        or lineage.get("warm_start") is not None
        or covered != measurement_log.step_ids
        or int(lineage.get("record_count", -1)) != measurement_log.record_count
    ):
        raise ContractError("Final MLE output must be a cold fit over the complete MeasurementLog.")
    provenance = _mle_provenance(result)
    if provenance.get("measurement_log_sha256") != measurement_log.measurement_log_sha256:
        raise ContractError("Final MLE output is bound to a different MeasurementLog.")
    if mode == "spectral" and result.diagnostics.get("converged") is not True:
        raise ContractError("The authoritative final spectral MLE must report converged=true.")


def _cluster_report(result: MLEResultInfo) -> list[dict[str, object]]:
    return [
        {
            "cluster_id": int(cluster["cluster_id"]),
            "isotope": str(cluster["isotope"]),
            "centroid_xyz": [float(value) for value in cluster["centroid_xyz"]],  # type: ignore[arg-type]
            "integrated_strength_cps_1m": float(cluster["integrated_strength_cps_1m"]),
            "surface_kinds": [str(value) for value in cluster["surface_kinds"]],  # type: ignore[arg-type]
            "patch_ids": [int(value) for value in cluster["patch_ids"]],  # type: ignore[arg-type]
        }
        for cluster in result.hotspot_clusters
    ]


def _snapshot_refs(snapshots: Sequence[MLESnapshotInfo]) -> list[dict[str, object]]:
    refs: list[dict[str, object]] = []
    for snapshot in sorted(snapshots, key=lambda item: item.cutoff_step):
        warm = snapshot.payload["warm_start"]
        assert isinstance(warm, dict)
        refs.append(
            {
                "snapshot_id": str(snapshot.payload["snapshot_id"]),
                "sha256": snapshot.snapshot_sha256,
                "data_cutoff_step": snapshot.cutoff_step,
                "data_cutoff_station": int(snapshot.payload["data_cutoff_station"]),
                "warm_start_used": bool(warm["used"]),
            }
        )
    return refs


def _directive_refs(directives: Sequence[PFDirectiveInfo]) -> list[dict[str, object]]:
    return [
        {
            "directive_id": str(directive.payload["directive_id"]),
            "sha256": directive.directive_sha256,
            "snapshot_id": str(directive.payload["snapshot_id"]),
            "data_cutoff_step": directive.cutoff_step,
            "directive_kind": str(directive.payload["directive_kind"]),
            "proposal_count": len(directive.payload["proposals"]),  # type: ignore[arg-type]
        }
        for directive in sorted(
            directives,
            key=lambda item: (item.cutoff_step, str(item.payload["directive_id"])),
        )
    ]


def _receipt_refs(receipts: Sequence[PFDirectiveReceiptInfo]) -> list[dict[str, object]]:
    return [
        {
            "receipt_id": str(receipt.payload["receipt_id"]),
            "sha256": receipt.receipt_sha256,
            "directive_id": str(receipt.payload["directive_id"]),
            "status": str(receipt.payload["status"]),
        }
        for receipt in sorted(receipts, key=lambda item: str(item.payload["directive_id"]))
    ]


def _future_score_refs(
    scores: Sequence[FutureCandidateScoreInfo],
) -> list[dict[str, object]]:
    """Project validated frozen-snapshot score artifacts into the final manifest."""
    return [
        {
            "snapshot_id": str(score.payload["snapshot_id"]),
            "sha256": score.score_sha256,
            "future_step_ids": list(score.future_step_ids),
        }
        for score in sorted(
            scores,
            key=lambda item: (
                int(item.payload["snapshot_data_cutoff_step"]),
                item.future_step_ids[-1],
            ),
        )
    ]


def _planning_recommendation_refs(
    recommendations: Sequence[HybridPlanningRecommendationInfo],
) -> list[dict[str, object]]:
    """Project validated recommendation-only DSS-PP artifacts into the report."""
    refs: list[dict[str, object]] = []
    for recommendation in sorted(
        recommendations,
        key=lambda item: int(item.payload["causal_boundary"]["data_cutoff_step"]),  # type: ignore[index]
    ):
        payload = recommendation.payload
        boundary = payload["causal_boundary"]
        provenance = payload["provenance"]
        selected = payload["selected_action"]
        assert isinstance(boundary, dict)
        assert isinstance(provenance, dict)
        assert isinstance(selected, dict)
        refs.append(
            {
                "recommendation_id": str(payload["recommendation_id"]),
                "sha256": recommendation.recommendation_sha256,
                "data_cutoff_step": int(boundary["data_cutoff_step"]),
                "data_cutoff_station": int(boundary["data_cutoff_station"]),
                "causal_planning_request_sha256": str(provenance["causal_planning_request_sha256"]),
                "selected_action": dict(selected),
                "robot_actuation_authorized": bool(payload["robot_actuation_authorized"]),
            }
        )
    return refs


def _ledger_verification_counts(
    ledger: HybridLedgerSummaryInfo,
) -> dict[str, int]:
    """Reconstruct current proposal states from the append-only ledger."""
    states: dict[tuple[str, str], str] = {}
    events = ledger.payload["events"]
    assert isinstance(events, list)
    for event in events:
        assert isinstance(event, dict)
        event_type = str(event["event_type"])
        payload = event["payload"]
        assert isinstance(payload, dict)
        if event_type == "directive_issued":
            directive_id = str(payload["directive_id"])
            for proposal_id in payload["proposal_ids"]:  # type: ignore[union-attr]
                states[(directive_id, str(proposal_id))] = "pending"
        elif event_type == "corroboration":
            key = (str(payload["directive_id"]), str(payload["proposal_id"]))
            if key not in states:
                raise ContractError("Ledger corroboration references an unknown proposal.")
            state = str(payload["candidate_state"])
            if state not in {"pending", "verified", "quarantined"}:
                raise ContractError("Ledger candidate state is invalid.")
            states[key] = state
    return {
        name: sum(state == name for state in states.values())
        for name in ("pending", "verified", "quarantined")
    }


def build_hybrid_result(
    *,
    hybrid_run_id: str,
    hybrid_mode: object,
    measurement_log: MeasurementLogInfo,
    source_measurement_log_sha256: str,
    source_measurement_log_record_count: int,
    station_boundary_schedule_sha256: str,
    final_pf_result: PFResultInfo,
    final_count_mle_result: MLEResultInfo,
    final_spectral_mle_result: MLEResultInfo,
    ledger: HybridLedgerSummaryInfo,
    snapshots: Sequence[MLESnapshotInfo],
    directives: Sequence[PFDirectiveInfo],
    receipts: Sequence[PFDirectiveReceiptInfo],
    future_candidate_scores: Sequence[FutureCandidateScoreInfo],
    planning_recommendations: Sequence[HybridPlanningRecommendationInfo],
    verification_queue_sha256: str,
    verification_counts: Mapping[str, int],
    orchestrator_commit: str,
    hybrid_config_sha256: str,
    pin_registry_sha256: str,
    execution_evidence_sha256: str,
) -> dict[str, object]:
    """Build the deterministic v1 report from already validated estimator artifacts."""
    if not hybrid_run_id or not orchestrator_commit:
        raise ContractError("Hybrid result requires run and orchestrator revision identifiers.")
    mode = _mode(hybrid_mode)
    source_log_hash = _sha256(
        source_measurement_log_sha256,
        label="source_measurement_log_sha256",
    )
    if source_measurement_log_record_count != measurement_log.record_count:
        raise ContractError(
            "Station-marker derivation may not change the MeasurementLog record count."
        )
    schedule_hash = _sha256(
        station_boundary_schedule_sha256,
        label="station_boundary_schedule_sha256",
    )
    _sha256(hybrid_config_sha256, label="hybrid_config_sha256")
    _sha256(pin_registry_sha256, label="pin_registry_sha256")
    queue_hash = _sha256(
        verification_queue_sha256,
        label="verification_queue_sha256",
    )
    _sha256(execution_evidence_sha256, label="execution_evidence_sha256")
    if ledger.payload["source_run_id"] != measurement_log.manifest["run_id"]:
        raise ContractError("Hybrid ledger and MeasurementLog have different source run IDs.")
    if ledger.payload["station_boundary_schedule_sha256"] != schedule_hash:
        raise ContractError("Hybrid ledger uses a different station-boundary schedule.")

    pf_provenance = final_pf_result.posterior["provenance"]
    assert isinstance(pf_provenance, dict)
    if pf_provenance.get("measurement_log_sha256") != measurement_log.measurement_log_sha256:
        raise ContractError("Final PF output is bound to a different MeasurementLog.")
    _require_final_mle(
        final_count_mle_result,
        mode="count",
        measurement_log=measurement_log,
    )
    _require_final_mle(
        final_spectral_mle_result,
        mode="spectral",
        measurement_log=measurement_log,
    )

    snapshot_refs = _snapshot_refs(snapshots)
    directive_refs = _directive_refs(directives)
    receipt_refs = _receipt_refs(receipts)
    future_score_refs = _future_score_refs(future_candidate_scores)
    planning_refs = _planning_recommendation_refs(planning_recommendations)
    counts = {
        name: int(verification_counts.get(name, 0))
        for name in ("pending", "verified", "quarantined")
    }
    if any(value < 0 for value in counts.values()):
        raise ContractError("Hybrid verification counts must be nonnegative.")
    if counts != _ledger_verification_counts(ledger):
        raise ContractError("Hybrid verification counts differ from the ledger state.")
    total = sum(counts.values())
    proposal_total = sum(int(reference["proposal_count"]) for reference in directive_refs)
    if total != proposal_total:
        raise ContractError("Hybrid verification counts must account for every proposal.")
    applied_mh = mode == "proposal_only_mh" and any(
        reference["status"] == "applied" for reference in receipt_refs
    )

    identity = {
        "schema_version": 1,
        "hybrid_run_id": hybrid_run_id,
        "source_run_id": str(measurement_log.manifest["run_id"]),
        "source_measurement_log_sha256": source_log_hash,
        "inference_measurement_log_sha256": measurement_log.measurement_log_sha256,
        "hybrid_mode": mode,
        "final_pf_result_sha256": final_pf_result.result_sha256,
        "final_spectral_mle_result_sha256": final_spectral_mle_result.result_sha256,
        "ledger_sha256": ledger.summary_sha256,
        "verification_queue_sha256": queue_hash,
        "future_candidate_scores": future_score_refs,
        "planning_recommendations": planning_refs,
    }
    result_id = f"hybrid-result-{sha256_bytes(canonical_json_bytes(identity))[:20]}"
    return {
        "schema_version": 1,
        "hybrid_result_id": result_id,
        "hybrid_run_id": hybrid_run_id,
        "source_run_id": str(measurement_log.manifest["run_id"]),
        "milestone": "pf_mle_hybrid_v1",
        "status": "complete",
        "hybrid_mode": mode,
        "measurement_log": {
            "schema_version": int(measurement_log.manifest["schema_version"]),
            "source_measurement_log_sha256": source_log_hash,
            "inference_measurement_log_sha256": measurement_log.measurement_log_sha256,
            "source_record_count": source_measurement_log_record_count,
            "inference_record_count": measurement_log.record_count,
            "derivation_kind": "predeclared_station_boundary_markers_only",
            "final_step_id": measurement_log.step_ids[-1],
            "station_boundary_schedule_sha256": schedule_hash,
        },
        "estimator_roles": {
            "online_pf": {
                "estimator_family": "particle_filter",
                "role": "online_history_uncertainty_and_proposal_consumer",
                "final_estimate_source": "pf_posterior",
                "uses_all_history_batch_fit": False,
            },
            "causal_prefix_mle": {
                "estimator_family": "surface_mle",
                "estimator_variant": "count",
                "role": "proposal_metadata_and_future_verification",
                "fit_scope": "station_complete_observation_prefix",
                "warm_start_semantics": "initialization_only_full_prefix_objective",
            },
            "final_count_diagnostic": {
                "estimator_family": "surface_mle",
                "estimator_variant": "count",
                "role": "diagnostic",
                "fit_scope": "full_measurement_log",
                "initialization": "cold",
                "uses_pf_state": False,
                "uses_pf_candidates": False,
            },
            "final_report": {
                "estimator_family": "surface_mle",
                "estimator_variant": "spectral",
                "role": "authoritative_final_report",
                "fit_scope": "full_measurement_log",
                "initialization": "cold",
                "uses_pf_state": False,
                "uses_pf_candidates": False,
            },
        },
        "artifacts": {
            "final_pf_result_sha256": final_pf_result.result_sha256,
            "final_count_mle_result_sha256": final_count_mle_result.result_sha256,
            "final_spectral_mle_result_sha256": final_spectral_mle_result.result_sha256,
            "hybrid_ledger_summary_sha256": ledger.summary_sha256,
            "verification_queue_sha256": queue_hash,
            "ledger_event_count": int(ledger.payload["event_count"]),
            "ledger_last_event_sha256": str(ledger.payload["last_event_sha256"]),
            "snapshots": snapshot_refs,
            "directives": directive_refs,
            "receipts": receipt_refs,
            "future_candidate_scores": future_score_refs,
            "planning_recommendations": planning_refs,
        },
        "authoritative_report": {
            "estimator_family": "surface_mle",
            "estimator_variant": "spectral",
            "fit_scope": "full_measurement_log",
            "initialization": "cold",
            "candidate_domain": "complete_surface_dictionary",
            "uses_pf_state": False,
            "uses_pf_candidates": False,
            "result_sha256": final_spectral_mle_result.result_sha256,
            "converged": True,
            "objective_value": float(final_spectral_mle_result.diagnostics["objective_value"]),
            "poisson_deviance": float(final_spectral_mle_result.diagnostics["poisson_deviance"]),
            "hotspot_clusters": _cluster_report(final_spectral_mle_result),
        },
        "verification_summary": {
            "total": total,
            **counts,
            "evidence_family": ("frozen_count_snapshot_cluster_log_predictive_ratio"),
            "threshold_quantity": "cumulative_log_predictive_likelihood_ratio",
        },
        "safety": {
            "causal_prefix_mle": True,
            "station_complete_cutoffs": True,
            "once_only_directives": True,
            "future_only_verification": True,
            "all_applied_proposals_target_preserving": True,
            "target_preserving_fixed_cardinality_mh_performed": applied_mh,
            "feedback_changes_cardinality": False,
            "direct_mle_objective_reweight_performed": False,
            "hard_prune_performed": False,
            "final_report_uses_pf_state": False,
            "final_report_uses_pf_candidates": False,
            "pure_pf_baseline_path_preserved": True,
            "pure_mle_baseline_path_preserved": True,
            "planner_recommendations_authorize_actuation": False,
        },
        "truth_isolation": {
            "truth_passed_to_estimator_commands": False,
            "truth_read_during_hybrid_inference": False,
            "all_inference_outputs_validated_before_evaluation": True,
        },
        "limitations": {
            "rj_birth_death_implemented": False,
            "hard_prune_implemented": False,
            "direct_mle_reweight_implemented": False,
            "live_closed_loop_planner_control_implemented": False,
            "algorithmic_planner_recommendation_implemented": True,
            "planner_scope": "offline_replay_recommendation_no_actuation",
            "fixed_cardinality_position_relocation_only": True,
        },
        "provenance": {
            "orchestrator_commit": orchestrator_commit,
            "hybrid_config_sha256": hybrid_config_sha256,
            "pin_registry_sha256": pin_registry_sha256,
            "execution_evidence_sha256": execution_evidence_sha256,
        },
    }


def write_hybrid_result_bundle(
    output_directory: str | Path,
    payload: Mapping[str, object],
    *,
    expected_measurement_log: MeasurementLogInfo | None = None,
    expected_source_measurement_log: MeasurementLogInfo | None = None,
    expected_pf_result: PFResultInfo | None = None,
    expected_final_count_mle_result: MLEResultInfo | None = None,
    expected_final_spectral_mle_result: MLEResultInfo | None = None,
    expected_ledger: HybridLedgerSummaryInfo | None = None,
    expected_snapshots: Sequence[MLESnapshotInfo] | None = None,
    expected_directives: Sequence[PFDirectiveInfo] | None = None,
    expected_receipts: Sequence[PFDirectiveReceiptInfo] | None = None,
    expected_future_candidate_scores: Sequence[FutureCandidateScoreInfo] | None = None,
    expected_planning_recommendations: Sequence[HybridPlanningRecommendationInfo] | None = None,
    expected_verification_queue_sha256: str | None = None,
) -> tuple[HybridResultInfo, Path]:
    """Atomically write, validate, and hash one hybrid result plus sidecar."""
    root = Path(output_directory)
    result_path = write_json_atomic(root / "hybrid_result.json", dict(payload))
    try:
        result = validate_hybrid_result(
            result_path,
            expected_measurement_log=expected_measurement_log,
            expected_source_measurement_log=expected_source_measurement_log,
            expected_pf_result=expected_pf_result,
            expected_final_count_mle_result=expected_final_count_mle_result,
            expected_final_spectral_mle_result=expected_final_spectral_mle_result,
            expected_ledger=expected_ledger,
            expected_snapshots=(None if expected_snapshots is None else tuple(expected_snapshots)),
            expected_directives=(
                None if expected_directives is None else tuple(expected_directives)
            ),
            expected_receipts=(None if expected_receipts is None else tuple(expected_receipts)),
            expected_future_candidate_scores=(
                None
                if expected_future_candidate_scores is None
                else tuple(expected_future_candidate_scores)
            ),
            expected_planning_recommendations=(
                None
                if expected_planning_recommendations is None
                else tuple(expected_planning_recommendations)
            ),
            expected_verification_queue_sha256=expected_verification_queue_sha256,
        )
        digest = sha256_file(result_path)
        if digest != result.result_sha256:
            raise RuntimeError("Hybrid result canonical serialization changed unexpectedly.")
        sidecar = root / "hybrid_result.sha256"
        descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(f"{digest}  hybrid_result.json\n".encode("ascii"))
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            sidecar.unlink(missing_ok=True)
            raise
    except BaseException:
        result_path.unlink(missing_ok=True)
        raise
    return result, sidecar


__all__ = ["build_hybrid_result", "write_hybrid_result_bundle"]
