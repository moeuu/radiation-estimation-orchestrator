"""Checkpoint-only 3-D measurement planning over runtime-attested candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orchestrator.contracts import (
    HybridPlanningRecommendationInfo,
    HybridPlanningRequestInfo,
    PFCheckpointInfo,
    validate_hybrid_planning_recommendation,
)
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes, write_json_atomic

from .artifacts import load_checkpoint_state, repository_commit
from .context import load_estimator_context
from .forward import predict_particle_spectra_for_actions
from .pf import ParticleFilter, ParticleFilterConfig


def _external_hypotheses(
    context,
    modes: list[dict[str, object]],
    slot_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    included = [mode for mode in modes if mode["verification_state"] != "quarantined"]
    if not included:
        return (
            np.zeros((0, len(context.isotopes), slot_count), dtype=np.int64),
            np.zeros((0, len(context.isotopes), slot_count), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            [],
        )
    charts = np.full((len(included), len(context.isotopes), slot_count), -1, dtype=np.int64)
    strengths = np.zeros_like(charts, dtype=np.float64)
    weights = np.zeros(len(included), dtype=np.float64)
    centers = np.asarray(context.surface_geometry.centers_xyz, dtype=np.float64)
    identifiers: list[str] = []
    for index, mode in enumerate(included):
        isotope = str(mode["isotope"])
        if isotope not in context.isotopes:
            raise ContractError("Planner external mode isotope is absent from the runtime model.")
        isotope_index = context.isotopes.index(isotope)
        position = np.asarray(mode["position_xyz"], dtype=np.float64)
        chart = int(np.argmin(np.linalg.norm(centers - position, axis=1)))
        charts[index, isotope_index, 0] = chart
        strengths[index, isotope_index, 0] = float(mode["strength_cps_1m"])
        weights[index] = float(mode["weight"])
        identifiers.append(str(mode["mode_id"]))
    weights /= float(np.sum(weights))
    return charts, strengths, weights, identifiers


def plan_from_checkpoint(
    measurement_log: str | Path,
    *,
    config_path: str | Path,
    checkpoint: PFCheckpointInfo,
    request: HybridPlanningRequestInfo,
    output_path: str | Path,
) -> HybridPlanningRecommendationInfo:
    """Rank attested XYZ/pair actions without mutating checkpoint state."""
    config = ParticleFilterConfig.from_path(config_path)
    context = load_estimator_context(
        measurement_log,
        patch_edge_m=config.patch_edge_m,
        use_gpu=config.use_gpu,
    )
    if checkpoint.payload["prefix_measurement_log_sha256"] != (
        context.measurement_log.measurement_log_sha256
    ):
        raise DataReuseError("Planning checkpoint and MeasurementLog prefix differ.")
    if int(request.payload["data_cutoff_step"]) != checkpoint.cutoff_step:
        raise DataReuseError("Planning request and checkpoint cutoffs differ.")
    state = load_checkpoint_state(checkpoint)
    pf = ParticleFilter(
        context,
        config,
        random_seed=int(checkpoint.payload["random_seed"]),
        state=state,
    )
    state_hash = str(checkpoint.payload["state_artifact_sha256"])
    poses = np.asarray(request.payload["candidate_poses_xyz"], dtype=np.float64)
    raw_pairs = request.payload.get("allowed_pair_ids")
    if isinstance(raw_pairs, list) and raw_pairs:
        pair_ids = tuple(sorted({int(value) for value in raw_pairs}))
    else:
        configured = request.payload["dsspp_config"].get("allowed_pair_ids")  # type: ignore[union-attr]
        if isinstance(configured, list) and configured:
            pair_ids = tuple(sorted({int(value) for value in configured}))
        elif request.payload["current_pair_id"] is not None:
            pair_ids = (int(request.payload["current_pair_id"]),)
        else:
            pair_ids = tuple(range(64))
    if any(pair < 0 or pair >= 64 for pair in pair_ids):
        raise ContractError("Planning shield pair IDs must lie in [0, 63].")
    pose_index = np.repeat(np.arange(poses.shape[0], dtype=np.int64), len(pair_ids))
    action_pairs = np.tile(np.asarray(pair_ids, dtype=np.int64), poses.shape[0])
    action_poses = poses[pose_index]
    dwell = float(request.payload["dsspp_config"].get("planning_live_time_s", 1.0))  # type: ignore[union-attr]
    pf_prediction = predict_particle_spectra_for_actions(
        context,
        chart_ids=state.chart_ids,
        strengths_cps_1m=state.strengths_cps_1m,
        detector_positions_xyz=action_poses,
        fe_orientation_indices=action_pairs // 8,
        pb_orientation_indices=action_pairs % 8,
        live_times_s=np.full(action_pairs.size, dwell, dtype=np.float64),
    ).mean_spectra
    hypothesis_prediction = pf_prediction
    hypothesis_weights = pf.weights
    raw_modes = [dict(value) for value in request.payload["external_modes"]]  # type: ignore[arg-type]
    external_charts, external_strengths, external_weights, included_ids = _external_hypotheses(
        context,
        raw_modes,
        config.max_sources_per_isotope,
    )
    if external_charts.shape[0]:
        external_prediction = predict_particle_spectra_for_actions(
            context,
            chart_ids=external_charts,
            strengths_cps_1m=external_strengths,
            detector_positions_xyz=action_poses,
            fe_orientation_indices=action_pairs // 8,
            pb_orientation_indices=action_pairs % 8,
            live_times_s=np.full(action_pairs.size, dwell, dtype=np.float64),
        ).mean_spectra
        external_mass = min(
            float(request.payload["dsspp_config"].get("external_mode_mass", 0.1)),  # type: ignore[union-attr]
            0.49,
        )
        hypothesis_weights = np.concatenate(
            ((1.0 - external_mass) * pf.weights, external_mass * external_weights)
        )
        hypothesis_prediction = np.concatenate((pf_prediction, external_prediction), axis=0)
    mean = np.einsum(
        "n,nab->ab", hypothesis_weights, hypothesis_prediction, optimize=True
    )
    information = np.einsum(
        "n,nab->a",
        hypothesis_weights,
        np.square(hypothesis_prediction - mean[None, :, :])
        / np.maximum(mean[None, :, :], 1.0),
        optimize=True,
    )
    count_utility = np.log1p(np.sum(mean, axis=1))
    current = np.asarray(request.payload["current_pose_xyz"], dtype=np.float64)
    travel = np.linalg.norm(action_poses - current[None, :], axis=1)
    visited = np.asarray(request.payload["visited_poses_xyz"], dtype=np.float64)
    if visited.size:
        vertical = np.min(np.abs(action_poses[:, None, 2] - visited[None, :, 2]), axis=1)
    else:
        vertical = np.zeros(action_pairs.size, dtype=np.float64)
    dsspp = request.payload["dsspp_config"]
    assert isinstance(dsspp, dict)
    score = (
        float(dsspp.get("information_weight", 1.0)) * information
        + float(dsspp.get("count_weight", 0.05)) * count_utility
        + float(dsspp.get("vertical_diversity_weight", 0.2)) * vertical
        - float(dsspp.get("travel_cost_weight", 0.05)) * travel
    )
    best = int(np.argmax(score))
    selected_pose_index = int(pose_index[best])
    selected_pair = int(action_pairs[best])
    excluded_ids = [
        str(mode["mode_id"])
        for mode in raw_modes
        if mode["verification_state"] == "quarantined"
    ]
    selected = poses[selected_pose_index].tolist()
    identity = {
        "request": request.request_sha256,
        "state": state_hash,
        "pose": selected_pose_index,
        "pair": selected_pair,
    }
    payload = {
        "schema_version": 1,
        "recommendation_id": (
            f"recommendation-{sha256_bytes(canonical_json_bytes(identity))[:20]}"
        ),
        "recommendation_kind": "algorithmic_dsspp_action_recommendation",
        "algorithmic_recommendation_only": True,
        "robot_actuation_authorized": False,
        "selected_action": {
            "candidate_index": selected_pose_index,
            "dsspp_filtered_pose_index": selected_pose_index,
            "pose_xyz": selected,
            "detector_height_m": float(selected[2]),
            "shield_program": {
                "name": f"pair-{selected_pair}",
                "kind": "fixed_pair",
                "pair_ids": [selected_pair],
            },
            "score": float(score[best]),
        },
        "sequence": [],
        "diagnostics": {
            "action_count": int(score.size),
            "information_score": float(information[best]),
            "count_utility": float(count_utility[best]),
            "vertical_diversity_m": float(vertical[best]),
            "travel_distance_m": float(travel[best]),
            "score_range": [float(np.min(score)), float(np.max(score))],
        },
        "belief": {
            "planner_belief_sources": ["joint_pf_particles", "verified_mle_modes"],
            "external_modes_included": [
                mode for mode in raw_modes if mode["verification_state"] != "quarantined"
            ],
            "external_modes_quarantined_excluded": [
                mode for mode in raw_modes if mode["verification_state"] == "quarantined"
            ],
            "included_external_mode_ids": included_ids,
            "excluded_quarantined_mode_ids": excluded_ids,
            "excluded_quarantined_mode_count": len(excluded_ids),
            "external_strengths_and_weights_are_planner_metadata_only": True,
        },
        "candidate_attestation": request.payload["candidate_attestation"],
        "causal_boundary": {
            "source_run_id": request.payload["source_run_id"],
            "data_cutoff_step": request.payload["data_cutoff_step"],
            "data_cutoff_station": request.payload["data_cutoff_station"],
            "covered_records_sha256": request.payload["covered_records_sha256"],
            "pf_resolved_config_sha256": request.payload["pf_resolved_config_sha256"],
            "causal_identity_uses_record_prefix_only": True,
        },
        "external_relocation": {"performed": False},
        "pf_state_integrity": {
            "state_sha256_before_planning": state_hash,
            "state_sha256_after_planning": state_hash,
            "pf_particles_or_weights_mutated_by_planning": False,
            "external_modes_mutated_pf": False,
        },
        "provenance": {
            "estimator_repository": "radiation-estimation-orchestrator",
            "estimator_commit": repository_commit(),
            "pf_resolved_config_sha256": request.payload["pf_resolved_config_sha256"],
            "causal_planning_request_sha256": request.request_sha256,
        },
    }
    path = write_json_atomic(output_path, payload)
    return validate_hybrid_planning_recommendation(path, expected_request=request)


__all__ = ["plan_from_checkpoint"]
