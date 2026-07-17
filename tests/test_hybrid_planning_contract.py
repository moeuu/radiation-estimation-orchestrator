from __future__ import annotations

import copy
from pathlib import Path

import pytest

from orchestrator.contracts import (
    validate_hybrid_planning_recommendation,
    validate_hybrid_planning_request,
)
from orchestrator.errors import ContractError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes, write_json_atomic

SHA_A = "a" * 64
SHA_B = "b" * 64


def _request() -> dict[str, object]:
    candidates = [[0.8, 0.8, 0.45], [2.0, 4.8, 1.8]]
    return {
        "schema_version": 1,
        "request_id": "planning-request-1",
        "source_run_id": "run-1",
        "data_cutoff_step": 5,
        "data_cutoff_station": 1,
        "covered_records_sha256": SHA_A,
        "pf_resolved_config_sha256": SHA_B,
        "current_pose_xyz": [4.8, 0.9, 1.1],
        "current_pair_id": 59,
        "visited_poses_xyz": [[0.8, 0.8, 0.45], [4.8, 0.9, 1.1]],
        "candidate_poses_xyz": candidates,
        "candidate_attestation": {
            "candidate_poses_sha256": sha256_bytes(canonical_json_bytes(candidates)),
            "workspace_sha256": SHA_A,
            "planning_config_sha256": SHA_B,
            "collision_checked": True,
            "reachability_filtered": True,
        },
        "dsspp_config": {
            "augment_candidates": False,
            "include_runtime_rescue_modes": False,
            "include_global_surface_rescue_modes": False,
        },
        "external_modes": [
            {
                "mode_id": "pending-1",
                "isotope": "Cs-137",
                "position_xyz": [1.0, 2.0, 0.5],
                "strength_cps_1m": 10.0,
                "weight": 1.0,
                "spread_m": 0.2,
                "verification_state": "pending",
                "source_snapshot_id": "snapshot-1",
            },
            {
                "mode_id": "quarantined-1",
                "isotope": "Cs-137",
                "position_xyz": [5.0, 5.0, 2.5],
                "strength_cps_1m": 1000.0,
                "weight": 1e-12,
                "spread_m": 0.2,
                "verification_state": "quarantined",
                "source_snapshot_id": "snapshot-1",
            },
        ],
    }


def _recommendation(request_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "recommendation_id": "recommendation-1",
        "recommendation_kind": "algorithmic_dsspp_action_recommendation",
        "algorithmic_recommendation_only": True,
        "robot_actuation_authorized": False,
        "selected_action": {
            "candidate_index": 1,
            "dsspp_filtered_pose_index": 1,
            "pose_xyz": [2.0, 4.8, 1.8],
            "detector_height_m": 1.8,
            "shield_program": {"name": "one", "kind": "fixed", "pair_ids": [0]},
            "score": 2.5,
        },
        "sequence": [],
        "diagnostics": {},
        "belief": {
            "planner_belief_sources": ["pf_posterior", "external_mode_pending"],
            "external_modes_included": [],
            "external_modes_quarantined_excluded": [],
            "included_external_mode_ids": ["pending-1"],
            "excluded_quarantined_mode_ids": ["quarantined-1"],
            "excluded_quarantined_mode_count": 1,
            "external_strengths_and_weights_are_planner_metadata_only": True,
        },
        "candidate_attestation": _request()["candidate_attestation"],
        "causal_boundary": {
            "source_run_id": "run-1",
            "data_cutoff_step": 5,
            "data_cutoff_station": 1,
            "covered_records_sha256": SHA_A,
            "pf_resolved_config_sha256": SHA_B,
            "causal_identity_uses_record_prefix_only": True,
        },
        "external_relocation": {},
        "pf_state_integrity": {
            "state_sha256_before_planning": SHA_A,
            "state_sha256_after_planning": SHA_A,
            "pf_particles_or_weights_mutated_by_planning": False,
            "external_modes_mutated_pf": False,
        },
        "provenance": {
            "pf_resolved_config_sha256": SHA_B,
            "causal_planning_request_sha256": request_sha256,
        },
    }


def test_planning_contract_binds_exact_request_and_filters_quarantine(
    tmp_path: Path,
) -> None:
    request = validate_hybrid_planning_request(
        write_json_atomic(tmp_path / "request.json", _request())
    )
    recommendation = validate_hybrid_planning_recommendation(
        write_json_atomic(
            tmp_path / "recommendation.json",
            _recommendation(request.request_sha256),
        ),
        expected_request=request,
    )
    assert recommendation.payload["robot_actuation_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["candidate_attestation"].__setitem__(  # type: ignore[union-attr]
                "candidate_poses_sha256", SHA_A
            ),
            "attestation hash",
        ),
        (
            lambda payload: payload["dsspp_config"].__setitem__(  # type: ignore[union-attr]
                "augment_candidates", True
            ),
            "augmentation",
        ),
    ],
)
def test_planning_request_rejects_unattested_or_augmented_candidates(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = _request()
    mutation(payload)
    with pytest.raises(ContractError, match=message):
        validate_hybrid_planning_request(
            write_json_atomic(tmp_path / f"invalid-{message}.json", payload)
        )


def test_planning_recommendation_rejects_different_request_bytes(tmp_path: Path) -> None:
    payload = _request()
    request = validate_hybrid_planning_request(
        write_json_atomic(tmp_path / "request.json", payload)
    )
    changed = copy.deepcopy(payload)
    changed["visited_poses_xyz"].append([3.0, 3.0, 1.0])  # type: ignore[union-attr]
    changed_request = validate_hybrid_planning_request(
        write_json_atomic(tmp_path / "changed-request.json", changed)
    )
    recommendation_path = write_json_atomic(
        tmp_path / "recommendation.json",
        _recommendation(request.request_sha256),
    )
    with pytest.raises(ContractError, match="exact request artifact"):
        validate_hybrid_planning_recommendation(
            recommendation_path,
            expected_request=changed_request,
        )
