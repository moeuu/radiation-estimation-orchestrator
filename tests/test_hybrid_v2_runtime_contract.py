from __future__ import annotations

import pytest

from orchestrator.errors import ContractError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes
from orchestrator.hybrid_v2.runtime_client import ResumableAdaptiveRuntimeClient


def _runtime_candidates() -> dict[str, object]:
    body = {
        "candidate_poses_xyz": [[1.0, 2.0, 0.5], [1.0, 2.0, 2.0]],
        "travel_costs_s": [1.0, 2.0],
        "candidate_path_sha256": ["1" * 64, "2" * 64],
        "allowed_pair_ids": [0, 63],
        "collision_checked": True,
        "reachability_filtered": True,
        "path_attestation_sha256": "a" * 64,
    }
    return {
        "snapshot_id": "runtime-candidates-1",
        **body,
        "snapshot_sha256": sha256_bytes(canonical_json_bytes(body)),
    }


def test_runtime_candidate_contract_preserves_xyz_and_safety_attestation() -> None:
    snapshot = ResumableAdaptiveRuntimeClient._parse_candidates(_runtime_candidates())

    assert {pose[2] for pose in snapshot.candidate_poses_xyz} == {0.5, 2.0}
    assert snapshot.collision_checked is True
    assert snapshot.reachability_filtered is True


def test_runtime_candidate_contract_rejects_unattested_candidates() -> None:
    payload = _runtime_candidates()
    payload["collision_checked"] = False
    body = {
        key: payload[key]
        for key in (
            "candidate_poses_xyz",
            "travel_costs_s",
            "candidate_path_sha256",
            "allowed_pair_ids",
            "collision_checked",
            "reachability_filtered",
            "path_attestation_sha256",
        )
    }
    payload["snapshot_sha256"] = sha256_bytes(canonical_json_bytes(body))

    with pytest.raises(ContractError, match="collision/reachability"):
        ResumableAdaptiveRuntimeClient._parse_candidates(payload)


def test_runtime_receipt_binds_durable_prefix_and_next_safe_candidates() -> None:
    candidates = _runtime_candidates()
    records = [
        {
            "step_id": 0,
            "station_id": 0,
            "detector_pose_xyz": [1.0, 2.0, 0.5],
            "fe_orientation_index": 0,
            "pb_orientation_index": 0,
            "live_time_s": 10.0,
            "travel_time_s": 1.0,
            "shield_actuation_time_s": 0.1,
            "metadata": {
                "station_complete": True,
                "realized_path_sha256": "1" * 64,
            },
        }
    ]
    receipt_body = {
        "decision_id": "decision-1",
        "runtime_receipt_id": "receipt-1",
        "records": records,
        "measurement_log_prefix_path": "/durable/prefix",
        "measurement_log_prefix_sha256": "9" * 64,
        "candidate_snapshot_sha256": candidates["snapshot_sha256"],
    }
    event = {
        "type": "action_receipt",
        "schema_version": 2,
        **receipt_body,
        "runtime_receipt_sha256": sha256_bytes(
            canonical_json_bytes(receipt_body)
        ),
        "candidates": candidates,
    }
    client = object.__new__(ResumableAdaptiveRuntimeClient)

    realized = client._parse_receipt(event, expected_decision_id="decision-1")

    assert realized.measurement_log_prefix_sha256 == "9" * 64
    assert realized.next_candidates.candidate_path_sha256[0] == "1" * 64
