from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.contracts import (
    validate_pf_checkpoint_v1,
    validate_pf_rj_receipt_v1,
    validate_spectral_mle_snapshot_v3,
)
from orchestrator.errors import ContractError
from orchestrator.hashing import sha256_file, write_json_atomic
from orchestrator.hybrid_v2 import (
    BlockEvidence,
    BlockVerificationQueue,
    VerificationPolicy,
    build_pf_rj_directive_v1,
)


def _snapshot(tmp_path: Path):
    path = write_json_atomic(
        tmp_path / "spectral_snapshot.json",
        {
            "schema_version": 3,
            "milestone": "pf_mle_hybrid_v2",
            "snapshot_id": "snapshot-2",
            "source_run_id": "run-v2",
            "prefix": {
                "measurement_log_schema_version": 2,
                "data_cutoff_step": 2,
                "data_cutoff_station": 0,
                "covered_step_ids": [0, 1, 2],
                "covered_records_sha256": "a" * 64,
                "prefix_measurement_log_sha256": "b" * 64,
                "station_boundaries_sha256": "c" * 64,
            },
            "fit": {
                "estimator_variant": "spectral",
                "converged": True,
                "warm_start": {
                    "used": False,
                    "source_snapshot_id": None,
                    "source_result_sha256": None,
                },
                "mle_result_sha256": "d" * 64,
            },
            "artifacts": {
                "estimate_npz_sha256": "a" * 64,
                "diagnostics_sha256": "b" * 64,
                "hotspot_clusters_sha256": "c" * 64,
                "predicted_spectra_sha256": "d" * 64,
                "predicted_spectra_shape": [3, 16],
            },
            "candidates": [
                {
                    "candidate_id": "candidate-1",
                    "isotope": "Cs-137",
                    "centroid_xyz": [1.0, 2.0, 3.0],
                    "covariance_xyz": [
                        [0.1, 0.0, 0.0],
                        [0.0, 0.2, 0.0],
                        [0.0, 0.0, 0.3],
                    ],
                    "integrated_strength_cps_1m": 1000.0,
                    "surface_kinds": ["ceiling"],
                    "patch_ids": [7],
                    "candidate_spectral_support_sha256": "e" * 64,
                }
            ],
            "safety": {
                "uses_only_prefix": True,
                "uses_pf_state": False,
                "uses_pf_candidates": False,
                "candidate_domain": "complete_surface_dictionary",
                "direct_pf_weight_increment": False,
            },
        },
    )
    return validate_spectral_mle_snapshot_v3(path)


def _directive(tmp_path: Path):
    snapshot = _snapshot(tmp_path)
    queue = BlockVerificationQueue(
        VerificationPolicy(
            support_log_likelihood_ratio=2.0,
            reject_log_likelihood_ratio=-2.0,
            minimum_blocks=2,
            minimum_distinct_stations=2,
        )
    )
    queue.register(
        snapshot_id="snapshot-2",
        snapshot_sha256=snapshot.snapshot_sha256,
        candidate_id="candidate-1",
        data_cutoff_step=2,
    )
    for block, station in (("block-1", 1), ("block-2", 2)):
        queue.corroborate(
            snapshot_id="snapshot-2",
            candidate_id="candidate-1",
            evidence=BlockEvidence(
                block_id=block,
                station_id=station,
                height_group_id=f"height-{station}",
                shield_program_id=f"shield-{station}",
                step_ids=(station + 2,),
                log_likelihood_ratio=1.5,
                score_artifact_sha256="f" * 64,
            ),
        )
    artifact = tmp_path / "pf_state.npz"
    artifact.write_bytes(b"opaque-pf-state")
    checkpoint_path = write_json_atomic(
        tmp_path / "pf_checkpoint.json",
        {
            "schema_version": 1,
            "checkpoint_family": "pure_pf_causal_state",
            "checkpoint_id": "checkpoint-5",
            "source_run_id": "run-v2",
            "measurement_log_schema_version": 2,
            "data_cutoff_step": 5,
            "data_cutoff_station": 2,
            "covered_step_ids": list(range(6)),
            "covered_records_sha256": "9" * 64,
            "prefix_measurement_log_sha256": "8" * 64,
            "pf_repository_commit": "1" * 40,
            "resolved_config_sha256": "2" * 64,
            "random_seed": 4,
            "state_artifact": artifact.name,
            "state_artifact_sha256": sha256_file(artifact),
            "rng_state_sha256": "3" * 64,
            "safety": {
                "prefix_causal": True,
                "truth_read": False,
                "batch_feedback_applied": False,
            },
        },
    )
    checkpoint = validate_pf_checkpoint_v1(checkpoint_path)
    return build_pf_rj_directive_v1(
        output_path=tmp_path / "rj_directive.json",
        snapshot=snapshot,
        pf_checkpoint=checkpoint,
        verification_candidates=queue.candidates,
        data_cutoff_station=2,
        prefix_measurement_log_sha256="8" * 64,
        covered_records_sha256="9" * 64,
        dimension_matching_transform="log_strength_auxiliary_v1",
    )


def _receipt(directive, *, log_alpha: float = 0.0) -> dict[str, object]:
    return {
        "schema_version": 1,
        "receipt_family": "pf_exact_rj_move",
        "receipt_id": "rj-receipt-1",
        "directive_id": directive.payload["directive_id"],
        "directive_sha256": directive.directive_sha256,
        "applied_once": True,
        "data_cutoff_step": directive.payload["data_cutoff_step"],
        "prefix_measurement_log_sha256": directive.payload[
            "prefix_measurement_log_sha256"
        ],
        "pf_checkpoint_before_sha256": directive.payload["pf_checkpoint_sha256"],
        "move": {
            "kind": "birth",
            "isotope": "Cs-137",
            "candidate_id": "candidate-1",
            "cardinality_before": 1,
            "cardinality_after": 2,
        },
        "target": {
            "log_density_before": -10.0,
            "log_density_after": -8.0,
            "includes_complete_pf_target": True,
        },
        "proposal": {
            "log_forward_density": -2.0,
            "log_reverse_density": -3.0,
            "log_abs_jacobian": 0.0,
            "dimension_matching_transform": "log_strength_auxiliary_v1",
        },
        "acceptance": {
            "log_acceptance_ratio": log_alpha,
            "uniform_draw": 0.5,
            "accepted": True,
        },
        "state": {
            "before_sha256": directive.payload["pf_state_before_sha256"],
            "after_sha256": "2" * 64,
        },
        "safety": {
            "direct_weight_change": False,
            "hard_prune": False,
            "pf_target_preserved": True,
        },
    }


def test_verified_spectral_candidate_builds_proposal_only_rj_directive(
    tmp_path: Path,
) -> None:
    directive = _directive(tmp_path)
    receipt_path = write_json_atomic(
        tmp_path / "rj_receipt.json",
        _receipt(directive),
    )

    receipt = validate_pf_rj_receipt_v1(receipt_path, expected_directive=directive)

    assert receipt.payload["move"]["cardinality_after"] == 2  # type: ignore[index]
    assert directive.payload["safety"]["hard_prune_requested"] is False  # type: ignore[index]


def test_rj_receipt_recomputes_acceptance_ratio(tmp_path: Path) -> None:
    directive = _directive(tmp_path)
    receipt_path = write_json_atomic(
        tmp_path / "rj_receipt.json",
        _receipt(directive, log_alpha=-0.5),
    )

    with pytest.raises(ContractError, match="acceptance ratio"):
        validate_pf_rj_receipt_v1(receipt_path, expected_directive=directive)
