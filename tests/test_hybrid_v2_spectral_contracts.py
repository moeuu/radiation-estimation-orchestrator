"""Contract and causality tests for raw-spectrum hybrid v2 artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.contracts import (
    validate_future_spectral_candidate_score_v2,
    validate_future_spectral_score_request_v1,
    validate_spectral_mle_snapshot_v3,
)
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import write_json_atomic

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _snapshot_payload() -> dict[str, object]:
    return {
        "schema_version": 3,
        "milestone": "pf_mle_hybrid_v2",
        "snapshot_id": "spectral-snapshot-2",
        "source_run_id": "run-v2",
        "prefix": {
            "measurement_log_schema_version": 2,
            "data_cutoff_step": 2,
            "data_cutoff_station": 0,
            "covered_step_ids": [0, 1, 2],
            "covered_records_sha256": SHA_A,
            "prefix_measurement_log_sha256": SHA_B,
            "station_boundaries_sha256": SHA_C,
        },
        "fit": {
            "estimator_variant": "spectral",
            "converged": True,
            "warm_start": {
                "used": False,
                "source_snapshot_id": None,
                "source_result_sha256": None,
            },
            "mle_result_sha256": SHA_A,
        },
        "artifacts": {
            "estimate_npz_sha256": SHA_A,
            "diagnostics_sha256": SHA_B,
            "hotspot_clusters_sha256": SHA_C,
            "predicted_spectra_sha256": SHA_A,
            "predicted_spectra_shape": [3, 16],
        },
        "candidates": [
            {
                "candidate_id": "candidate-cs-1",
                "isotope": "Cs-137",
                "centroid_xyz": [1.0, 2.0, 3.0],
                "covariance_xyz": [
                    [0.1, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.3],
                ],
                "integrated_strength_cps_1m": 1200.0,
                "surface_kinds": ["ceiling"],
                "patch_ids": [3, 4],
                "candidate_spectral_support_sha256": SHA_B,
            }
        ],
        "safety": {
            "uses_only_prefix": True,
            "uses_pf_state": False,
            "uses_pf_candidates": False,
            "candidate_domain": "complete_surface_dictionary",
            "direct_pf_weight_increment": False,
        },
    }


def _score_payload(snapshot_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "score_family": "frozen_spectral_full_vs_candidate_zero_log_likelihood_ratio",
        "snapshot_id": "spectral-snapshot-2",
        "snapshot_sha256": snapshot_sha256,
        "snapshot_data_cutoff_step": 2,
        "current_measurement_log_sha256": SHA_C,
        "future_step_ids": [3, 4, 5],
        "blocks": [
            {
                "block_id": "station-1-height-low-shield-a",
                "station_id": 1,
                "height_group_id": "height-low",
                "shield_program_id": "shield-a",
                "step_ids": [3, 4],
            },
            {
                "block_id": "station-2-height-high-shield-b",
                "station_id": 2,
                "height_group_id": "height-high",
                "shield_program_id": "shield-b",
                "step_ids": [5],
            },
        ],
        "candidates": [
            {
                "candidate_id": "candidate-cs-1",
                "cumulative_log_likelihood_ratio": 2.5,
                "block_scores": [
                    {
                        "block_id": "station-1-height-low-shield-a",
                        "full_model_poisson_deviance": 10.0,
                        "candidate_zero_poisson_deviance": 13.0,
                        "log_likelihood_ratio": 1.5,
                        "energy_bin_count": 16,
                    },
                    {
                        "block_id": "station-2-height-high-shield-b",
                        "full_model_poisson_deviance": 5.0,
                        "candidate_zero_poisson_deviance": 7.0,
                        "log_likelihood_ratio": 1.0,
                        "energy_bin_count": 16,
                    },
                ],
            }
        ],
        "safety": {
            "future_only": True,
            "snapshot_frozen": True,
            "same_observation_reweight": False,
        },
    }


def test_spectral_snapshot_and_future_blocks_validate(tmp_path: Path) -> None:
    snapshot_path = write_json_atomic(tmp_path / "snapshot.json", _snapshot_payload())
    snapshot = validate_spectral_mle_snapshot_v3(snapshot_path)
    score_path = write_json_atomic(
        tmp_path / "score.json",
        _score_payload(snapshot.snapshot_sha256),
    )

    score = validate_future_spectral_candidate_score_v2(
        score_path,
        expected_snapshot=snapshot,
    )

    assert score.payload["future_step_ids"] == [3, 4, 5]
    assert snapshot.cutoff_station == 0


def test_spectral_score_rejects_cutoff_reuse(tmp_path: Path) -> None:
    snapshot_path = write_json_atomic(tmp_path / "snapshot.json", _snapshot_payload())
    snapshot = validate_spectral_mle_snapshot_v3(snapshot_path)
    payload = _score_payload(snapshot.snapshot_sha256)
    payload["future_step_ids"] = [2, 3, 4, 5]
    blocks = payload["blocks"]
    assert isinstance(blocks, list)
    block = blocks[0]
    assert isinstance(block, dict)
    block["step_ids"] = [2, 3, 4]
    score_path = write_json_atomic(tmp_path / "score.json", payload)

    with pytest.raises(DataReuseError, match="post-cutoff"):
        validate_future_spectral_candidate_score_v2(score_path, expected_snapshot=snapshot)


def test_spectral_score_requires_exact_block_partition(tmp_path: Path) -> None:
    snapshot_path = write_json_atomic(tmp_path / "snapshot.json", _snapshot_payload())
    snapshot = validate_spectral_mle_snapshot_v3(snapshot_path)
    payload = _score_payload(snapshot.snapshot_sha256)
    blocks = payload["blocks"]
    assert isinstance(blocks, list)
    block = blocks[1]
    assert isinstance(block, dict)
    block["step_ids"] = [4, 5]
    score_path = write_json_atomic(tmp_path / "score.json", payload)

    with pytest.raises(ContractError, match="partition"):
        validate_future_spectral_candidate_score_v2(score_path, expected_snapshot=snapshot)


def test_incremental_score_request_rejects_reused_steps(tmp_path: Path) -> None:
    snapshot_path = write_json_atomic(tmp_path / "snapshot.json", _snapshot_payload())
    snapshot = validate_spectral_mle_snapshot_v3(snapshot_path)
    request_path = write_json_atomic(
        tmp_path / "request.json",
        {
            "schema_version": 1,
            "request_id": "request-1",
            "snapshot_id": snapshot.payload["snapshot_id"],
            "snapshot_sha256": snapshot.snapshot_sha256,
            "snapshot_data_cutoff_step": 2,
            "current_measurement_log_sha256": SHA_C,
            "requested_future_step_ids": [4, 5],
            "previously_scored_step_ids": [3, 4],
            "grouping": {
                "station": "station_id",
                "height": "detector_z_tolerance_group",
                "shield_program": "metadata_shield_program_id",
                "height_tolerance_m": 0.05,
            },
            "safety": {
                "future_only": True,
                "snapshot_frozen": True,
                "refit_allowed": False,
                "steps_once_only": True,
            },
        },
    )

    with pytest.raises(DataReuseError, match="reuse"):
        validate_future_spectral_score_request_v1(
            request_path,
            expected_snapshot=snapshot,
        )
