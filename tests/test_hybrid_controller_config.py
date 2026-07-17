from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.contracts import validate_measurement_log
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import write_json_atomic
from orchestrator.hybrid.controller import HybridController
from orchestrator.hybrid.prefix import StationBoundarySchedule
from orchestrator.hybrid.run_config import HybridRunConfig


def _payload(repository_root: Path, tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "hybrid_run_id": "pytest-hybrid-v1",
        "measurement_log": str(
            repository_root / "fixtures" / "shared_small_run" / "measurement_log"
        ),
        "output_directory": str(tmp_path / "hybrid-output"),
        "pin_registry": str(repository_root / "PINNED_ESTIMATORS.json"),
        "pf_profile": "pf_strict",
        "random_seed": 12,
        "relocation_seed": 34,
        "expected_resolved_estimator_config_sha256": {
            "pf_strict": "a" * 64,
            "mle_count": "b" * 64,
            "mle_spectral": "c" * 64,
        },
        "estimator_configs": {
            "pf": str(repository_root / "configs" / "estimators" / "pf_strict_shared_small.json"),
            "mle_count": str(
                repository_root / "configs" / "estimators" / "mle_count_shared_small.json"
            ),
            "mle_spectral": str(
                repository_root / "configs" / "estimators" / "mle_spectral_shared_small.json"
            ),
        },
        "station_boundaries": [
            {"station_id": 0, "terminal_step_id": 2},
            {"station_id": 1, "terminal_step_id": 5},
            {"station_id": 2, "terminal_step_id": 8},
            {"station_id": 3, "terminal_step_id": 11},
        ],
        "hybrid_policy": {
            "mode": "proposal_only_mh",
            "station_interval": 1,
            "predictive_deviance_threshold": None,
        },
        "proposal_kernel": {
            "family": "defensive_truncated_gaussian_position",
            "position_sigma_xyz_m": [0.25, 0.25, 0.4],
            "defensive_weight": 0.1,
            "candidate_weight_floor": 1e-9,
        },
        "adapters": {"particle_filter": {}, "surface_mle": {}},
    }


def test_hybrid_run_config_resolves_only_safe_proposal_mh(
    repository_root: Path, tmp_path: Path
) -> None:
    path = write_json_atomic(tmp_path / "hybrid.json", _payload(repository_root, tmp_path))
    config = HybridRunConfig.load(path)
    assert config.hybrid_policy.mode.value == "proposal_only_mh"
    assert config.station_end_steps == ((0, 2), (1, 5), (2, 8), (3, 11))
    kernel = config.proposal_kernel.for_candidate(0.0)
    assert kernel["candidate_weight"] == 1e-9
    assert kernel["defensive_weight"] == 0.1


def test_hybrid_run_config_rejects_truth_and_unsafe_feedback(
    repository_root: Path, tmp_path: Path
) -> None:
    truth_payload = _payload(repository_root, tmp_path)
    truth_payload["truth"] = "do-not-open.json"
    truth_path = write_json_atomic(tmp_path / "with-truth.json", truth_payload)
    with pytest.raises(ContractError, match="truth paths"):
        HybridRunConfig.load(truth_path)

    unsafe_payload = _payload(repository_root, tmp_path)
    policy = unsafe_payload["hybrid_policy"]
    assert isinstance(policy, dict)
    policy["allow_hard_prune"] = True
    unsafe_path = write_json_atomic(tmp_path / "unsafe.json", unsafe_payload)
    with pytest.raises(ContractError, match="hard pruning"):
        HybridRunConfig.load(unsafe_path)


def test_controller_boundary_preflight_requires_exact_full_schedule(
    measurement_log_path: Path,
) -> None:
    log = validate_measurement_log(measurement_log_path)
    valid = StationBoundarySchedule.create(
        source_run_id=str(log.manifest["run_id"]),
        station_end_steps=((0, 2), (1, 5), (2, 8), (3, 11)),
    )
    HybridController._validate_explicit_boundaries(log, valid)
    incomplete = StationBoundarySchedule.create(
        source_run_id=str(log.manifest["run_id"]),
        station_end_steps=((0, 2), (1, 5), (2, 8)),
    )
    with pytest.raises(DataReuseError, match="exactly cover"):
        HybridController._validate_explicit_boundaries(log, incomplete)


def test_shared_hybrid_config_enables_one_attested_planning_boundary(
    repository_root: Path,
) -> None:
    config = HybridRunConfig.load(repository_root / "configs" / "hybrid" / "shared_small_run.json")
    assert tuple(config.planning_requests) == (5,)
    request = config.planning_requests[5]
    assert request["candidate_attestation"]["collision_checked"] is True  # type: ignore[index]
    assert request["candidate_attestation"]["reachability_filtered"] is True  # type: ignore[index]


def test_hybrid_config_rejects_controller_owned_planner_modes(
    repository_root: Path,
    tmp_path: Path,
) -> None:
    payload = _payload(repository_root, tmp_path)
    payload["planning"] = {
        "enabled": True,
        "requests": [
            {
                "data_cutoff_step": 5,
                "candidate_poses_xyz": [[1.0, 1.0, 0.5]],
                "candidate_attestation": {},
                "dsspp_config": {},
                "external_modes": [],
            }
        ],
    }
    path = write_json_atomic(tmp_path / "planner-mode-injection.json", payload)
    with pytest.raises(ContractError, match="Planning templates"):
        HybridRunConfig.load(path)
