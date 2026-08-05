from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.errors import ContractError
from orchestrator.hashing import write_json_atomic
from orchestrator.hybrid_v2.live_config import LiveSpectralHybridRunConfig


def _payload(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "milestone": "pf_mle_hybrid_live_v2",
        "hybrid_run_id": "live-smoke",
        "output_directory": str(tmp_path / "output"),
        "mode": "spectral_exact_rj",
        "estimator_configs": {
            "pf_strict": str(tmp_path / "pf.json"),
            "mle_spectral": str(tmp_path / "mle.json"),
        },
        "scheduler": {"station_interval": 2},
        "verification": {
            "support_log_likelihood_ratio": 3.0,
            "reject_log_likelihood_ratio": -3.0,
            "minimum_distinct_stations": 2,
            "minimum_distinct_heights": 2,
            "minimum_distinct_shield_programs": 2,
        },
        "runtime": {
            "repository_path": str(tmp_path / "runtime"),
            "revision": "3" * 40,
            "scenario_path": str(tmp_path / "scenario.json"),
        },
        "planner": {
            "dwell_time_s": 10.0,
            "dsspp_config": {"candidate_count": 512},
        },
        "mission_budget": {
            "max_actions": 5,
            "max_total_time_s": 1000.0,
            "max_live_time_s": 500.0,
            "max_travel_time_s": 500.0,
            "max_shield_actuation_time_s": 100.0,
        },
    }


def test_live_config_requires_exact_runtime_and_distinct_inference_inputs(
    tmp_path: Path,
) -> None:
    write_json_atomic(tmp_path / "pf.json", {"num_particles": 8})
    write_json_atomic(tmp_path / "mle.json", {"maximum_iterations": 5})
    path = write_json_atomic(tmp_path / "live.json", _payload(tmp_path))

    config = LiveSpectralHybridRunConfig.load(path)

    assert config.mode.value == "spectral_exact_rj"
    assert config.runtime.revision == "3" * 40
    assert config.verification_policy.minimum_distinct_heights == 2
    assert config.budget.max_actions == 5


def test_live_config_rejects_truth_or_prebuilt_log(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["truth_path"] = "truth.json"
    payload["measurement_log"] = "existing-log"
    path = write_json_atomic(tmp_path / "invalid-live.json", payload)

    with pytest.raises(ContractError, match="truth or a prebuilt"):
        LiveSpectralHybridRunConfig.load(path)


def test_live_config_rejects_external_estimator_adapters(tmp_path: Path) -> None:
    write_json_atomic(tmp_path / "pf.json", {})
    write_json_atomic(tmp_path / "mle.json", {})
    payload = _payload(tmp_path)
    payload["pin_registry"] = "PINNED_ESTIMATORS.json"
    payload["adapters"] = {"particle_filter": {}}
    path = write_json_atomic(tmp_path / "external.json", payload)

    with pytest.raises(ContractError, match="estimators are local"):
        LiveSpectralHybridRunConfig.load(path)
