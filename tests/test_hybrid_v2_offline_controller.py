from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.errors import ContractError
from orchestrator.hashing import write_json_atomic
from orchestrator.hybrid_v2.run_config import SpectralHybridRunConfig


def _payload(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "milestone": "pf_mle_hybrid_v2",
        "hybrid_run_id": "offline-v2-smoke",
        "measurement_log": str(tmp_path / "measurement-log"),
        "output_directory": str(tmp_path / "output"),
        "mode": "spectral_verification_only",
        "estimator_configs": {
            "pf_strict": str(tmp_path / "pf.json"),
            "mle_spectral": str(tmp_path / "mle.json"),
        },
        "station_boundaries": [{"station_id": 0, "terminal_step_id": 1}],
        "scheduler": {"station_interval": 1, "minimum_completed_stations": 1},
        "verification": {
            "support_log_likelihood_ratio": 3.0,
            "reject_log_likelihood_ratio": -3.0,
        },
    }


def test_offline_hybrid_resolves_repository_local_estimators(tmp_path: Path) -> None:
    write_json_atomic(tmp_path / "pf.json", {"num_particles": 8})
    write_json_atomic(tmp_path / "mle.json", {"maximum_iterations": 5})
    config_path = write_json_atomic(tmp_path / "hybrid.json", _payload(tmp_path))

    config = SpectralHybridRunConfig.load(config_path)

    assert config.mode.value == "spectral_verification_only"
    assert config.expected_pf_resolved_config_sha256
    assert config.expected_mle_resolved_config_sha256


def test_offline_hybrid_rejects_sibling_estimator_adapters(tmp_path: Path) -> None:
    write_json_atomic(tmp_path / "pf.json", {})
    write_json_atomic(tmp_path / "mle.json", {})
    payload = _payload(tmp_path)
    payload["pin_registry"] = "PINNED_ESTIMATORS.json"
    payload["adapters"] = {"surface_mle": {}}
    config_path = write_json_atomic(tmp_path / "external.json", payload)

    with pytest.raises(ContractError, match="estimators are local"):
        SpectralHybridRunConfig.load(config_path)
