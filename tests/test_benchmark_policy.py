from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.benchmark import BenchmarkConfig
from orchestrator.errors import ContractError
from orchestrator.hashing import write_json_atomic


def _benchmark_payload(repository_root: Path, tmp_path: Path) -> dict[str, object]:
    pin_registry = tmp_path / "archived-v1-pins.json"
    if not pin_registry.exists():
        pin_registry.write_text("{}\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "benchmark_id": "archived-policy-test",
        "measurement_log": str(
            repository_root / "fixtures" / "shared_small_run" / "measurement_log"
        ),
        "truth": str(
            repository_root / "fixtures" / "shared_small_run" / "evaluation" / "truth.json"
        ),
        "output_directory": str(tmp_path / "output"),
        "pin_registry": str(pin_registry),
        "pf_profile": "pf_strict",
        "random_seed": 1,
        "expected_resolved_estimator_config_sha256": {
            "pf_strict": "a" * 64,
            "mle_count": "b" * 64,
            "mle_spectral": "c" * 64,
        },
        "estimator_configs": {
            "pf": str(
                repository_root / "configs" / "estimators" / "pf_strict_shared_small.json"
            ),
            "mle_count": str(
                repository_root / "configs" / "estimators" / "mle_count_shared_small.json"
            ),
            "mle_spectral": str(
                repository_root / "configs" / "estimators" / "mle_spectral_shared_small.json"
            ),
        },
        "adapters": {"particle_filter": {}, "surface_mle": {}},
    }


@pytest.mark.parametrize(
    ("adapter_name", "field", "value", "message"),
    [
        ("particle_filter", "verify_revision", False, "verify_revision must be true"),
        ("surface_mle", "require_clean", False, "require_clean must be true"),
        (
            "particle_filter",
            "allowed_dirty_prefixes",
            ["results/", "src/"],
            "broadens the artifact-only dirty allowlist",
        ),
        ("surface_mle", "command", ["python", "untrusted.py"], "cannot override"),
    ],
)
def test_production_adapter_policy_cannot_be_disabled_or_broadened(
    repository_root: Path,
    tmp_path: Path,
    adapter_name: str,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _benchmark_payload(repository_root, tmp_path)
    adapters = payload["adapters"]
    assert isinstance(adapters, dict)
    adapter = adapters[adapter_name]
    assert isinstance(adapter, dict)
    adapter[field] = value
    config = write_json_atomic(tmp_path / f"{adapter_name}-{field}.json", payload)
    with pytest.raises(ContractError, match=message):
        BenchmarkConfig.load(config)


def test_benchmark_requires_independent_expected_resolved_hashes(
    repository_root: Path, tmp_path: Path
) -> None:
    payload = _benchmark_payload(repository_root, tmp_path)
    del payload["expected_resolved_estimator_config_sha256"]
    config = write_json_atomic(tmp_path / "missing-resolved-hashes.json", payload)
    with pytest.raises(ContractError, match="expected_resolved_estimator_config_sha256"):
        BenchmarkConfig.load(config)


def test_v2_benchmark_rejects_count_mle_or_missing_spectral_pair(
    repository_root: Path, tmp_path: Path
) -> None:
    payload = _benchmark_payload(repository_root, tmp_path)
    payload["schema_version"] = 2
    payload["measurement_log_schema_version"] = 2
    payload["estimator_runs"] = ["pf_strict", "mle_count", "mle_spectral"]
    config = write_json_atomic(tmp_path / "invalid-v2-estimators.json", payload)

    with pytest.raises(ContractError, match="requires estimator_runs"):
        BenchmarkConfig.load(config)
