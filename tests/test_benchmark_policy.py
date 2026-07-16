from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.benchmark import BenchmarkConfig
from orchestrator.errors import ContractError
from orchestrator.hashing import load_json, write_json_atomic


def _benchmark_payload(repository_root: Path, tmp_path: Path) -> dict[str, object]:
    payload = load_json(repository_root / "configs" / "benchmark" / "shared_small_run.json")
    payload["measurement_log"] = str(
        repository_root / "fixtures" / "shared_small_run" / "measurement_log"
    )
    payload["truth"] = str(
        repository_root / "fixtures" / "shared_small_run" / "evaluation" / "truth.json"
    )
    payload["output_directory"] = str(tmp_path / "output")
    payload["pin_registry"] = str(repository_root / "PINNED_ESTIMATORS.json")
    payload["estimator_configs"] = {
        "pf": str(repository_root / "configs" / "estimators" / "pf_strict_shared_small.json"),
        "mle_count": str(
            repository_root / "configs" / "estimators" / "mle_count_shared_small.json"
        ),
        "mle_spectral": str(
            repository_root / "configs" / "estimators" / "mle_spectral_shared_small.json"
        ),
    }
    return payload


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
