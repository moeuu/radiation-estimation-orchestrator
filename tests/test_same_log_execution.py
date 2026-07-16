from __future__ import annotations

import json
from pathlib import Path

from orchestrator.hashing import sha256_file


def test_benchmark_runs_all_estimators_on_one_identical_truth_free_log(
    benchmark_output: Path, measurement_log_path: Path, truth_path: Path
) -> None:
    manifest = json.loads((benchmark_output / "benchmark_manifest.json").read_text())
    executions = manifest["executions"]
    expected_hash = manifest["measurement_log"]["sha256"]
    assert set(executions) == {"pf_strict", "mle_count", "mle_spectral"}
    for execution in executions.values():
        assert execution["measurement_log_sha256"] == expected_hash
        command = execution["command"]
        assert str(measurement_log_path.resolve()) in command
        assert str(truth_path.resolve()) not in command
        assert execution["exit_code"] == 0
        assert execution["timed_out"] is False
        assert execution["output_sha256"]
        assert execution["stdout_sha256"]
        assert execution["stderr_sha256"]
        assert ".staging" not in execution["stdout_path"]
        assert ".staging" not in execution["stderr_path"]
        assert (benchmark_output / execution["stdout_path"]).is_file()
        assert (benchmark_output / execution["stderr_path"]).is_file()
    assert manifest["truth_isolation"]["opened_only_after_all_result_validation"] is True
    assert manifest["truth_isolation"]["passed_to_estimator_commands"] is False


def test_benchmark_writes_hash_rich_validated_manifest(
    benchmark_output: Path, repository_root: Path
) -> None:
    manifest_path = benchmark_output / "benchmark_manifest.json"
    sidecar = (benchmark_output / "benchmark_manifest.sha256").read_text().split()[0]
    manifest = json.loads(manifest_path.read_text())
    assert sidecar == sha256_file(manifest_path)
    assert manifest["status"] == "complete"
    assert manifest["pipeline_order"][:4] == [
        "validate_measurement_log",
        "pure_pf_replay",
        "count_mle_replay",
        "spectral_mle_replay",
    ]
    assert set(manifest["validated_outputs"]) == {"pf_strict", "mle_count", "mle_spectral"}
    expected_names = {
        "pf_strict",
        "mle_count",
        "mle_spectral",
    }
    benchmark_config = manifest["benchmark_config"]
    file_hashes = benchmark_config["estimator_config_file_sha256"]
    resolved_hashes = benchmark_config["resolved_estimator_config_sha256"]
    expected_resolved_hashes = benchmark_config["expected_resolved_estimator_config_sha256"]
    assert set(file_hashes) == expected_names
    assert set(resolved_hashes) == expected_names
    assert resolved_hashes == expected_resolved_hashes
    assert file_hashes == {
        "pf_strict": sha256_file(
            repository_root / "configs" / "estimators" / "pf_strict_shared_small.json"
        ),
        "mle_count": sha256_file(
            repository_root / "configs" / "estimators" / "mle_count_shared_small.json"
        ),
        "mle_spectral": sha256_file(
            repository_root / "configs" / "estimators" / "mle_spectral_shared_small.json"
        ),
    }
    pf = json.loads((benchmark_output / "results" / "pf_strict" / "pf_posterior.json").read_text())
    count = json.loads(
        (benchmark_output / "results" / "mle_count" / "mle_diagnostics.json").read_text()
    )
    spectral = json.loads(
        (benchmark_output / "results" / "mle_spectral" / "mle_diagnostics.json").read_text()
    )
    assert resolved_hashes == {
        "pf_strict": pf["provenance"]["resolved_config_sha256"],
        "mle_count": count["diagnostics"]["provenance"]["resolved_estimator_config_sha256"],
        "mle_spectral": spectral["diagnostics"]["provenance"]["resolved_estimator_config_sha256"],
    }
    assert any(file_hashes[name] != resolved_hashes[name] for name in expected_names)


def test_all_required_metric_families_are_present(benchmark_output: Path) -> None:
    metrics = json.loads((benchmark_output / "metrics.json").read_text())
    operational = metrics["operational"]
    for key in (
        "measurement_count",
        "unique_xyz_actions",
        "detector_height_distribution_m",
        "motion_time_s",
        "shield_actuation_time_s",
        "live_time_s",
        "estimators",
    ):
        assert key in operational
    for estimator in ("pf_strict", "mle_count", "mle_spectral"):
        point = metrics["estimators"][estimator]["point_source"]
        for key in (
            "position_error_3d_m",
            "position_error_xy_m",
            "position_error_z_m",
            "cardinality_exact_match",
            "source_precision",
            "source_recall",
            "integrated_strength_error_cps_1m",
            "ceiling_source_recall",
        ):
            assert key in point
    for estimator in ("mle_count", "mle_spectral"):
        surface = metrics["estimators"][estimator]["surface_mle"]
        assert set(surface) == {
            "hotspot_cluster_centroid_error_m",
            "integrated_cluster_strength_error_cps_1m",
            "surface_kind_classification_accuracy",
            "mass_recovered_near_truth",
            "held_out_deviance",
        }
