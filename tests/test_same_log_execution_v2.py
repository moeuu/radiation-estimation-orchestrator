"""Raw-spectrum MeasurementLog v2 benchmark integration tests."""

from __future__ import annotations

import json
from pathlib import Path

from orchestrator.adapters.base import recover_adapter_execution
from orchestrator.contracts import validate_measurement_log, validate_mle_result
from orchestrator.evaluation import evaluate_spectral_mission


def test_v2_benchmark_runs_only_pf_and_spectral_mle(benchmark_v2_output: Path) -> None:
    manifest = json.loads(
        (benchmark_v2_output / "benchmark_manifest.json").read_text(encoding="utf-8")
    )

    assert set(manifest["executions"]) == {"pf_strict", "mle_spectral"}
    assert set(manifest["validated_outputs"]) == {"pf_strict", "mle_spectral"}
    assert manifest["contracts"]["measurement_log"] == 2
    assert manifest["pipeline_order"][:3] == [
        "validate_measurement_log",
        "pure_pf_replay",
        "spectral_mle_replay",
    ]
    assert "count_mle_replay" not in manifest["pipeline_order"]
    assert manifest["truth_isolation"]["opened_only_after_all_result_validation"] is True


def test_v2_benchmark_metrics_match_executed_estimators(benchmark_v2_output: Path) -> None:
    metrics = json.loads((benchmark_v2_output / "metrics.json").read_text(encoding="utf-8"))

    assert set(metrics["estimators"]) == {"pf_strict", "mle_spectral"}
    assert set(metrics["operational"]["estimators"]) == {"pf_strict", "mle_spectral"}
    assert "surface_mle" in metrics["estimators"]["mle_spectral"]


def test_completed_spectral_mission_evaluation_opens_separate_truth(
    benchmark_v2_output: Path,
    truth_path: Path,
) -> None:
    log = validate_measurement_log(
        benchmark_v2_output.parent / "measurement-log-v2"
    )
    result = validate_mle_result(
        benchmark_v2_output / "results" / "mle_spectral",
        expected_mode="spectral",
    )

    metrics = evaluate_spectral_mission(
        measurement_log=log,
        truth_path=truth_path,
        mle_spectral_result=result,
        estimator_runtime_s=1.5,
    )

    assert metrics["operational"]["measurement_count"] == log.record_count
    assert "hybrid_final_mle_spectral" in metrics["estimators"]


def test_adapter_execution_receipt_recovers_without_rerun(
    benchmark_v2_output: Path,
) -> None:
    execution = recover_adapter_execution(
        benchmark_v2_output
        / "executions"
        / "mle_spectral"
        / "adapter_execution.json",
        output_directory=benchmark_v2_output / "results" / "mle_spectral",
    )

    assert execution.exit_code == 0
    assert execution.estimator == "surface_mle:spectral"
