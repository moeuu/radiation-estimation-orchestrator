from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from orchestrator.contracts import (
    MLEResultInfo,
    validate_hybrid_ledger_summary,
    validate_hybrid_result,
    validate_measurement_log,
    validate_mle_result,
    validate_pf_result,
)
from orchestrator.errors import ContractError
from orchestrator.hashing import write_json_atomic
from orchestrator.hybrid import ObservationUseLedger
from orchestrator.hybrid.reporting import build_hybrid_result, write_hybrid_result_bundle

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _cold_full(result: MLEResultInfo, *, step_ids: tuple[int, ...]) -> MLEResultInfo:
    diagnostics = copy.deepcopy(dict(result.diagnostics))
    nested = diagnostics["diagnostics"]
    assert isinstance(nested, dict)
    lineage = {
        "schema_version": 1,
        "covered_step_ids": list(step_ids),
        "data_cutoff_step": step_ids[-1],
        "data_cutoff_station": 3,
        "record_count": len(step_ids),
        "covered_records_sha256": SHA_C,
        "station_boundary_attestation": {},
        "fit_kind": "cold_start_all_history",
        "warm_start": None,
    }
    nested["causal_lineage"] = lineage
    provenance = nested["provenance"]
    assert isinstance(provenance, dict)
    provenance["causal_lineage"] = lineage
    return replace(result, diagnostics=MappingProxyType(diagnostics))


def _inputs(measurement_log_path: Path, benchmark_output: Path, tmp_path: Path):
    measurement = validate_measurement_log(measurement_log_path)
    pf = validate_pf_result(
        benchmark_output / "results" / "pf_strict",
        expected_log_sha256=measurement.measurement_log_sha256,
    )
    count = _cold_full(
        validate_mle_result(
            benchmark_output / "results" / "mle_count",
            expected_mode="count",
            expected_log_sha256=measurement.measurement_log_sha256,
        ),
        step_ids=measurement.step_ids,
    )
    spectral = _cold_full(
        validate_mle_result(
            benchmark_output / "results" / "mle_spectral",
            expected_mode="spectral",
            expected_log_sha256=measurement.measurement_log_sha256,
        ),
        step_ids=measurement.step_ids,
    )
    ledger_payload = ObservationUseLedger(
        source_run_id=str(measurement.manifest["run_id"]),
        station_boundary_schedule_sha256=SHA_A,
    ).summary()
    ledger_path = write_json_atomic(tmp_path / "hybrid_ledger_summary.json", ledger_payload)
    ledger = validate_hybrid_ledger_summary(ledger_path)
    return measurement, pf, count, spectral, ledger


def _build(measurement_log_path: Path, benchmark_output: Path, tmp_path: Path):
    measurement, pf, count, spectral, ledger = _inputs(
        measurement_log_path,
        benchmark_output,
        tmp_path,
    )
    payload = build_hybrid_result(
        hybrid_run_id="pytest-hybrid-v1",
        hybrid_mode="verification_only",
        measurement_log=measurement,
        source_measurement_log_sha256=measurement.measurement_log_sha256,
        source_measurement_log_record_count=measurement.record_count,
        station_boundary_schedule_sha256=SHA_A,
        final_pf_result=pf,
        final_count_mle_result=count,
        final_spectral_mle_result=spectral,
        ledger=ledger,
        snapshots=(),
        directives=(),
        receipts=(),
        future_candidate_scores=(),
        planning_recommendations=(),
        verification_queue_sha256=SHA_A,
        verification_counts={},
        orchestrator_commit="test-orchestrator-commit",
        hybrid_config_sha256=SHA_B,
        pin_registry_sha256=SHA_C,
        execution_evidence_sha256=SHA_A,
    )
    return payload, measurement, pf, count, spectral, ledger


def test_hybrid_report_is_deterministic_and_binds_authoritative_spectral_mle(
    measurement_log_path: Path,
    benchmark_output: Path,
    tmp_path: Path,
) -> None:
    payload, measurement, pf, count, spectral, ledger = _build(
        measurement_log_path,
        benchmark_output,
        tmp_path,
    )
    second = build_hybrid_result(
        hybrid_run_id="pytest-hybrid-v1",
        hybrid_mode="verification_only",
        measurement_log=measurement,
        source_measurement_log_sha256=measurement.measurement_log_sha256,
        source_measurement_log_record_count=measurement.record_count,
        station_boundary_schedule_sha256=SHA_A,
        final_pf_result=pf,
        final_count_mle_result=count,
        final_spectral_mle_result=spectral,
        ledger=ledger,
        snapshots=(),
        directives=(),
        receipts=(),
        future_candidate_scores=(),
        planning_recommendations=(),
        verification_queue_sha256=SHA_A,
        verification_counts={},
        orchestrator_commit="test-orchestrator-commit",
        hybrid_config_sha256=SHA_B,
        pin_registry_sha256=SHA_C,
        execution_evidence_sha256=SHA_A,
    )
    assert payload == second

    result, sidecar = write_hybrid_result_bundle(
        tmp_path / "result",
        payload,
        expected_measurement_log=measurement,
        expected_source_measurement_log=measurement,
        expected_pf_result=pf,
        expected_final_count_mle_result=count,
        expected_final_spectral_mle_result=spectral,
        expected_ledger=ledger,
        expected_snapshots=(),
        expected_directives=(),
        expected_receipts=(),
        expected_planning_recommendations=(),
    )
    assert sidecar.read_text(encoding="ascii").endswith("  hybrid_result.json\n")
    assert result.payload["authoritative_report"]["result_sha256"] == spectral.result_sha256  # type: ignore[index]
    assert result.authoritative_clusters
    assert result.payload["limitations"] == {  # type: ignore[comparison-overlap]
        "rj_birth_death_implemented": False,
        "hard_prune_implemented": False,
        "direct_mle_reweight_implemented": False,
        "live_closed_loop_planner_control_implemented": False,
        "algorithmic_planner_recommendation_implemented": True,
        "planner_scope": "offline_replay_recommendation_no_actuation",
        "fixed_cardinality_position_relocation_only": True,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["safety"].__setitem__(  # type: ignore[union-attr]
                "target_preserving_fixed_cardinality_mh_performed", True
            ),
            "MH safety claim",
        ),
        (
            lambda payload: payload["verification_summary"].__setitem__(  # type: ignore[union-attr]
                "pending", 1
            ),
            "state counts",
        ),
        (
            lambda payload: payload["estimator_roles"]["final_report"].__setitem__(  # type: ignore[index,union-attr]
                "estimator_variant", "count"
            ),
            "final report role",
        ),
    ],
)
def test_hybrid_result_rejects_cross_contract_drift(
    measurement_log_path: Path,
    benchmark_output: Path,
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload, *_ = _build(measurement_log_path, benchmark_output, tmp_path)
    mutation(payload)
    path = write_json_atomic(tmp_path / f"invalid-{message.replace(' ', '-')}.json", payload)
    with pytest.raises(ContractError, match=message):
        validate_hybrid_result(path)


def test_hybrid_report_builder_rejects_warm_started_final_mle(
    measurement_log_path: Path,
    benchmark_output: Path,
    tmp_path: Path,
) -> None:
    measurement, pf, count, spectral, ledger = _inputs(
        measurement_log_path,
        benchmark_output,
        tmp_path,
    )
    diagnostics = copy.deepcopy(dict(spectral.diagnostics))
    nested = diagnostics["diagnostics"]
    assert isinstance(nested, dict)
    lineage = nested["causal_lineage"]
    assert isinstance(lineage, dict)
    lineage["fit_kind"] = "warm_start_all_history"
    lineage["warm_start"] = {"snapshot_id": "prior"}
    warmed = replace(spectral, diagnostics=MappingProxyType(diagnostics))

    with pytest.raises(ContractError, match="cold fit"):
        build_hybrid_result(
            hybrid_run_id="pytest-hybrid-v1",
            hybrid_mode="proposal_only_mh",
            measurement_log=measurement,
            source_measurement_log_sha256=measurement.measurement_log_sha256,
            source_measurement_log_record_count=measurement.record_count,
            station_boundary_schedule_sha256=SHA_A,
            final_pf_result=pf,
            final_count_mle_result=count,
            final_spectral_mle_result=warmed,
            ledger=ledger,
            snapshots=(),
            directives=(),
            receipts=(),
            future_candidate_scores=(),
            planning_recommendations=(),
            verification_queue_sha256=SHA_A,
            verification_counts={},
            orchestrator_commit="test-orchestrator-commit",
            hybrid_config_sha256=SHA_B,
            pin_registry_sha256=SHA_C,
            execution_evidence_sha256=SHA_A,
        )


def test_hybrid_report_rejects_nonconverged_authoritative_spectral_mle(
    measurement_log_path: Path,
    benchmark_output: Path,
    tmp_path: Path,
) -> None:
    measurement, pf, count, spectral, ledger = _inputs(
        measurement_log_path,
        benchmark_output,
        tmp_path,
    )
    diagnostics = copy.deepcopy(dict(spectral.diagnostics))
    diagnostics["converged"] = False
    nonconverged = replace(spectral, diagnostics=MappingProxyType(diagnostics))

    with pytest.raises(ContractError, match="converged=true"):
        build_hybrid_result(
            hybrid_run_id="pytest-hybrid-v1",
            hybrid_mode="proposal_only_mh",
            measurement_log=measurement,
            source_measurement_log_sha256=measurement.measurement_log_sha256,
            source_measurement_log_record_count=measurement.record_count,
            station_boundary_schedule_sha256=SHA_A,
            final_pf_result=pf,
            final_count_mle_result=count,
            final_spectral_mle_result=nonconverged,
            ledger=ledger,
            snapshots=(),
            directives=(),
            receipts=(),
            future_candidate_scores=(),
            planning_recommendations=(),
            verification_queue_sha256=SHA_A,
            verification_counts={},
            orchestrator_commit="test-orchestrator-commit",
            hybrid_config_sha256=SHA_B,
            pin_registry_sha256=SHA_C,
            execution_evidence_sha256=SHA_A,
        )


def test_hybrid_result_forbids_claiming_unimplemented_inference(
    measurement_log_path: Path,
    benchmark_output: Path,
    tmp_path: Path,
) -> None:
    payload, *_ = _build(measurement_log_path, benchmark_output, tmp_path)
    limitations = payload["limitations"]
    assert isinstance(limitations, dict)
    limitations["rj_birth_death_implemented"] = True
    path = write_json_atomic(tmp_path / "unsafe-claim.json", payload)
    with pytest.raises(ContractError, match="False"):
        validate_hybrid_result(path)


def test_hybrid_result_authoritative_clusters_must_mirror_spectral_bundle(
    measurement_log_path: Path,
    benchmark_output: Path,
    tmp_path: Path,
) -> None:
    payload, measurement, pf, count, spectral, ledger = _build(
        measurement_log_path,
        benchmark_output,
        tmp_path,
    )
    report = payload["authoritative_report"]
    assert isinstance(report, dict)
    clusters = report["hotspot_clusters"]
    assert isinstance(clusters, list)
    clusters[0]["integrated_strength_cps_1m"] += 1.0
    path = write_json_atomic(tmp_path / "cluster-drift.json", payload)
    with pytest.raises(ContractError, match="mirror final spectral"):
        validate_hybrid_result(
            path,
            expected_measurement_log=measurement,
            expected_pf_result=pf,
            expected_final_count_mle_result=count,
            expected_final_spectral_mle_result=spectral,
            expected_ledger=ledger,
        )


def test_hybrid_result_distinguishes_source_and_station_marked_inference_logs(
    measurement_log_path: Path,
    benchmark_output: Path,
    tmp_path: Path,
) -> None:
    payload, measurement, *_ = _build(measurement_log_path, benchmark_output, tmp_path)
    log_identity = payload["measurement_log"]
    assert isinstance(log_identity, dict)
    log_identity["source_measurement_log_sha256"] = SHA_B
    path = write_json_atomic(tmp_path / "separate-log-identities.json", payload)

    accepted = validate_hybrid_result(path)
    accepted_log = accepted.payload["measurement_log"]
    assert isinstance(accepted_log, dict)
    assert accepted_log["derivation_kind"] == "predeclared_station_boundary_markers_only"
    assert (
        accepted_log["source_measurement_log_sha256"]
        != (accepted_log["inference_measurement_log_sha256"])
    )
    with pytest.raises(ContractError, match="source MeasurementLog hash"):
        validate_hybrid_result(path, expected_source_measurement_log=measurement)


def test_station_marker_derivation_cannot_change_record_count(
    measurement_log_path: Path,
    benchmark_output: Path,
    tmp_path: Path,
) -> None:
    payload, *_ = _build(measurement_log_path, benchmark_output, tmp_path)
    log_identity = payload["measurement_log"]
    assert isinstance(log_identity, dict)
    log_identity["source_record_count"] -= 1
    path = write_json_atomic(tmp_path / "changed-record-count.json", payload)
    with pytest.raises(ContractError, match="may not change"):
        validate_hybrid_result(path)
