from __future__ import annotations

import copy
from pathlib import Path
from types import MappingProxyType

import pytest

from orchestrator.contracts import (
    MLESnapshotInfo,
    validate_future_candidate_score,
    validate_measurement_log,
)
from orchestrator.errors import ContractError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes, sha256_file, write_json_atomic
from orchestrator.hybrid.prefix_log import measurement_records_sha256


def _artifacts(measurement_log_path: Path, tmp_path: Path):
    log = validate_measurement_log(measurement_log_path)
    cutoff = 2
    candidate = {
        "snapshot_candidate_id": "candidate-0",
        "cluster_id": 0,
        "isotope": log.isotopes[0],
        "centroid_xyz": [1.0, 2.0, 0.0],
        "integrated_strength_cps_1m": 3.0,
        "surface_kinds": ["floor"],
        "patch_ids": [0],
    }
    snapshot_payload = {
        "snapshot_id": "snapshot-0",
        "source_run_id": log.manifest["run_id"],
        "data_cutoff_step": cutoff,
        "data_cutoff_station": 0,
        "mle_result_sha256": "a" * 64,
        "prefix_measurement_log_sha256": "b" * 64,
        "covered_station_boundaries_sha256": "c" * 64,
        "clusters": [candidate],
    }
    snapshot_path = write_json_atomic(tmp_path / "snapshot.json", snapshot_payload)
    snapshot = MLESnapshotInfo(
        path=snapshot_path,
        payload=MappingProxyType(snapshot_payload),
        snapshot_sha256=sha256_bytes(canonical_json_bytes(snapshot_payload)),
    )
    future_steps = [step for step in log.step_ids if step > cutoff]
    future_stations = [
        station
        for step, station in zip(log.step_ids, log.station_ids, strict=True)
        if step > cutoff
    ]
    rows = [
        {
            "step_id": step,
            "station_id": station,
            "log_predictive_likelihood_ratio": 0.25,
        }
        for step, station in zip(future_steps, future_stations, strict=True)
    ]
    payload = {
        "schema_version": 1,
        "score_family": "frozen_count_snapshot_cluster_log_predictive_ratio",
        "source_run_id": log.manifest["run_id"],
        "snapshot_id": "snapshot-0",
        "snapshot_data_cutoff_step": cutoff,
        "snapshot_data_cutoff_station": 0,
        "future_step_ids": future_steps,
        "future_station_ids": future_stations,
        "isotope_names": list(log.isotopes),
        "candidates": [
            {
                "snapshot_candidate_id": "candidate-0",
                "cluster_id": 0,
                "isotope": log.isotopes[0],
                "patch_ids": [0],
                "future_step_scores": rows,
                "cumulative_log_predictive_likelihood_ratio": 0.25 * len(rows),
            }
        ],
        "hashes": {
            "snapshot_file_sha256": sha256_file(snapshot_path),
            "snapshot_canonical_sha256": snapshot.snapshot_sha256,
            "snapshot_mle_report_sha256": "a" * 64,
            "snapshot_prefix_measurement_log_sha256": "b" * 64,
            "current_measurement_log_sha256": log.measurement_log_sha256,
            "current_covered_records_sha256": measurement_records_sha256(log),
            "snapshot_covered_station_boundaries_sha256": "c" * 64,
        },
        "safety": {
            "future_only": True,
            "snapshot_parameters_frozen": True,
            "no_refit": True,
            "truth_used": False,
        },
    }
    return log, snapshot, payload


def test_future_candidate_score_binds_exact_snapshot_and_current_suffix(
    measurement_log_path: Path, tmp_path: Path
) -> None:
    log, snapshot, payload = _artifacts(measurement_log_path, tmp_path)
    path = write_json_atomic(tmp_path / "score.json", payload)
    info = validate_future_candidate_score(
        path,
        expected_snapshot=snapshot,
        expected_current_log=log,
        expected_current_covered_records_sha256=measurement_records_sha256(log),
    )
    assert info.future_step_ids == tuple(range(3, 12))
    assert info.payload["safety"]["no_refit"] is True  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["future_step_ids"].__setitem__(0, 2),
            "strictly after",
        ),
        (
            lambda payload: payload["candidates"][0].__setitem__(
                "snapshot_candidate_id", "different"
            ),
            "snapshot clusters",
        ),
        (
            lambda payload: payload["hashes"].__setitem__(
                "current_covered_records_sha256", "f" * 64
            ),
            "covered-record digest",
        ),
    ],
)
def test_future_candidate_score_rejects_identity_or_causality_drift(
    measurement_log_path: Path,
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    log, snapshot, payload = _artifacts(measurement_log_path, tmp_path)
    broken = copy.deepcopy(payload)
    mutate(broken)
    path = write_json_atomic(tmp_path / f"broken-{message.replace(' ', '-')}.json", broken)
    with pytest.raises(ContractError, match=message):
        validate_future_candidate_score(
            path,
            expected_snapshot=snapshot,
            expected_current_log=log,
            expected_current_covered_records_sha256=measurement_records_sha256(log),
        )
