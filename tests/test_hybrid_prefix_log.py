from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from orchestrator.contracts import validate_measurement_log
from orchestrator.errors import DataReuseError
from orchestrator.hashing import canonical_json_bytes, directory_inventory, sha256_file
from orchestrator.hybrid.prefix import StationBoundarySchedule
from orchestrator.hybrid.prefix_log import (
    build_and_materialize_measurement_prefix,
    materialize_station_marked_log,
    measurement_records_sha256,
)


def test_station_marking_and_prefix_publication_are_deterministic(
    measurement_log_path: Path,
    tmp_path: Path,
) -> None:
    source = validate_measurement_log(measurement_log_path)
    assert measurement_records_sha256(source, record_count=3) == (
        "f57c5e5cc83689dfed4b12310e3b63d27e3e95d0c5d53e0904763879f7430efb"
    )
    schedule = StationBoundarySchedule.from_measurement_log(source)
    marked, rebound = materialize_station_marked_log(
        source,
        schedule,
        tmp_path / "marked",
    )

    rows = [
        json.loads(line)
        for line in (marked.root / "observation_metadata.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    marked_steps = {
        int(row["step_id"]) for row in rows if row["metadata"].get("station_complete") is True
    }
    assert marked_steps == {2, 5, 8, 11}
    assert rebound.schedule_sha256 == schedule.schedule_sha256

    first_prefix, first = build_and_materialize_measurement_prefix(
        marked,
        cutoff_step=2,
        station_boundaries=rebound,
        station_complete_marker=True,
        output_directory=tmp_path / "prefix-a",
    )
    second_prefix, second = build_and_materialize_measurement_prefix(
        marked,
        cutoff_step=2,
        station_boundaries=rebound,
        station_complete_marker=True,
        output_directory=tmp_path / "prefix-b",
    )

    assert first.step_ids == (0, 1, 2)
    assert first.manifest["metadata"]["measurement_log_prefix"] == {  # type: ignore[index]
        "schema_version": 1,
        "source_run_id": source.manifest["run_id"],
        "data_cutoff_step": 2,
        "data_cutoff_station": 0,
        "station_boundary_attestation": "covered_prefix_markers_v1",
        "covered_station_boundaries_sha256": schedule.covered_sha256(cutoff_step=2),
    }
    assert "station_boundary_attestation" not in first.manifest["metadata"]
    assert first.measurement_log_sha256 == second.measurement_log_sha256
    assert directory_inventory(first.root) == directory_inventory(second.root)
    assert first_prefix == second_prefix
    assert first_prefix.covered_records_sha256 == measurement_records_sha256(
        marked,
        record_count=3,
    )


def test_prefix_rejects_a_nonterminal_declared_step(
    measurement_log_path: Path,
    tmp_path: Path,
) -> None:
    source = validate_measurement_log(measurement_log_path)
    schedule = StationBoundarySchedule.from_measurement_log(source)
    marked, rebound = materialize_station_marked_log(
        source,
        schedule,
        tmp_path / "marked",
    )

    with pytest.raises(DataReuseError, match="terminal step"):
        build_and_materialize_measurement_prefix(
            marked,
            cutoff_step=1,
            station_boundaries=rebound,
            station_complete_marker=True,
            output_directory=tmp_path / "invalid-prefix",
        )


def test_prefix_identity_is_independent_of_unseen_future_schedule(
    measurement_log_path: Path,
    tmp_path: Path,
) -> None:
    source = validate_measurement_log(measurement_log_path)
    schedule_a = StationBoundarySchedule.create(
        source_run_id=str(source.manifest["run_id"]),
        station_end_steps=((0, 2), (1, 5), (2, 8), (3, 11)),
    )
    schedule_b = StationBoundarySchedule.create(
        source_run_id=str(source.manifest["run_id"]),
        station_end_steps=((0, 2), (1, 4), (2, 9), (3, 11)),
    )
    marked_a, _ = materialize_station_marked_log(source, schedule_a, tmp_path / "marked-a")
    marked_b, _ = materialize_station_marked_log(source, schedule_b, tmp_path / "marked-b")
    prefix_a, log_a = build_and_materialize_measurement_prefix(
        marked_a,
        cutoff_step=2,
        station_boundaries=schedule_a,
        station_complete_marker=True,
        output_directory=tmp_path / "prefix-schedule-a",
    )
    prefix_b, log_b = build_and_materialize_measurement_prefix(
        marked_b,
        cutoff_step=2,
        station_boundaries=schedule_b,
        station_complete_marker=True,
        output_directory=tmp_path / "prefix-schedule-b",
    )
    assert prefix_a.station_boundary_schedule_sha256 != (prefix_b.station_boundary_schedule_sha256)
    assert prefix_a.covered_station_boundaries_sha256 == (
        prefix_b.covered_station_boundaries_sha256
    )
    assert prefix_a.prefix_measurement_log_sha256 == prefix_b.prefix_measurement_log_sha256
    assert log_a.measurement_log_sha256 == log_b.measurement_log_sha256


def test_prefix_identity_excludes_direct_schedule_keys_and_unreferenced_artifacts(
    measurement_log_path: Path,
    tmp_path: Path,
) -> None:
    sources: list = []
    for label, marker in (("a", "future-a"), ("b", "future-b")):
        root = tmp_path / f"source-{label}"
        shutil.copytree(measurement_log_path, root)
        suffix = root / "suffix-only.json"
        suffix.write_bytes(canonical_json_bytes({"marker": marker}))
        manifest_path = root / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metadata"]["station_boundary_schedule_sha256"] = (
            "a" if label == "a" else "b"
        ) * 64
        manifest["artifact_hashes"]["suffix-only.json"] = sha256_file(suffix)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        sources.append(validate_measurement_log(root))

    prefixes = []
    for index, source in enumerate(sources):
        schedule = StationBoundarySchedule.from_measurement_log(source)
        marked, _ = materialize_station_marked_log(
            source,
            schedule,
            tmp_path / f"marked-extra-{index}",
        )
        _, prefix = build_and_materialize_measurement_prefix(
            marked,
            cutoff_step=2,
            station_boundaries=schedule,
            station_complete_marker=True,
            output_directory=tmp_path / f"prefix-extra-{index}",
        )
        prefixes.append(prefix)

    assert prefixes[0].measurement_log_sha256 == prefixes[1].measurement_log_sha256
    assert not (prefixes[0].root / "suffix-only.json").exists()
    assert "station_boundary_schedule_sha256" not in prefixes[0].manifest["metadata"]
