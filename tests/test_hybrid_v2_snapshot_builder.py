"""Spectral snapshot builder tests at the repository subprocess boundary."""

from __future__ import annotations

import json
from pathlib import Path

from orchestrator.contracts import validate_measurement_log, validate_mle_result
from orchestrator.hybrid_v2.snapshot import build_spectral_mle_snapshot_v3


def test_snapshot_builder_hashes_predicted_spectra_and_surface_regions(
    benchmark_v2_output: Path,
    tmp_path: Path,
) -> None:
    manifest = json.loads(
        (benchmark_v2_output / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    log = validate_measurement_log(manifest["measurement_log"]["path"])
    result = validate_mle_result(
        benchmark_v2_output / "results" / "mle_spectral",
        expected_mode="spectral",
    )

    snapshot = build_spectral_mle_snapshot_v3(
        output_path=tmp_path / "snapshot.json",
        prefix_log=log,
        mle_result=result,
        station_boundaries_sha256="a" * 64,
        covered_records_sha256="b" * 64,
    )

    artifacts = snapshot.payload["artifacts"]
    assert isinstance(artifacts, dict)
    assert artifacts["predicted_spectra_shape"] == [12, 16]
    candidates = snapshot.payload["candidates"]
    assert isinstance(candidates, list)
    assert len(candidates) == 3
    assert candidates[1]["surface_kinds"] == ["ceiling"]
