"""Create once-only future spectral scoring requests."""

from __future__ import annotations

from pathlib import Path

from orchestrator.contracts import (
    FutureSpectralScoreRequestInfo,
    MeasurementLogInfo,
    SpectralMLESnapshotInfo,
    validate_future_spectral_score_request_v1,
)
from orchestrator.errors import DataReuseError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes, write_json_idempotent


def build_future_spectral_score_request_v1(
    *,
    output_path: str | Path,
    snapshot: SpectralMLESnapshotInfo,
    current_log: MeasurementLogInfo,
    previously_scored_step_ids: tuple[int, ...] = (),
    height_tolerance_m: float = 0.05,
) -> FutureSpectralScoreRequestInfo:
    """Select all currently available, not-yet-scored post-cutoff rows."""
    previous = tuple(sorted(set(int(value) for value in previously_scored_step_ids)))
    requested = tuple(
        step
        for step in current_log.step_ids
        if step > snapshot.cutoff_step and step not in set(previous)
    )
    if not requested:
        raise DataReuseError("No new post-cutoff observations are available for scoring.")
    identity = {
        "snapshot_sha256": snapshot.snapshot_sha256,
        "current_log": current_log.measurement_log_sha256,
        "requested": list(requested),
        "previous": list(previous),
    }
    payload = {
        "schema_version": 1,
        "request_id": (
            f"spectral-score-request-{sha256_bytes(canonical_json_bytes(identity))[:20]}"
        ),
        "snapshot_id": snapshot.payload["snapshot_id"],
        "snapshot_sha256": snapshot.snapshot_sha256,
        "snapshot_data_cutoff_step": snapshot.cutoff_step,
        "current_measurement_log_sha256": current_log.measurement_log_sha256,
        "requested_future_step_ids": list(requested),
        "previously_scored_step_ids": list(previous),
        "grouping": {
            "station": "station_id",
            "height": "detector_z_tolerance_group",
            "shield_program": "metadata_or_fe_pb_sequence",
            "height_tolerance_m": float(height_tolerance_m),
        },
        "safety": {
            "future_only": True,
            "snapshot_frozen": True,
            "refit_allowed": False,
            "steps_once_only": True,
        },
    }
    path = write_json_idempotent(output_path, payload)
    return validate_future_spectral_score_request_v1(
        path,
        expected_snapshot=snapshot,
    )


__all__ = ["build_future_spectral_score_request_v1"]
