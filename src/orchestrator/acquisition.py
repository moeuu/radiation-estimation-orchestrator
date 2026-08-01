"""Estimator-neutral acquisition through the shared simulation runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from runtime.session import run_acquisition_plan


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Identify one immutable raw full-spectrum MeasurementLog v2 bundle."""

    measurement_log_path: Path
    measurement_log_sha256: str
    record_count: int
    run_id: str


def acquire_measurement_log(plan_path: str | Path) -> AcquisitionResult:
    """Run one private physical plan and return its estimator-safe log identity."""
    log = run_acquisition_plan(plan_path)
    if log.schema_version != 2:
        raise RuntimeError("Shared acquisition must publish MeasurementLog v2.")
    return AcquisitionResult(
        measurement_log_path=log.path.resolve(),
        measurement_log_sha256=log.content_sha256,
        record_count=len(log.records),
        run_id=log.run_id,
    )


__all__ = ["AcquisitionResult", "acquire_measurement_log"]
