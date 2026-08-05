"""Validation of PF-owned pre-update raw-spectrum prediction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orchestrator.contracts import MeasurementLogInfo
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import sha256_file


@dataclass(frozen=True, slots=True)
class PFSpectralPredictions:
    """Opaque-to-Orchestrator PF predictions aligned with causal log rows."""

    path: Path
    step_ids: tuple[int, ...]
    prediction_data_cutoff_steps: tuple[int, ...]
    predicted_spectra: np.ndarray
    artifact_sha256: str


def load_pf_spectral_predictions(
    path: str | Path,
    *,
    measurement_log: MeasurementLogInfo,
    record_count: int,
) -> PFSpectralPredictions:
    """Load an exact prefix of pre-assimilation PF predictive means."""
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise ContractError("PF spectral prediction artifact is missing or a symlink.")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {
                "step_id",
                "prediction_data_cutoff_step",
                "predicted_spectra",
            }:
                raise ContractError("PF spectral prediction NPZ has an unexpected array set.")
            steps = np.asarray(archive["step_id"])
            cutoffs = np.asarray(archive["prediction_data_cutoff_step"])
            predicted = np.asarray(archive["predicted_spectra"])
    except (OSError, ValueError) as exc:
        raise ContractError(f"Could not read PF spectral prediction artifact: {exc}") from exc
    count = int(record_count)
    bins = int(measurement_log.arrays["spectrum_counts"].shape[1])
    if steps.dtype != np.int64 or steps.shape != (count,):
        raise ContractError("PF spectral prediction step_id has wrong dtype or shape.")
    if cutoffs.dtype != np.int64 or cutoffs.shape != (count,):
        raise ContractError("PF spectral prediction cutoff has wrong dtype or shape.")
    if predicted.dtype != np.float64 or predicted.shape != (count, bins):
        raise ContractError("PF predicted_spectra has wrong dtype or shape.")
    expected_steps = np.asarray(measurement_log.step_ids[:count], dtype=np.int64)
    if not np.array_equal(steps, expected_steps):
        raise DataReuseError("PF spectral predictions do not match the exact log prefix.")
    if np.any(cutoffs >= steps) or np.any(cutoffs < -1):
        raise DataReuseError("Every PF spectral prediction must predate its observation.")
    if not np.all(np.isfinite(predicted)) or np.any(predicted < 0):
        raise ContractError("PF predicted spectra must be finite and nonnegative.")
    immutable = np.array(predicted, copy=True)
    immutable.setflags(write=False)
    return PFSpectralPredictions(
        path=source,
        step_ids=tuple(int(value) for value in steps),
        prediction_data_cutoff_steps=tuple(int(value) for value in cutoffs),
        predicted_spectra=immutable,
        artifact_sha256=sha256_file(source),
    )


__all__ = ["PFSpectralPredictions", "load_pf_spectral_predictions"]
