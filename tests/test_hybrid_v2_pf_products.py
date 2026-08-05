from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orchestrator.contracts import validate_measurement_log
from orchestrator.errors import DataReuseError
from orchestrator.hybrid_v2 import load_pf_spectral_predictions


def test_pf_spectral_predictions_are_exactly_pre_update(
    benchmark_v2_output: Path,
    tmp_path: Path,
) -> None:
    log = validate_measurement_log(
        benchmark_v2_output.parent / "measurement-log-v2"
    )
    count, bins = np.asarray(log.arrays["spectrum_counts"]).shape
    path = tmp_path / "pf_spectral_predictions.npz"
    np.savez(
        path,
        step_id=np.arange(count, dtype=np.int64),
        prediction_data_cutoff_step=np.arange(-1, count - 1, dtype=np.int64),
        predicted_spectra=np.full((count, bins), 1.5, dtype=np.float64),
    )

    info = load_pf_spectral_predictions(path, measurement_log=log, record_count=count)

    assert info.predicted_spectra.shape == (count, bins)


def test_pf_spectral_predictions_reject_post_update_prediction(
    benchmark_v2_output: Path,
    tmp_path: Path,
) -> None:
    log = validate_measurement_log(
        benchmark_v2_output.parent / "measurement-log-v2"
    )
    count, bins = np.asarray(log.arrays["spectrum_counts"]).shape
    path = tmp_path / "pf_spectral_predictions.npz"
    np.savez(
        path,
        step_id=np.arange(count, dtype=np.int64),
        prediction_data_cutoff_step=np.arange(count, dtype=np.int64),
        predicted_spectra=np.ones((count, bins), dtype=np.float64),
    )

    with pytest.raises(DataReuseError, match="predate"):
        load_pf_spectral_predictions(path, measurement_log=log, record_count=count)
