from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from orchestrator.contracts import validate_measurement_log
from orchestrator.hashing import sha256_file


def test_shared_fixture_binary_is_reproducible(
    repository_root: Path, measurement_log_path: Path, tmp_path: Path
) -> None:
    regenerated = tmp_path / "observations.npz"
    subprocess.run(
        (
            sys.executable,
            str(repository_root / "fixtures" / "shared_small_run" / "generate_observations.py"),
            "--output",
            str(regenerated),
        ),
        check=True,
    )
    assert sha256_file(regenerated) == sha256_file(measurement_log_path / "observations.npz")


def test_shared_fixture_spectra_use_the_manifest_production_lines(
    measurement_log_path: Path,
) -> None:
    log = validate_measurement_log(measurement_log_path)
    spectra = log.arrays["spectrum_counts"]
    isotope_counts = log.arrays["isotope_counts"]
    edges = log.arrays["energy_bin_edges_keV"]
    background = np.asarray(
        [8, 7, 6, 5, 5, 5, 6, 5, 5, 5, 6, 7, 7, 6, 6, 5],
        dtype=np.float64,
    )
    expected = np.broadcast_to(background, spectra.shape).copy()
    table = log.forward_model_manifest["line_mu_by_isotope"]
    assert isinstance(table, dict)
    for isotope_index, isotope in enumerate(log.isotopes):
        rows = table[isotope]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, dict)
            energy = float(row["energy_keV"])
            weight = float(row["weight"])
            bin_index = int(np.searchsorted(edges, energy, side="right") - 1)
            expected[:, bin_index] += weight * isotope_counts[:, isotope_index]
    np.testing.assert_allclose(spectra, expected, rtol=0.0, atol=0.0)

    active_net_bins = set(np.flatnonzero(np.any(spectra != background, axis=0)).tolist())
    assert active_net_bins == {6, 7, 8, 9, 11, 12, 13, 14, 15}
