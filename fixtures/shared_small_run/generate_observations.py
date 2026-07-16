"""Regenerate the deterministic binary observation fixture.

This script owns only the mechanical NPZ serialization. The isotope counts are
hand-authored contract-smoke values, not truth-forward-simulated observations. Spectra
distribute those values over the production line basis solely to exercise spectral
contracts. Contract and scenario metadata remain reviewable JSON files beside it.
"""

from __future__ import annotations

import argparse
import zipfile
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "measurement_log"
TARGET = ROOT / "observations.npz"
ISOTOPE_ORDER = ("Cs-137", "Co-60", "Eu-154")
PRODUCTION_LINES = {
    "Cs-137": ((662.0, 0.85),),
    "Co-60": ((1173.0, 0.5), (1332.0, 0.5)),
    "Eu-154": (
        (723.3, 0.25),
        (873.2, 0.14),
        (996.3, 0.14),
        (1274.5, 0.45),
        (1494.0, 0.01),
        (1596.5, 0.02),
    ),
}


def spectrum_from_isotope_counts(isotope_counts: np.ndarray) -> np.ndarray:
    """Allocate net detector counts with the production kernel's normalized lines."""
    energy_edges = np.linspace(0.0, 1600.0, 17, dtype=np.float64)
    background = np.asarray(
        [8, 7, 6, 5, 5, 5, 6, 5, 5, 5, 6, 7, 7, 6, 6, 5],
        dtype=np.float64,
    )
    spectra = np.broadcast_to(background, (isotope_counts.shape[0], 16)).copy()
    for isotope_index, isotope in enumerate(ISOTOPE_ORDER):
        lines = PRODUCTION_LINES[isotope]
        total_intensity = sum(max(float(intensity), 0.0) for _, intensity in lines)
        for energy_keV, raw_intensity in lines:
            weight = max(float(raw_intensity), 0.0) / total_intensity
            bin_index = int(np.searchsorted(energy_edges, energy_keV, side="right") - 1)
            if bin_index < 0 or bin_index >= spectra.shape[1]:
                raise ValueError(f"Production line {energy_keV} keV lies outside fixture bins.")
            spectra[:, bin_index] += weight * isotope_counts[:, isotope_index]
    return spectra


def arrays() -> dict[str, np.ndarray]:
    isotope_counts = np.asarray(
        [
            [260.0, 95.0, 150.0],
            [215.0, 120.0, 135.0],
            [175.0, 145.0, 118.0],
            [190.0, 170.0, 125.0],
            [155.0, 205.0, 112.0],
            [130.0, 230.0, 100.0],
            [120.0, 180.0, 245.0],
            [100.0, 160.0, 275.0],
            [85.0, 145.0, 305.0],
            [140.0, 260.0, 105.0],
            [115.0, 295.0, 92.0],
            [95.0, 330.0, 80.0],
        ],
        dtype=np.float64,
    )
    spectra = spectrum_from_isotope_counts(isotope_counts)
    covariance = np.zeros((12, 3, 3), dtype=np.float64)
    for row in range(12):
        diagonal = np.maximum(isotope_counts[row], 1.0) * 1.2
        covariance[row] = np.diag(diagonal)
        for left in range(3):
            for right in range(left + 1, 3):
                value = 0.03 * float(np.sqrt(diagonal[left] * diagonal[right]))
                covariance[row, left, right] = value
                covariance[row, right, left] = value
    poses = np.asarray(
        [
            [0.8, 0.8, 0.45],
            [0.8, 0.8, 0.45],
            [0.8, 0.8, 0.45],
            [4.8, 0.9, 1.10],
            [4.8, 0.9, 1.10],
            [4.8, 0.9, 1.10],
            [2.0, 4.8, 1.80],
            [2.0, 4.8, 1.80],
            [2.0, 4.8, 1.80],
            [5.1, 4.7, 2.55],
            [5.1, 4.7, 2.55],
            [5.1, 4.7, 2.55],
        ],
        dtype=np.float64,
    )
    full_counts_mask = np.ones((12, 3), dtype=np.bool_)
    full_covariance_mask = np.ones((12, 3, 3), dtype=np.bool_)
    return {
        "step_id": np.arange(12, dtype=np.int64),
        "action_id": np.arange(12, dtype=np.int64),
        "station_id": np.repeat(np.arange(4, dtype=np.int64), 3),
        "detector_pose_xyz": poses,
        "detector_quat_wxyz": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (12, 1)),
        "fe_orientation_index": np.asarray([0, 2, 5, 1, 3, 7, 0, 4, 6, 1, 3, 7], dtype=np.int64),
        "pb_orientation_index": np.asarray([0, 4, 7, 6, 1, 3, 7, 2, 5, 1, 6, 0], dtype=np.int64),
        "live_time_s": np.full(12, 10.0, dtype=np.float64),
        "travel_time_s": np.asarray(
            [0.0, 0.0, 0.0, 4.5, 0.0, 0.0, 5.2, 0.0, 0.0, 4.8, 0.0, 0.0],
            dtype=np.float64,
        ),
        "shield_actuation_time_s": np.asarray(
            [0.0, 0.6, 0.6, 0.7, 0.6, 0.6, 0.7, 0.6, 0.6, 0.7, 0.6, 0.6],
            dtype=np.float64,
        ),
        "energy_bin_edges_keV": np.linspace(0.0, 1600.0, 17, dtype=np.float64),
        "spectrum_counts": spectra,
        "spectrum_variance": spectra + 1.0,
        "spectrum_variance_present": np.ones(12, dtype=np.bool_),
        "isotope_counts": isotope_counts,
        "isotope_counts_present": full_counts_mask,
        "isotope_counts_record_present": np.ones(12, dtype=np.bool_),
        "isotope_count_covariance": covariance,
        "isotope_count_covariance_present": full_covariance_mask,
        "isotope_count_covariance_record_present": np.ones(12, dtype=np.bool_),
    }


def write_observations(target: Path) -> Path:
    """Write the deterministic fixture archive without replacing an artifact."""
    if target.exists():
        raise FileExistsError(f"Refusing to replace {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as raw:
        with zipfile.ZipFile(raw, mode="w", compression=zipfile.ZIP_STORED) as archive:
            for name, array in arrays().items():
                buffer = BytesIO()
                np.lib.format.write_array(buffer, array, version=(2, 0), allow_pickle=False)
                entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_STORED
                entry.create_system = 3
                entry.external_attr = 0o600 << 16
                archive.writestr(entry, buffer.getvalue())
    return target


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TARGET)
    args = parser.parse_args(None if argv is None else list(argv))
    write_observations(args.output.resolve())


if __name__ == "__main__":
    main()
