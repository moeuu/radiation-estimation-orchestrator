"""Build spectral hybrid-v2 snapshots from validated standalone MLE artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orchestrator.contracts import (
    MeasurementLogInfo,
    MLEResultInfo,
    SpectralMLESnapshotInfo,
    validate_spectral_mle_snapshot_v3,
)
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    write_json_idempotent,
)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return sha256_bytes(
        canonical_json_bytes(
            {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "bytes_sha256": sha256_bytes(array.tobytes(order="C")),
            }
        )
    )


def _cluster_covariance(
    *,
    patch_ids: tuple[int, ...],
    isotope: str,
    result: MLEResultInfo,
) -> list[list[float]]:
    ids = np.asarray(result.arrays["patch_ids"], dtype=np.int64)
    centers = np.asarray(result.arrays["patch_centroids_xyz"], dtype=np.float64)
    isotope_names = tuple(str(value) for value in result.arrays["isotope_names"].tolist())
    strengths = np.asarray(result.arrays["patch_strength_by_isotope"], dtype=np.float64)
    try:
        isotope_index = isotope_names.index(isotope)
    except ValueError as exc:
        raise ContractError(
            f"MLE cluster isotope {isotope!r} is absent from its estimate."
        ) from exc
    index_by_id = {int(patch_id): index for index, patch_id in enumerate(ids)}
    try:
        indices = np.asarray([index_by_id[patch_id] for patch_id in patch_ids], dtype=np.int64)
    except KeyError as exc:
        raise ContractError("MLE cluster references a patch absent from mle_estimate.npz.") from exc
    selected = centers[indices]
    weights = strengths[isotope_index, indices]
    total = float(np.sum(weights))
    if total <= 0:
        weights = np.full(indices.size, 1.0 / indices.size, dtype=np.float64)
    else:
        weights = weights / total
    mean = np.sum(selected * weights[:, None], axis=0)
    centered = selected - mean[None, :]
    covariance = np.einsum("n,ni,nj->ij", weights, centered, centered, optimize=True)
    covariance += np.eye(3, dtype=np.float64) * 1e-12
    return covariance.tolist()


def build_spectral_mle_snapshot_v3(
    *,
    output_path: str | Path,
    prefix_log: MeasurementLogInfo,
    mle_result: MLEResultInfo,
    station_boundaries_sha256: str,
    covered_records_sha256: str,
    warm_start_snapshot: SpectralMLESnapshotInfo | None = None,
) -> SpectralMLESnapshotInfo:
    """Create a hash-bound spectral snapshot without importing MLE implementation code."""
    if prefix_log.schema_version != 2:
        raise ContractError("Spectral hybrid v2 requires MeasurementLog v2.")
    if mle_result.mode != "spectral":
        raise ContractError("Spectral hybrid snapshot requires a spectral MLE result.")
    expected_steps = tuple(range(prefix_log.record_count))
    if prefix_log.step_ids != expected_steps:
        raise DataReuseError("Hybrid-v2 prefix step IDs must be exactly 0..cutoff.")
    diagnostics = mle_result.diagnostics
    if diagnostics.get("converged") is not True:
        raise ContractError("Spectral prefix MLE must converge before creating a snapshot.")
    predicted = mle_result.arrays.get("predicted_spectra")
    if predicted is None:
        raise ContractError("Spectral MLE result must retain predicted_spectra for hybrid v2.")
    predicted_spectra = np.asarray(predicted, dtype=np.float64)
    observed_shape = np.asarray(prefix_log.arrays["spectrum_counts"]).shape
    if predicted_spectra.shape != observed_shape:
        raise ContractError("Spectral MLE predicted_spectra shape differs from its prefix log.")
    if not np.all(np.isfinite(predicted_spectra)) or np.any(predicted_spectra < 0):
        raise ContractError("Spectral MLE predicted_spectra must be finite and nonnegative.")
    predicted_hash = _array_sha256(predicted_spectra)
    candidates: list[dict[str, object]] = []
    for cluster in mle_result.hotspot_clusters:
        patch_ids = tuple(int(value) for value in cluster["patch_ids"])  # type: ignore[arg-type]
        isotope = str(cluster["isotope"])
        descriptor = {
            "mle_result_sha256": mle_result.result_sha256,
            "predicted_spectra_sha256": predicted_hash,
            "cluster_id": int(cluster["cluster_id"]),
            "isotope": isotope,
            "patch_ids": list(patch_ids),
        }
        candidate_id = (
            f"spectral-candidate-{sha256_bytes(canonical_json_bytes(descriptor))[:20]}"
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "isotope": isotope,
                "centroid_xyz": [float(value) for value in cluster["centroid_xyz"]],  # type: ignore[arg-type]
                "covariance_xyz": _cluster_covariance(
                    patch_ids=patch_ids,
                    isotope=isotope,
                    result=mle_result,
                ),
                "integrated_strength_cps_1m": float(
                    cluster["integrated_strength_cps_1m"]
                ),
                "surface_kinds": sorted(str(value) for value in cluster["surface_kinds"]),  # type: ignore[arg-type]
                "patch_ids": list(patch_ids),
                "candidate_spectral_support_sha256": sha256_bytes(
                    canonical_json_bytes(descriptor)
                ),
            }
        )
    cutoff_step = prefix_log.step_ids[-1]
    cutoff_station = prefix_log.station_ids[-1]
    warm_used = warm_start_snapshot is not None
    identity = {
        "milestone": "pf_mle_hybrid_v2",
        "source_run_id": prefix_log.manifest["run_id"],
        "cutoff_step": cutoff_step,
        "prefix_measurement_log_sha256": prefix_log.measurement_log_sha256,
        "mle_result_sha256": mle_result.result_sha256,
    }
    estimate_path = mle_result.root / "mle_estimate.npz"
    diagnostics_path = mle_result.root / "mle_diagnostics.json"
    clusters_path = mle_result.root / "hotspot_clusters.json"
    payload = {
        "schema_version": 3,
        "milestone": "pf_mle_hybrid_v2",
        "snapshot_id": f"spectral-snapshot-{sha256_bytes(canonical_json_bytes(identity))[:20]}",
        "source_run_id": str(prefix_log.manifest["run_id"]),
        "prefix": {
            "measurement_log_schema_version": 2,
            "data_cutoff_step": cutoff_step,
            "data_cutoff_station": cutoff_station,
            "covered_step_ids": list(prefix_log.step_ids),
            "covered_records_sha256": covered_records_sha256,
            "prefix_measurement_log_sha256": prefix_log.measurement_log_sha256,
            "station_boundaries_sha256": station_boundaries_sha256,
        },
        "fit": {
            "estimator_variant": "spectral",
            "converged": True,
            "warm_start": {
                "used": warm_used,
                "source_snapshot_id": (
                    None
                    if warm_start_snapshot is None
                    else warm_start_snapshot.payload["snapshot_id"]
                ),
                "source_result_sha256": (
                    None
                    if warm_start_snapshot is None
                    else warm_start_snapshot.payload["fit"]["mle_result_sha256"]  # type: ignore[index]
                ),
            },
            "mle_result_sha256": mle_result.result_sha256,
        },
        "artifacts": {
            "estimate_npz_sha256": sha256_file(estimate_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "hotspot_clusters_sha256": sha256_file(clusters_path),
            "predicted_spectra_sha256": predicted_hash,
            "predicted_spectra_shape": list(predicted_spectra.shape),
        },
        "candidates": candidates,
        "safety": {
            "uses_only_prefix": True,
            "uses_pf_state": False,
            "uses_pf_candidates": False,
            "candidate_domain": "complete_surface_dictionary",
            "direct_pf_weight_increment": False,
        },
    }
    path = write_json_idempotent(output_path, payload)
    return validate_spectral_mle_snapshot_v3(
        path,
        expected_source_run_id=str(prefix_log.manifest["run_id"]),
        expected_prefix_measurement_log_sha256=prefix_log.measurement_log_sha256,
    )


__all__ = ["build_spectral_mle_snapshot_v3"]
