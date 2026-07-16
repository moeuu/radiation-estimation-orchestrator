"""Truth-gated common point, operational, and surface-MLE metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .adapters import AdapterExecution
from .contracts import MeasurementLogInfo, MLEResultInfo, PFResultInfo, validate_truth
from .hashing import sha256_file


@dataclass(frozen=True, slots=True)
class SourceEstimate:
    isotope: str
    position_xyz: tuple[float, float, float]
    strength_cps_1m: float
    surface_kinds: tuple[str, ...] = ()
    identifier: str = ""


@dataclass(frozen=True, slots=True)
class MatchedPair:
    truth_index: int
    estimate_index: int
    distance_3d_m: float
    distance_xy_m: float
    distance_z_m: float


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=float)))


def _rmse(values: list[float]) -> float | None:
    return None if not values else float(np.sqrt(np.mean(np.square(values))))


def _hungarian(cost: NDArray[np.float64]) -> list[int]:
    """Return minimum-cost column assignments for a square finite matrix."""
    matrix = np.asarray(cost, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Hungarian cost matrix must be square.")
    size = matrix.shape[0]
    if size == 0:
        return []
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Hungarian cost matrix must be finite.")
    u = np.zeros(size + 1, dtype=float)
    v = np.zeros(size + 1, dtype=float)
    p = np.zeros(size + 1, dtype=np.int64)
    way = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        p[0] = row
        min_value = np.full(size + 1, np.inf, dtype=float)
        used = np.zeros(size + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = int(p[column0])
            delta = math.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = matrix[row0 - 1, column - 1] - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = float(min_value[column])
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = int(way[column0])
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        if p[column] != 0:
            assignment[int(p[column]) - 1] = column - 1
    return assignment


def match_sources(
    truth: list[SourceEstimate], estimates: list[SourceEstimate], *, radius_m: float
) -> tuple[MatchedPair, ...]:
    """Maximum-cardinality, minimum-distance isotope-aware gated assignment."""
    if radius_m <= 0:
        raise ValueError("radius_m must be positive.")
    truth_count = len(truth)
    estimate_count = len(estimates)
    if truth_count == 0 or estimate_count == 0:
        return ()
    size = truth_count + estimate_count
    unmatched_cost = float(radius_m) + 1.0
    invalid_cost = 3.0 * unmatched_cost
    cost = np.zeros((size, size), dtype=float)
    cost[:truth_count, :estimate_count] = invalid_cost
    cost[:truth_count, estimate_count:] = unmatched_cost
    cost[truth_count:, :estimate_count] = unmatched_cost
    for truth_index, truth_source in enumerate(truth):
        truth_position = np.asarray(truth_source.position_xyz, dtype=float)
        for estimate_index, estimate in enumerate(estimates):
            if estimate.isotope != truth_source.isotope:
                continue
            distance = float(
                np.linalg.norm(np.asarray(estimate.position_xyz, dtype=float) - truth_position)
            )
            if distance <= radius_m:
                cost[truth_index, estimate_index] = distance
    assignment = _hungarian(cost)
    result: list[MatchedPair] = []
    for truth_index in range(truth_count):
        estimate_index = assignment[truth_index]
        if estimate_index < 0 or estimate_index >= estimate_count:
            continue
        if cost[truth_index, estimate_index] > radius_m:
            continue
        delta = np.asarray(estimates[estimate_index].position_xyz) - np.asarray(
            truth[truth_index].position_xyz
        )
        result.append(
            MatchedPair(
                truth_index=truth_index,
                estimate_index=estimate_index,
                distance_3d_m=float(np.linalg.norm(delta)),
                distance_xy_m=float(np.linalg.norm(delta[:2])),
                distance_z_m=float(abs(delta[2])),
            )
        )
    return tuple(result)


def _truth_sources(payload: dict[str, object]) -> list[SourceEstimate]:
    raw_sources = payload["sources"]
    assert isinstance(raw_sources, list)
    return [
        SourceEstimate(
            isotope=str(source["isotope"]),
            position_xyz=tuple(float(value) for value in source["position_xyz"]),
            strength_cps_1m=float(source["strength_cps_1m"]),
            surface_kinds=(str(source["surface_kind"]),),
            identifier=str(source["source_id"]),
        )
        for source in raw_sources
        if isinstance(source, dict)
    ]


def pf_sources(result: PFResultInfo) -> list[SourceEstimate]:
    """Extract the canonical PF-posterior modes without batch postprocessing."""
    isotopes = result.posterior["isotopes"]
    assert isinstance(isotopes, dict)
    sources: list[SourceEstimate] = []
    for isotope, estimate in isotopes.items():
        assert isinstance(estimate, dict)
        modes = estimate.get("modes")
        if isinstance(modes, list):
            for index, mode in enumerate(modes):
                assert isinstance(mode, dict)
                sources.append(
                    SourceEstimate(
                        isotope=str(isotope),
                        position_xyz=tuple(float(value) for value in mode["position_mean_xyz"]),
                        strength_cps_1m=float(mode["strength_mean_cps_1m"]),
                        identifier=f"{isotope}:pf-mode:{index}",
                    )
                )
            continue
        legacy = estimate.get("sources", [])
        assert isinstance(legacy, list)
        for index, source in enumerate(legacy):
            assert isinstance(source, dict)
            sources.append(
                SourceEstimate(
                    isotope=str(isotope),
                    position_xyz=tuple(float(value) for value in source["position_xyz"]),
                    strength_cps_1m=float(source["integrated_strength_cps_1m"]),
                    identifier=f"{isotope}:legacy-source:{index}",
                )
            )
    return sources


def mle_cluster_sources(result: MLEResultInfo) -> list[SourceEstimate]:
    """Extract surface hotspot clusters as common point-source estimates."""
    return [
        SourceEstimate(
            isotope=str(cluster["isotope"]),
            position_xyz=tuple(float(value) for value in cluster["centroid_xyz"]),  # type: ignore[arg-type]
            strength_cps_1m=float(cluster["integrated_strength_cps_1m"]),
            surface_kinds=tuple(str(value) for value in cluster["surface_kinds"]),  # type: ignore[arg-type]
            identifier=f"cluster:{cluster['cluster_id']}",
        )
        for cluster in result.hotspot_clusters
    ]


def _integrated_strength_metrics(
    truth: list[SourceEstimate],
    estimates: list[SourceEstimate],
    matches: tuple[MatchedPair, ...],
) -> dict[str, object]:
    """Account for matched errors, unmatched mass, and isotope-total errors."""
    matched_truth = {pair.truth_index for pair in matches}
    matched_estimates = {pair.estimate_index for pair in matches}
    matched_absolute = [
        abs(
            estimates[pair.estimate_index].strength_cps_1m - truth[pair.truth_index].strength_cps_1m
        )
        for pair in matches
    ]
    matched_relative = [
        error / max(truth[pair.truth_index].strength_cps_1m, 1e-12)
        for error, pair in zip(matched_absolute, matches, strict=True)
    ]
    unmatched_truth = sum(
        source.strength_cps_1m for index, source in enumerate(truth) if index not in matched_truth
    )
    unmatched_estimate = sum(
        source.strength_cps_1m
        for index, source in enumerate(estimates)
        if index not in matched_estimates
    )
    truth_total = sum(source.strength_cps_1m for source in truth)
    estimate_total = sum(source.strength_cps_1m for source in estimates)
    per_isotope: dict[str, dict[str, float]] = {}
    isotope_total_absolute_error = 0.0
    for isotope in sorted({source.isotope for source in [*truth, *estimates]}):
        isotope_truth = sum(source.strength_cps_1m for source in truth if source.isotope == isotope)
        isotope_estimate = sum(
            source.strength_cps_1m for source in estimates if source.isotope == isotope
        )
        absolute_error = abs(isotope_estimate - isotope_truth)
        isotope_total_absolute_error += absolute_error
        per_isotope[isotope] = {
            "truth_total": float(isotope_truth),
            "estimate_total": float(isotope_estimate),
            "total_absolute_error": float(absolute_error),
            "total_relative_error": float(absolute_error / max(isotope_truth, 1e-12)),
        }
    assignment_absolute_error = sum(matched_absolute) + unmatched_truth + unmatched_estimate
    assignment_count = (
        len(matches) + len(truth) - len(matched_truth) + len(estimates) - len(matched_estimates)
    )
    return {
        "matched_mae": _mean(matched_absolute),
        "matched_mean_relative_error": _mean(matched_relative),
        "matched_absolute_errors": matched_absolute,
        "truth_total": float(truth_total),
        "estimate_total": float(estimate_total),
        "unmatched_truth_strength": float(unmatched_truth),
        "unmatched_estimate_strength": float(unmatched_estimate),
        "isotope_total_absolute_error": float(isotope_total_absolute_error),
        "assignment_absolute_error": float(assignment_absolute_error),
        "assignment_mean_absolute_error": (
            float(assignment_absolute_error / assignment_count) if assignment_count else None
        ),
        # The historical aggregate names now deliberately include every unmatched source.
        "total_absolute_error": float(assignment_absolute_error),
        "total_relative_error": float(assignment_absolute_error / max(truth_total, 1e-12)),
        "per_isotope": per_isotope,
    }


def point_source_metrics(
    truth: list[SourceEstimate],
    estimates: list[SourceEstimate],
    *,
    radius_m: float,
    ceiling_z_m: float,
) -> dict[str, object]:
    """Compute all shared point-source metrics with explicit gated matches."""
    matches = match_sources(truth, estimates, radius_m=radius_m)
    matched_truth = {pair.truth_index for pair in matches}
    matched_estimates = {pair.estimate_index for pair in matches}
    precision = len(matches) / len(estimates) if estimates else (1.0 if not truth else 0.0)
    recall = len(matches) / len(truth) if truth else 1.0
    isotopes = sorted({source.isotope for source in [*truth, *estimates]})
    cardinality_by_isotope: dict[str, object] = {}
    for isotope in isotopes:
        truth_count = sum(source.isotope == isotope for source in truth)
        estimate_count = sum(source.isotope == isotope for source in estimates)
        cardinality_by_isotope[isotope] = {
            "truth": truth_count,
            "estimated": estimate_count,
            "exact_match": truth_count == estimate_count,
        }
    ceiling_truth = {
        index
        for index, source in enumerate(truth)
        if "ceiling" in source.surface_kinds or source.position_xyz[2] >= ceiling_z_m - 1e-9
    }
    ceiling_recovered = ceiling_truth & matched_truth
    distances_3d = [pair.distance_3d_m for pair in matches]
    distances_xy = [pair.distance_xy_m for pair in matches]
    distances_z = [pair.distance_z_m for pair in matches]
    return {
        "match_radius_m": float(radius_m),
        "truth_count": len(truth),
        "estimate_count": len(estimates),
        "matched_count": len(matches),
        "cardinality_exact_match": len(truth) == len(estimates)
        and all(item["exact_match"] for item in cardinality_by_isotope.values()),  # type: ignore[index]
        "cardinality_by_isotope": cardinality_by_isotope,
        "source_precision": float(precision),
        "source_recall": float(recall),
        "false_positive_count": len(estimates) - len(matched_estimates),
        "false_negative_count": len(truth) - len(matched_truth),
        "position_error_3d_m": {
            "mean": _mean(distances_3d),
            "rmse": _rmse(distances_3d),
            "max": None if not distances_3d else max(distances_3d),
            "values": distances_3d,
        },
        "position_error_xy_m": {
            "mean": _mean(distances_xy),
            "rmse": _rmse(distances_xy),
            "max": None if not distances_xy else max(distances_xy),
            "values": distances_xy,
        },
        "position_error_z_m": {
            "mean": _mean(distances_z),
            "rmse": _rmse(distances_z),
            "max": None if not distances_z else max(distances_z),
            "values": distances_z,
        },
        "integrated_strength_error_cps_1m": _integrated_strength_metrics(truth, estimates, matches),
        "ceiling_source_recall": (
            len(ceiling_recovered) / len(ceiling_truth) if ceiling_truth else 1.0
        ),
        "matches": [
            {
                "truth_id": truth[pair.truth_index].identifier,
                "estimate_id": estimates[pair.estimate_index].identifier,
                "isotope": truth[pair.truth_index].isotope,
                "distance_3d_m": pair.distance_3d_m,
            }
            for pair in matches
        ],
    }


def operational_metrics(
    measurement_log: MeasurementLogInfo,
    executions: dict[str, AdapterExecution],
) -> dict[str, object]:
    """Compute measurement and estimator operational costs from logged facts."""
    arrays = measurement_log.arrays
    poses = np.asarray(arrays["detector_pose_xyz"], dtype=float)
    heights = poses[:, 2]
    unique_heights, height_counts = np.unique(heights, return_counts=True)
    return {
        "measurement_count": measurement_log.record_count,
        "unique_xyz_actions": len({tuple(float(value) for value in row) for row in poses}),
        "unique_action_ids": int(np.unique(arrays["action_id"]).size),
        "detector_height_distribution_m": {
            "min": float(np.min(heights)),
            "max": float(np.max(heights)),
            "mean": float(np.mean(heights)),
            "std": float(np.std(heights)),
            "median": float(np.median(heights)),
            "counts": {
                format(float(height), ".12g"): int(count)
                for height, count in zip(unique_heights, height_counts, strict=True)
            },
        },
        "motion_time_s": float(np.sum(arrays["travel_time_s"])),
        "shield_actuation_time_s": float(np.sum(arrays["shield_actuation_time_s"])),
        "live_time_s": float(np.sum(arrays["live_time_s"])),
        "estimators": {
            name: {
                "runtime_s": execution.runtime_s,
                "peak_memory_bytes": execution.peak_memory_bytes,
            }
            for name, execution in sorted(executions.items())
        },
    }


def surface_mle_metrics(
    result: MLEResultInfo,
    truth: list[SourceEstimate],
    *,
    radius_m: float,
) -> dict[str, object]:
    """Compute cluster and surface-mass metrics from complete MLE artifacts."""
    clusters = mle_cluster_sources(result)
    matches = match_sources(truth, clusters, radius_m=radius_m)
    centroid_errors = [pair.distance_3d_m for pair in matches]
    kind_results: list[bool] = []
    for pair in matches:
        truth_kinds = set(truth[pair.truth_index].surface_kinds)
        estimate_kinds = set(clusters[pair.estimate_index].surface_kinds)
        kind_results.append(bool(truth_kinds & estimate_kinds))
    centers = np.asarray(result.arrays["patch_centroids_xyz"], dtype=float)
    strengths = np.asarray(result.arrays["patch_strength_by_isotope"], dtype=float)
    isotope_names = tuple(str(value) for value in result.arrays["isotope_names"].tolist())
    recovered = 0.0
    total = float(np.sum(strengths))
    for isotope_index, isotope in enumerate(isotope_names):
        isotope_truth = [source for source in truth if source.isotope == isotope]
        if not isotope_truth:
            continue
        distances = np.stack(
            [
                np.linalg.norm(centers - np.asarray(source.position_xyz), axis=1)
                for source in isotope_truth
            ],
            axis=0,
        )
        near = np.min(distances, axis=0) <= radius_m
        recovered += float(np.sum(strengths[isotope_index, near]))
    truth_total = sum(source.strength_cps_1m for source in truth)
    diagnostics = result.diagnostics["diagnostics"]
    assert isinstance(diagnostics, dict)
    return {
        "hotspot_cluster_centroid_error_m": {
            "mean": _mean(centroid_errors),
            "rmse": _rmse(centroid_errors),
            "values": centroid_errors,
        },
        "integrated_cluster_strength_error_cps_1m": _integrated_strength_metrics(
            truth, clusters, matches
        ),
        "surface_kind_classification_accuracy": (
            sum(kind_results) / len(kind_results) if kind_results else None
        ),
        "mass_recovered_near_truth": {
            "radius_m": float(radius_m),
            "strength_cps_1m": float(recovered),
            "fraction_of_estimated_mass": float(recovered / total) if total > 0 else 0.0,
            "fraction_of_truth_strength": float(recovered / truth_total)
            if truth_total > 0
            else 1.0,
        },
        "held_out_deviance": diagnostics.get("held_out_poisson_deviance"),
    }


def evaluate_benchmark(
    *,
    measurement_log: MeasurementLogInfo,
    truth_path: str | Path,
    pf_result: PFResultInfo,
    mle_count_result: MLEResultInfo,
    mle_spectral_result: MLEResultInfo,
    executions: dict[str, AdapterExecution],
) -> dict[str, object]:
    """Open truth only in this evaluation entry point and compute every metric."""
    truth_payload = validate_truth(
        truth_path, expected_run_id=str(measurement_log.manifest["run_id"])
    )
    truth = _truth_sources(truth_payload)
    radius = float(truth_payload["match_radius_m"])
    ceiling_z = float(truth_payload["ceiling_z_m"])
    pf_estimates = pf_sources(pf_result)
    count_estimates = mle_cluster_sources(mle_count_result)
    spectral_estimates = mle_cluster_sources(mle_spectral_result)
    return {
        "schema_version": 1,
        "truth_sha256": sha256_file(truth_path),
        "operational": operational_metrics(measurement_log, executions),
        "estimators": {
            "pf_strict": {
                "point_source": point_source_metrics(
                    truth, pf_estimates, radius_m=radius, ceiling_z_m=ceiling_z
                )
            },
            "mle_count": {
                "point_source": point_source_metrics(
                    truth, count_estimates, radius_m=radius, ceiling_z_m=ceiling_z
                ),
                "surface_mle": surface_mle_metrics(mle_count_result, truth, radius_m=radius),
            },
            "mle_spectral": {
                "point_source": point_source_metrics(
                    truth, spectral_estimates, radius_m=radius, ceiling_z_m=ceiling_z
                ),
                "surface_mle": surface_mle_metrics(mle_spectral_result, truth, radius_m=radius),
            },
        },
    }
