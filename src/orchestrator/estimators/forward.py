"""Runtime-authenticated full-spectrum forward evaluation for local estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from measurement.continuous_kernels import LineTransportComponents
from spectrum.additive_scatter import physical_scatter_basis_numpy

from .context import EstimatorContext


@dataclass(frozen=True, slots=True)
class SpectralPrediction:
    mean_spectra: np.ndarray
    total_line_contributions: np.ndarray
    uncollided_line_contributions: np.ndarray
    transport_features: np.ndarray


def _row_arrays(context: EstimatorContext, row_indices: np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = context.measurement_log.arrays
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if rows.size == 0 or np.any(rows < 0) or np.any(rows >= context.measurement_log.record_count):
        raise ValueError("row_indices must select at least one valid observation row.")
    return (
        np.asarray(arrays["detector_pose_xyz"], dtype=np.float64)[rows],
        np.asarray(arrays["fe_orientation_index"], dtype=np.int64)[rows],
        np.asarray(arrays["pb_orientation_index"], dtype=np.int64)[rows],
        np.asarray(arrays["live_time_s"], dtype=np.float64)[rows],
    )


def predict_particle_spectra(
    context: EstimatorContext,
    *,
    chart_ids: np.ndarray,
    strengths_cps_1m: np.ndarray,
    row_indices: np.ndarray,
) -> SpectralPrediction:
    """Predict spectra for particles shaped ``N x isotope x source-slot``."""
    detectors, fe_indices, pb_indices, live_times = _row_arrays(context, row_indices)
    return predict_particle_spectra_for_actions(
        context,
        chart_ids=chart_ids,
        strengths_cps_1m=strengths_cps_1m,
        detector_positions_xyz=detectors,
        fe_orientation_indices=fe_indices,
        pb_orientation_indices=pb_indices,
        live_times_s=live_times,
    )


def predict_particle_spectra_for_actions(
    context: EstimatorContext,
    *,
    chart_ids: np.ndarray,
    strengths_cps_1m: np.ndarray,
    detector_positions_xyz: np.ndarray,
    fe_orientation_indices: np.ndarray,
    pb_orientation_indices: np.ndarray,
    live_times_s: np.ndarray,
) -> SpectralPrediction:
    """Predict unobserved actions without fabricating MeasurementLog records."""
    charts = np.asarray(chart_ids, dtype=np.int64)
    strengths = np.asarray(strengths_cps_1m, dtype=np.float64)
    if charts.ndim != 3 or strengths.shape != charts.shape:
        raise ValueError("chart_ids and strengths must share shape (N, I, K).")
    particle_count, isotope_count, slot_count = charts.shape
    if isotope_count != len(context.isotopes) or particle_count <= 0 or slot_count <= 0:
        raise ValueError("Particle state dimensions disagree with the inference context.")
    chart_count = context.surface_geometry.chart_count
    if np.any(charts < -1) or np.any(charts >= chart_count):
        raise ValueError("Particle chart IDs are out of range.")
    active = charts >= 0
    if np.any(~np.isfinite(strengths)) or np.any(strengths < 0.0):
        raise ValueError("Particle strengths must be finite and nonnegative.")
    if np.any(strengths[~active] != 0.0) or np.any(strengths[active] <= 0.0):
        raise ValueError("Active and inactive particle slots have inconsistent strengths.")
    detectors = np.asarray(detector_positions_xyz, dtype=np.float64)
    fe_indices = np.asarray(fe_orientation_indices, dtype=np.int64).reshape(-1)
    pb_indices = np.asarray(pb_orientation_indices, dtype=np.int64).reshape(-1)
    live_times = np.asarray(live_times_s, dtype=np.float64).reshape(-1)
    if detectors.ndim != 2 or detectors.shape[1] != 3:
        raise ValueError("Action detector positions must be shaped V x 3.")
    if any(value.shape != (detectors.shape[0],) for value in (fe_indices, pb_indices, live_times)):
        raise ValueError("Action shield indices and live times must match detector positions.")
    if (
        np.any(~np.isfinite(detectors))
        or np.any(~np.isfinite(live_times))
        or np.any(live_times <= 0.0)
        or np.any(fe_indices < 0)
        or np.any(fe_indices > 7)
        or np.any(pb_indices < 0)
        or np.any(pb_indices > 7)
    ):
        raise ValueError("Action geometry, shield indices or live times are invalid.")
    view_count = detectors.shape[0]
    source_count = isotope_count * slot_count
    line_rows = tuple(context.spectral_model.line_identity)
    line_count = len(line_rows)
    total = np.zeros((particle_count, view_count, source_count, line_count), dtype=np.float64)
    uncollided = np.zeros_like(total)
    features = np.zeros((*total.shape, 4), dtype=np.float64)
    slot_grid = np.arange(slot_count, dtype=np.int64)
    particle_grid = np.arange(particle_count, dtype=np.int64)

    for isotope_index, isotope in enumerate(context.isotopes):
        global_lines = np.asarray(
            [index for index, line in enumerate(line_rows) if line["isotope"] == isotope],
            dtype=np.int64,
        )
        local_lines = np.asarray(
            [int(line_rows[index]["transport_line_index"]) for index in global_lines],
            dtype=np.int64,
        )
        if global_lines.size == 0:
            raise RuntimeError(f"Full-spectrum model lacks lines for {isotope!r}.")
        flat_charts = charts[:, isotope_index, :].reshape(-1)
        safe_charts = np.where(flat_charts >= 0, flat_charts, 0)
        positions = context.transport_positions_xyz[safe_charts]
        components = context.kernel.line_transport_components_selected_pairs_for_detectors(
            isotope=isotope,
            detector_positions=detectors,
            sources=positions,
            fe_indices=fe_indices,
            pb_indices=pb_indices,
            positive_line_indices=local_lines,
        )
        if not isinstance(components, LineTransportComponents):
            raise RuntimeError("Shared runtime returned an unexpected transport component type.")
        branching = context.kernel.line_branching_weights(isotope, local_lines)
        source_strengths = strengths[:, isotope_index, :].reshape(-1)
        scale = source_strengths[None, :, None] * branching[None, None, :]
        unattenuated_values = components.unattenuated_kernel * scale
        uncollided_values = components.uncollided_kernel * scale
        raw_features = np.stack(
            (
                components.tau_fe,
                components.tau_pb,
                components.tau_obstacle,
                components.distance_m,
            ),
            axis=-1,
        )
        scatter_basis = physical_scatter_basis_numpy(
            tau_fe=components.tau_fe,
            tau_pb=components.tau_pb,
            tau_obstacle=components.tau_obstacle,
            tau_obstacle_compton=components.tau_obstacle_compton,
            distance_m=components.distance_m,
            energy_keV=np.asarray(
                [line_rows[index]["energy_keV"] for index in global_lines], dtype=np.float64
            )[None, None, :],
            mu_fe_cm_inv=np.asarray(
                [line_rows[index]["mu_fe_cm_inv"] for index in global_lines], dtype=np.float64
            )[None, None, :],
            mu_pb_cm_inv=np.asarray(
                [line_rows[index]["mu_pb_cm_inv"] for index in global_lines], dtype=np.float64
            )[None, None, :],
        )
        additive = context.spectral_model.additive_scatter_response
        if additive is None:
            total_values = uncollided_values
            corrected_uncollided = uncollided_values
        else:
            total_values = additive.total_kernel_numpy(
                unattenuated_values, uncollided_values, scatter_basis
            )
            corrected_uncollided = additive.corrected_uncollided_kernel_numpy(
                uncollided_values, scatter_basis
            )
        for slot_index in slot_grid:
            source_axis = isotope_index * slot_count + int(slot_index)
            flat_indices = particle_grid * slot_count + int(slot_index)
            total[:, :, source_axis, global_lines] = np.transpose(
                total_values[:, flat_indices, :], (1, 0, 2)
            )
            uncollided[:, :, source_axis, global_lines] = np.transpose(
                corrected_uncollided[:, flat_indices, :], (1, 0, 2)
            )
            features[:, :, source_axis, global_lines, :] = np.transpose(
                raw_features[:, flat_indices, :, :], (1, 0, 2, 3)
            )
    mean = context.spectral_model.predict_mean_numpy(total, uncollided, features, live_times)
    return SpectralPrediction(
        mean_spectra=np.asarray(mean, dtype=np.float64),
        total_line_contributions=total,
        uncollided_line_contributions=uncollided,
        transport_features=features,
    )


def particle_log_likelihood(
    context: EstimatorContext,
    *,
    chart_ids: np.ndarray,
    strengths_cps_1m: np.ndarray,
    row_indices: np.ndarray,
) -> tuple[np.ndarray, SpectralPrediction]:
    prediction = predict_particle_spectra(
        context,
        chart_ids=chart_ids,
        strengths_cps_1m=strengths_cps_1m,
        row_indices=row_indices,
    )
    rows = np.asarray(row_indices, dtype=np.int64)
    observed = context.spectrum_counts[rows]
    live = np.asarray(context.measurement_log.arrays["live_time_s"], dtype=np.float64)[rows]
    values = context.spectral_model.log_likelihood_numpy(
        observed,
        prediction.total_line_contributions,
        prediction.uncollided_line_contributions,
        prediction.transport_features,
        live,
    )
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
        raise RuntimeError("Full-spectrum likelihood returned invalid particle values.")
    return result, prediction


__all__ = [
    "SpectralPrediction",
    "particle_log_likelihood",
    "predict_particle_spectra",
    "predict_particle_spectra_for_actions",
]
