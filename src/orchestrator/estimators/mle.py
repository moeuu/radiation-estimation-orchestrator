"""Standalone all-history spectral surface MLE owned by this repository.

The shared runtime supplies the immutable line transport and detector response.
This module owns the statistical inverse problem, regularization, optimization,
clustering and warm-start state.  No PF state or PF-selected candidate domain is
accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from orchestrator.hashing import load_json

from .context import EstimatorContext
from .forward import predict_particle_spectra


@dataclass(frozen=True, slots=True)
class SpectralMLEConfig:
    patch_edge_m: float = 1.0
    l1_penalty: float = 1.0e-4
    tv_penalty: float = 1.0e-4
    tv_smoothing_cps_1m_m2: float = 1.0
    maximum_iterations: int = 250
    convergence_tolerance: float = 1.0e-7
    initial_strength_cps_1m: float = 10.0
    cluster_relative_threshold: float = 0.05
    cluster_absolute_threshold_cps_1m: float = 1.0
    response_candidate_chunk_size: int = 64
    use_gpu: bool = False

    @classmethod
    def from_path(cls, path: str | Path) -> SpectralMLEConfig:
        payload = load_json(path)
        defaults = cls()
        aliases = {
            "patch_edge_m": ("patch_edge_m", "patch_spacing_m"),
            "l1_penalty": ("l1_penalty", "l1_weight"),
            "tv_penalty": ("tv_penalty", "tv_weight"),
            "tv_smoothing_cps_1m_m2": ("tv_smoothing_cps_1m_m2",),
            "maximum_iterations": ("maximum_iterations", "max_iterations"),
            "convergence_tolerance": ("convergence_tolerance", "tolerance"),
            "initial_strength_cps_1m": ("initial_strength_cps_1m",),
            "cluster_relative_threshold": ("cluster_relative_threshold",),
            "cluster_absolute_threshold_cps_1m": (
                "cluster_absolute_threshold_cps_1m",
            ),
            "response_candidate_chunk_size": ("response_candidate_chunk_size",),
            "use_gpu": ("use_gpu",),
        }
        values: dict[str, object] = {}
        for field, names in aliases.items():
            default = getattr(defaults, field)
            values[field] = next((payload[name] for name in names if name in payload), default)
        spacing = values["patch_edge_m"]
        if isinstance(spacing, list):
            if not spacing or any(float(value) <= 0.0 for value in spacing):
                raise ValueError("patch_spacing_m must contain positive values.")
            if not np.allclose(spacing, float(spacing[0]), rtol=0.0, atol=1.0e-12):
                raise ValueError("The local rectangular surface MLE requires isotropic spacing.")
            values["patch_edge_m"] = float(spacing[0])
        if "cluster_threshold_fraction" in payload and "cluster_relative_threshold" not in payload:
            values["cluster_relative_threshold"] = payload["cluster_threshold_fraction"]
        return cls(**values)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        positive = (
            self.patch_edge_m,
            self.tv_smoothing_cps_1m_m2,
            self.convergence_tolerance,
            self.initial_strength_cps_1m,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Spectral MLE positive configuration values are invalid.")
        if self.l1_penalty < 0.0 or self.tv_penalty < 0.0:
            raise ValueError("Spectral MLE penalties must be nonnegative.")
        if self.maximum_iterations < 1 or self.response_candidate_chunk_size < 1:
            raise ValueError("Spectral MLE iteration and chunk counts must be positive.")
        if not 0.0 <= self.cluster_relative_threshold <= 1.0:
            raise ValueError("cluster_relative_threshold must lie in [0, 1].")
        if self.cluster_absolute_threshold_cps_1m < 0.0:
            raise ValueError("cluster_absolute_threshold_cps_1m must be nonnegative.")

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class SpectralResponse:
    source_pre_dead_time_counts: np.ndarray
    background_pre_dead_time_counts: np.ndarray
    candidate_isotope_indices: np.ndarray
    candidate_chart_ids: np.ndarray


@dataclass(frozen=True, slots=True)
class SpectralMLEResult:
    patch_strength_by_isotope: np.ndarray
    density_by_isotope: np.ndarray
    predicted_spectra: np.ndarray
    objective_value: float
    poisson_deviance: float
    iterations: int
    converged: bool
    optimizer_message: str
    hotspot_clusters: tuple[dict[str, object], ...]
    response: SpectralResponse
    warm_start_used: bool


def _poisson_deviance(observed: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(observed, dtype=np.float64)
    mean = np.maximum(np.asarray(predicted, dtype=np.float64), np.finfo(np.float64).tiny)
    positive = y > 0.0
    terms = mean - y
    terms[positive] += y[positive] * np.log(y[positive] / mean[positive])
    return float(2.0 * np.sum(terms))


class SpectralMLE:
    """Nonnegative L1+graph-TV spectral reconstruction over every runtime chart."""

    def __init__(self, context: EstimatorContext, config: SpectralMLEConfig) -> None:
        self.context = context
        self.config = config
        self._response: SpectralResponse | None = None

    def build_response(self) -> SpectralResponse:
        """Build exact pre-dead-time spectral columns using shared-runtime physics."""
        if self._response is not None:
            return self._response
        chart_count = self.context.surface_geometry.chart_count
        isotope_count = len(self.context.isotopes)
        candidate_count = chart_count * isotope_count
        candidate_isotopes = np.repeat(np.arange(isotope_count, dtype=np.int64), chart_count)
        candidate_charts = np.tile(np.arange(chart_count, dtype=np.int64), isotope_count)
        rows = np.arange(self.context.measurement_log.record_count, dtype=np.int64)
        response_chunks: list[np.ndarray] = []
        background: np.ndarray | None = None
        chunk_size = self.config.response_candidate_chunk_size
        for start in range(0, candidate_count, chunk_size):
            stop = min(start + chunk_size, candidate_count)
            size = stop - start
            charts = np.full((size, isotope_count, 1), -1, dtype=np.int64)
            strengths = np.zeros_like(charts, dtype=np.float64)
            local = np.arange(size, dtype=np.int64)
            iso = candidate_isotopes[start:stop]
            charts[local, iso, 0] = candidate_charts[start:stop]
            strengths[local, iso, 0] = 1.0
            prediction = predict_particle_spectra(
                self.context,
                chart_ids=charts,
                strengths_cps_1m=strengths,
                row_indices=rows,
            )
            live = np.asarray(
                self.context.measurement_log.arrays["live_time_s"], dtype=np.float64
            )
            source, local_background = self.context.spectral_model.pre_dead_time_components_numpy(
                prediction.total_line_contributions,
                prediction.uncollided_line_contributions,
                prediction.transport_features,
                live,
            )
            source_array = np.asarray(source, dtype=np.float64)
            background_array = np.asarray(local_background, dtype=np.float64)
            if source_array.shape[:2] != (size, rows.size):
                raise RuntimeError("Shared runtime returned malformed source spectral columns.")
            if background_array.ndim == 3:
                if not np.allclose(background_array, background_array[:1], rtol=0.0, atol=0.0):
                    raise RuntimeError("Shared runtime background unexpectedly depends on source.")
                background_array = background_array[0]
            if background_array.shape != source_array.shape[1:]:
                raise RuntimeError("Shared runtime returned malformed spectral background.")
            if background is None:
                background = background_array.copy()
            elif not np.array_equal(background, background_array):
                raise RuntimeError("Shared runtime background changed between response chunks.")
            response_chunks.append(source_array)
        if background is None:
            raise RuntimeError("Spectral MLE surface dictionary is empty.")
        self._response = SpectralResponse(
            source_pre_dead_time_counts=np.concatenate(response_chunks, axis=0),
            background_pre_dead_time_counts=background,
            candidate_isotope_indices=candidate_isotopes,
            candidate_chart_ids=candidate_charts,
        )
        return self._response

    def _mean_and_gradient_terms(
        self,
        q: np.ndarray,
        response: SpectralResponse,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        columns = response.source_pre_dead_time_counts
        source = np.einsum("c,cvb->vb", q, columns, optimize=True)
        background = response.background_pre_dead_time_counts
        nodes = np.asarray(self.context.spectral_model.rate_scale_nodes, dtype=np.float64)
        weights = np.asarray(self.context.spectral_model.rate_scale_weights, dtype=np.float64)
        live = np.asarray(
            self.context.measurement_log.arrays["live_time_s"], dtype=np.float64
        )
        alpha = float(self.context.spectral_model.dead_time_tau_s) / live
        pre = background[None, :, :] + nodes[:, None, None] * source[None, :, :]
        denominator = 1.0 + alpha[None, :, None] * np.sum(pre, axis=-1, keepdims=True)
        mean = np.sum(weights[:, None, None] * pre / denominator, axis=0)
        return mean, pre, denominator, alpha

    def predict_from_strengths(
        self,
        patch_strength_by_isotope: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the frozen surface estimate with the exact runtime dead-time mean."""
        values = np.asarray(patch_strength_by_isotope, dtype=np.float64)
        expected = (
            len(self.context.isotopes),
            self.context.surface_geometry.chart_count,
        )
        if values.shape != expected or np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Frozen spectral surface strengths are invalid.")
        mean, _, _, _ = self._mean_and_gradient_terms(values.reshape(-1), self.build_response())
        return mean

    def _regularization(self, q: np.ndarray) -> tuple[float, np.ndarray]:
        areas = np.asarray(self.context.surface_geometry.areas_m2, dtype=np.float64)
        chart_count = areas.size
        isotope_count = len(self.context.isotopes)
        strengths = q.reshape(isotope_count, chart_count)
        density = strengths / areas[None, :]
        value = self.config.l1_penalty * float(np.sum(strengths))
        gradient = np.full_like(strengths, self.config.l1_penalty)
        edges = np.asarray(self.context.surface_geometry.adjacency_edges, dtype=np.int64)
        lengths = np.asarray(
            self.context.surface_geometry.shared_edge_lengths_m, dtype=np.float64
        )
        if self.config.tv_penalty > 0.0 and edges.size:
            left = edges[:, 0]
            right = edges[:, 1]
            difference = density[:, left] - density[:, right]
            epsilon = self.config.tv_smoothing_cps_1m_m2
            smooth = np.sqrt(np.square(difference) + epsilon**2) - epsilon
            value += self.config.tv_penalty * float(np.sum(lengths[None, :] * smooth))
            slope = (
                self.config.tv_penalty
                * lengths[None, :]
                * difference
                / np.sqrt(np.square(difference) + epsilon**2)
            )
            for isotope_index in range(isotope_count):
                np.add.at(gradient[isotope_index], left, slope[isotope_index] / areas[left])
                np.add.at(
                    gradient[isotope_index], right, -slope[isotope_index] / areas[right]
                )
        return value, gradient.reshape(-1)

    def _objective_and_gradient(
        self,
        q: np.ndarray,
        response: SpectralResponse,
    ) -> tuple[float, np.ndarray]:
        mean, pre, denominator, alpha = self._mean_and_gradient_terms(q, response)
        observed = self.context.spectrum_counts
        safe_mean = np.maximum(mean, np.finfo(np.float64).tiny)
        objective = float(np.sum(safe_mean - observed * np.log(safe_mean)))
        residual = 1.0 - observed / safe_mean
        columns = response.source_pre_dead_time_counts
        column_totals = np.sum(columns, axis=-1)
        nodes = np.asarray(self.context.spectral_model.rate_scale_nodes, dtype=np.float64)
        weights = np.asarray(self.context.spectral_model.rate_scale_weights, dtype=np.float64)
        coefficient = weights[:, None] * nodes[:, None] / denominator[..., 0]
        first = np.einsum(
            "jv,cvb,vb->c", coefficient, columns, residual, optimize=True
        )
        residual_pre = np.einsum("vb,jvb->jv", residual, pre, optimize=True)
        second_coefficient = (
            weights[:, None]
            * nodes[:, None]
            * alpha[None, :]
            * residual_pre
            / np.square(denominator[..., 0])
        )
        second = np.einsum(
            "jv,cv->c", second_coefficient, column_totals, optimize=True
        )
        regularization, regularization_gradient = self._regularization(q)
        return objective + regularization, first - second + regularization_gradient

    def _clusters(self, strengths: np.ndarray) -> tuple[dict[str, object], ...]:
        geometry = self.context.surface_geometry
        edges = np.asarray(geometry.adjacency_edges, dtype=np.int64)
        adjacency = [set() for _ in range(geometry.chart_count)]
        for left, right in edges.tolist():
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
        areas = np.asarray(geometry.areas_m2, dtype=np.float64)
        centers = np.asarray(geometry.centers_xyz, dtype=np.float64)
        result: list[dict[str, object]] = []
        cluster_id = 0
        for isotope_index, isotope in enumerate(self.context.isotopes):
            values = strengths[isotope_index]
            threshold = max(
                self.config.cluster_absolute_threshold_cps_1m,
                self.config.cluster_relative_threshold * float(np.max(values)),
            )
            active = set(np.flatnonzero(values >= threshold).tolist())
            while active:
                seed = min(active)
                pending = [seed]
                component: list[int] = []
                active.remove(seed)
                while pending:
                    chart = pending.pop()
                    component.append(chart)
                    for neighbor in sorted(adjacency[chart] & active):
                        active.remove(neighbor)
                        pending.append(neighbor)
                ids = np.asarray(sorted(component), dtype=np.int64)
                local_strength = values[ids]
                total = float(np.sum(local_strength))
                centroid = (
                    np.sum(centers[ids] * local_strength[:, None], axis=0) / total
                    if total > 0.0
                    else np.mean(centers[ids], axis=0)
                )
                result.append(
                    {
                        "isotope": isotope,
                        "cluster_id": cluster_id,
                        "patch_ids": ids.tolist(),
                        "centroid_xyz": centroid.tolist(),
                        "integrated_strength_cps_1m": total,
                        "peak_density_cps_1m_m2": float(
                            np.max(local_strength / areas[ids])
                        ),
                        "surface_kinds": sorted({geometry.kinds[index] for index in ids}),
                    }
                )
                cluster_id += 1
        return tuple(result)

    def fit(self, *, warm_start: SpectralMLEResult | np.ndarray | None = None) -> SpectralMLEResult:
        response = self.build_response()
        isotope_count = len(self.context.isotopes)
        chart_count = self.context.surface_geometry.chart_count
        candidate_count = isotope_count * chart_count
        if isinstance(warm_start, SpectralMLEResult):
            initial = np.asarray(warm_start.patch_strength_by_isotope, dtype=np.float64).reshape(-1)
            warm_used = True
        elif warm_start is not None:
            initial = np.asarray(warm_start, dtype=np.float64).reshape(-1)
            warm_used = True
        else:
            initial = np.full(candidate_count, self.config.initial_strength_cps_1m)
            warm_used = False
        if initial.shape != (candidate_count,) or np.any(initial < 0.0):
            raise ValueError("Spectral MLE warm start has incompatible shape or values.")
        optimized: OptimizeResult = minimize(
            fun=lambda values: self._objective_and_gradient(values, response),
            x0=initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * candidate_count,
            options={
                "maxiter": self.config.maximum_iterations,
                "ftol": self.config.convergence_tolerance,
                "gtol": self.config.convergence_tolerance,
                "maxls": 40,
            },
        )
        q = np.maximum(np.asarray(optimized.x, dtype=np.float64), 0.0)
        predicted, _, _, _ = self._mean_and_gradient_terms(q, response)
        strengths = q.reshape(isotope_count, chart_count)
        areas = np.asarray(self.context.surface_geometry.areas_m2, dtype=np.float64)
        return SpectralMLEResult(
            patch_strength_by_isotope=strengths,
            density_by_isotope=strengths / areas[None, :],
            predicted_spectra=predicted,
            objective_value=float(optimized.fun),
            poisson_deviance=_poisson_deviance(self.context.spectrum_counts, predicted),
            iterations=int(optimized.nit),
            converged=bool(optimized.success),
            optimizer_message=str(optimized.message),
            hotspot_clusters=self._clusters(strengths),
            response=response,
            warm_start_used=warm_used,
        )


__all__ = ["SpectralMLE", "SpectralMLEConfig", "SpectralMLEResult", "SpectralResponse"]
