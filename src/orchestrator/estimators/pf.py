"""Strict sequential full-spectrum particle filter owned by this repository."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orchestrator.errors import ContractError
from orchestrator.hashing import canonical_json_bytes, json_safe, load_json, sha256_bytes

from .context import EstimatorContext
from .forward import particle_log_likelihood, predict_particle_spectra


@dataclass(frozen=True, slots=True)
class ParticleFilterConfig:
    num_particles: int = 128
    max_sources_per_isotope: int = 2
    initial_presence_probability: float = 0.65
    expected_sources_per_isotope: float = 1.0
    strength_min_cps_1m: float = 100.0
    strength_max_cps_1m: float = 1.0e7
    resample_ess_fraction: float = 0.5
    patch_edge_m: float = 1.0
    use_gpu: bool = False

    @classmethod
    def from_path(cls, path: str | Path) -> ParticleFilterConfig:
        payload = load_json(path)
        defaults = cls()
        values = {
            "num_particles": payload.get("num_particles", defaults.num_particles),
            "max_sources_per_isotope": payload.get(
                "max_sources_per_isotope", defaults.max_sources_per_isotope
            ),
            "initial_presence_probability": payload.get(
                "initial_presence_probability", defaults.initial_presence_probability
            ),
            "expected_sources_per_isotope": payload.get(
                "expected_sources_per_isotope", defaults.expected_sources_per_isotope
            ),
            "strength_min_cps_1m": payload.get(
                "strength_min_cps_1m", defaults.strength_min_cps_1m
            ),
            "strength_max_cps_1m": payload.get(
                "strength_max_cps_1m", defaults.strength_max_cps_1m
            ),
            "resample_ess_fraction": payload.get(
                "resample_ess_fraction", defaults.resample_ess_fraction
            ),
            "patch_edge_m": payload.get(
                "patch_edge_m",
                payload.get("replay_candidate_spacing_m", defaults.patch_edge_m),
            ),
            "use_gpu": payload.get("use_gpu", defaults.use_gpu),
        }
        return cls(**values)  # type: ignore[arg-type]

    def __post_init__(self) -> None:
        if self.max_sources_per_isotope < 1:
            raise ValueError("PF requires at least one source slot.")
        if self.num_particles < self.max_sources_per_isotope + 1:
            raise ValueError(
                "PF requires enough particles to retain every initial cardinality stratum."
            )
        if not 0.0 <= self.initial_presence_probability <= 1.0:
            raise ValueError("initial_presence_probability must lie in [0, 1].")
        if self.expected_sources_per_isotope <= 0.0:
            raise ValueError("expected_sources_per_isotope must be positive.")
        if not 0.0 < self.strength_min_cps_1m < self.strength_max_cps_1m:
            raise ValueError("PF strength bounds must be positive and ordered.")
        if not 0.0 < self.resample_ess_fraction <= 1.0:
            raise ValueError("resample_ess_fraction must lie in (0, 1].")

    def to_dict(self) -> dict[str, object]:
        return {
            "num_particles": self.num_particles,
            "max_sources_per_isotope": self.max_sources_per_isotope,
            "initial_presence_probability": self.initial_presence_probability,
            "expected_sources_per_isotope": self.expected_sources_per_isotope,
            "strength_min_cps_1m": self.strength_min_cps_1m,
            "strength_max_cps_1m": self.strength_max_cps_1m,
            "resample_ess_fraction": self.resample_ess_fraction,
            "patch_edge_m": self.patch_edge_m,
            "use_gpu": self.use_gpu,
        }


@dataclass(slots=True)
class ParticleState:
    chart_ids: np.ndarray
    strengths_cps_1m: np.ndarray
    log_weights: np.ndarray
    processed_record_count: int
    predicted_spectra: np.ndarray
    prediction_cutoffs: np.ndarray
    rng_state: dict[str, object]
    applied_directive_ids: tuple[str, ...] = ()
    trace: tuple[dict[str, object], ...] = ()

    def copy(self) -> ParticleState:
        return ParticleState(
            chart_ids=self.chart_ids.copy(),
            strengths_cps_1m=self.strengths_cps_1m.copy(),
            log_weights=self.log_weights.copy(),
            processed_record_count=int(self.processed_record_count),
            predicted_spectra=self.predicted_spectra.copy(),
            prediction_cutoffs=self.prediction_cutoffs.copy(),
            rng_state=dict(json_safe(self.rng_state)),
            applied_directive_ids=tuple(self.applied_directive_ids),
            trace=tuple(json.loads(json.dumps(row)) for row in self.trace),
        )


def _normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    maximum = float(np.max(values))
    shifted = np.exp(values - maximum)
    total = float(np.sum(shifted))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("PF particle weights collapsed.")
    return shifted / total


class ParticleFilter:
    """Causal station-block PF with no batch optimizer or MLE feedback."""

    def __init__(
        self,
        context: EstimatorContext,
        config: ParticleFilterConfig,
        *,
        random_seed: int,
        state: ParticleState | None = None,
    ) -> None:
        self.context = context
        self.config = config
        self.random_seed = int(random_seed)
        self.rng = np.random.Generator(np.random.Philox(self.random_seed))
        if state is None:
            self.state = self._initial_state()
        else:
            self.state = state.copy()
            self.rng.bit_generator.state = self.state.rng_state
            self._validate_state()

    def _initial_state(self) -> ParticleState:
        n = self.config.num_particles
        isotope_count = len(self.context.isotopes)
        slots = self.config.max_sources_per_isotope
        charts = np.full((n, isotope_count, slots), -1, dtype=np.int64)
        strengths = np.zeros_like(charts, dtype=np.float64)
        areas = np.asarray(self.context.surface_geometry.areas_m2, dtype=np.float64)
        chart_probability = areas / float(np.sum(areas))
        lam = self.config.expected_sources_per_isotope
        positive = np.asarray(
            [lam**k / math.factorial(k) for k in range(1, slots + 1)],
            dtype=np.float64,
        )
        positive /= float(np.sum(positive))
        cardinality_probability = np.concatenate(
            (
                np.asarray([1.0 - self.config.initial_presence_probability]),
                self.config.initial_presence_probability * positive,
            )
        )
        for isotope_index in range(isotope_count):
            cardinality = self.rng.choice(
                slots + 1,
                size=n,
                replace=True,
                p=cardinality_probability,
            )
            # Deterministically retain every model-order stratum at initialization.
            cardinality[: slots + 1] = np.arange(slots + 1, dtype=np.int64)
            for slot in range(slots):
                active_particles = np.flatnonzero(cardinality > slot)
                charts[active_particles, isotope_index, slot] = self.rng.choice(
                    areas.size,
                    size=active_particles.size,
                    replace=True,
                    p=chart_probability,
                )
                log_strength = self.rng.uniform(
                    math.log(self.config.strength_min_cps_1m),
                    math.log(self.config.strength_max_cps_1m),
                    size=active_particles.size,
                )
                strengths[active_particles, isotope_index, slot] = np.exp(log_strength)
        bin_count = self.context.spectrum_counts.shape[1]
        return ParticleState(
            chart_ids=charts,
            strengths_cps_1m=strengths,
            log_weights=np.full(n, -math.log(n), dtype=np.float64),
            processed_record_count=0,
            predicted_spectra=np.zeros((0, bin_count), dtype=np.float64),
            prediction_cutoffs=np.zeros(0, dtype=np.int64),
            rng_state=dict(json_safe(self.rng.bit_generator.state)),
        )

    def _validate_state(self) -> None:
        expected = (
            self.config.num_particles,
            len(self.context.isotopes),
            self.config.max_sources_per_isotope,
        )
        state = self.state
        if state.chart_ids.shape != expected or state.strengths_cps_1m.shape != expected:
            raise ContractError("PF checkpoint state dimensions disagree with its config.")
        if state.log_weights.shape != (expected[0],):
            raise ContractError("PF checkpoint particle weights are malformed.")
        active = state.chart_ids >= 0
        if np.any((~active[..., :-1]) & active[..., 1:]):
            raise ContractError("PF checkpoint source slots must be contiguous.")
        if np.any(state.chart_ids >= self.context.surface_geometry.chart_count):
            raise ContractError("PF checkpoint references an unknown surface chart.")
        if np.any(state.strengths_cps_1m[~active] != 0.0):
            raise ContractError("PF checkpoint inactive source slots must have zero strength.")
        if np.any(state.strengths_cps_1m[active] <= 0.0):
            raise ContractError("PF checkpoint active sources must have positive strength.")
        if not 0 <= state.processed_record_count <= self.context.measurement_log.record_count:
            raise ContractError("PF checkpoint processed-record count is invalid.")
        if state.predicted_spectra.shape != (
            state.processed_record_count,
            self.context.spectrum_counts.shape[1],
        ):
            raise ContractError("PF checkpoint prediction history is malformed.")
        if state.prediction_cutoffs.shape != (state.processed_record_count,):
            raise ContractError("PF checkpoint prediction cutoffs are malformed.")
        if len(state.trace) != state.processed_record_count:
            raise ContractError("PF checkpoint causal trace length is malformed.")

    @property
    def weights(self) -> np.ndarray:
        return _normalized_weights(self.state.log_weights)

    def _predict_rows(self, rows: np.ndarray) -> np.ndarray:
        prediction = predict_particle_spectra(
            self.context,
            chart_ids=self.state.chart_ids,
            strengths_cps_1m=self.state.strengths_cps_1m,
            row_indices=rows,
        ).mean_spectra
        return np.einsum("n,nvb->vb", self.weights, prediction, optimize=True)

    def _systematic_resample(self) -> None:
        weights = self.weights
        n = weights.size
        positions = (float(self.rng.random()) + np.arange(n, dtype=np.float64)) / n
        cumulative = np.cumsum(weights)
        indices = np.searchsorted(cumulative, positions, side="right")
        self.state.chart_ids = self.state.chart_ids[indices].copy()
        self.state.strengths_cps_1m = self.state.strengths_cps_1m[indices].copy()
        self.state.log_weights = np.full(n, -math.log(n), dtype=np.float64)

    def advance(self, stop_after: int) -> ParticleState:
        """Advance causally through complete station blocks ending before stop_after."""
        stop = int(stop_after)
        start = self.state.processed_record_count
        if stop < start or stop > self.context.measurement_log.record_count:
            raise ValueError("stop_after must advance within the MeasurementLog.")
        stations = np.asarray(self.context.measurement_log.arrays["station_id"], dtype=np.int64)
        while start < stop:
            station = int(stations[start])
            end = start + 1
            while end < stop and int(stations[end]) == station:
                end += 1
            if end < self.context.measurement_log.record_count and int(stations[end]) == station:
                raise ContractError("PF checkpoints may be emitted only at station boundaries.")
            rows = np.arange(start, end, dtype=np.int64)
            prior_summary = self.posterior_summary()
            predictive = self._predict_rows(rows)
            likelihood, _ = particle_log_likelihood(
                self.context,
                chart_ids=self.state.chart_ids,
                strengths_cps_1m=self.state.strengths_cps_1m,
                row_indices=rows,
            )
            self.state.log_weights = np.log(self.weights) + likelihood
            normalized = _normalized_weights(self.state.log_weights)
            self.state.log_weights = np.full_like(normalized, -np.inf)
            positive = normalized > 0.0
            self.state.log_weights[positive] = np.log(normalized[positive])
            self.state.predicted_spectra = np.concatenate(
                (self.state.predicted_spectra, predictive), axis=0
            )
            self.state.prediction_cutoffs = np.concatenate(
                (
                    self.state.prediction_cutoffs,
                    np.full(rows.size, start - 1, dtype=np.int64),
                )
            )
            ess = 1.0 / float(np.sum(np.square(self.weights)))
            if ess < self.config.resample_ess_fraction * self.config.num_particles:
                self._systematic_resample()
            start = end
            self.state.processed_record_count = end
            completed_summary = self.posterior_summary()
            trace = list(self.state.trace)
            for index in rows[:-1]:
                trace.append(
                    {
                        "schema_version": 2,
                        "estimator_family": "pure_particle_filter",
                        "step_id": int(self.context.measurement_log.step_ids[int(index)]),
                        "station_id": station,
                        "station_update_complete": False,
                        "isotopes": prior_summary,
                    }
                )
            trace.append(
                {
                    "schema_version": 2,
                    "estimator_family": "pure_particle_filter",
                    "step_id": int(self.context.measurement_log.step_ids[end - 1]),
                    "station_id": station,
                    "station_update_complete": True,
                    "isotopes": completed_summary,
                }
            )
            self.state.trace = tuple(trace)
        self.state.rng_state = dict(json_safe(self.rng.bit_generator.state))
        return self.state.copy()

    def log_prior(self, chart_ids: np.ndarray, strengths: np.ndarray) -> np.ndarray:
        """Return the complete normalized discrete/strength prior per particle."""
        charts = np.asarray(chart_ids, dtype=np.int64)
        values = np.asarray(strengths, dtype=np.float64)
        active = charts >= 0
        areas = np.asarray(self.context.surface_geometry.areas_m2, dtype=np.float64)
        log_chart = np.log(areas / float(np.sum(areas)))
        result = np.zeros(charts.shape[0], dtype=np.float64)
        log_range = math.log(
            self.config.strength_max_cps_1m / self.config.strength_min_cps_1m
        )
        for isotope_index in range(charts.shape[1]):
            cardinality = np.sum(active[:, isotope_index, :], axis=1)
            lam = self.config.expected_sources_per_isotope
            normalizer = sum(
                math.exp(-lam) * lam**k / math.factorial(k)
                for k in range(self.config.max_sources_per_isotope + 1)
            )
            for k in range(self.config.max_sources_per_isotope + 1):
                selected = cardinality == k
                if np.any(selected):
                    result[selected] += math.log(
                        math.exp(-lam) * lam**k / math.factorial(k) / normalizer
                    )
            for slot in range(charts.shape[2]):
                selected = active[:, isotope_index, slot]
                if np.any(selected):
                    q = values[selected, isotope_index, slot]
                    valid = (
                        (q >= self.config.strength_min_cps_1m)
                        & (q <= self.config.strength_max_cps_1m)
                    )
                    indices = np.flatnonzero(selected)
                    result[indices[~valid]] = -np.inf
                    if np.any(valid):
                        valid_indices = indices[valid]
                        result[valid_indices] += (
                            log_chart[charts[valid_indices, isotope_index, slot]]
                            - np.log(q[valid])
                            - math.log(log_range)
                        )
        return result

    def full_log_target(self, chart_ids: np.ndarray, strengths: np.ndarray) -> np.ndarray:
        """Evaluate the exact processed-prefix target by independent station blocks."""
        likelihood = np.zeros(np.asarray(chart_ids).shape[0], dtype=np.float64)
        stations = np.asarray(
            self.context.measurement_log.arrays["station_id"], dtype=np.int64
        )
        start = 0
        while start < self.state.processed_record_count:
            station = int(stations[start])
            end = start + 1
            while (
                end < self.state.processed_record_count
                and int(stations[end]) == station
            ):
                end += 1
            block_likelihood, _ = particle_log_likelihood(
                self.context,
                chart_ids=chart_ids,
                strengths_cps_1m=strengths,
                row_indices=np.arange(start, end, dtype=np.int64),
            )
            likelihood += block_likelihood
            start = end
        return likelihood + self.log_prior(chart_ids, strengths)

    def posterior_summary(self) -> dict[str, object]:
        weights = self.weights
        positions = np.asarray(self.context.surface_geometry.centers_xyz, dtype=np.float64)
        result: dict[str, object] = {}
        for isotope_index, isotope in enumerate(self.context.isotopes):
            active = self.state.chart_ids[:, isotope_index, :] >= 0
            cardinalities = np.sum(active, axis=1)
            cardinality_distribution = {
                str(k): float(np.sum(weights[cardinalities == k]))
                for k in range(self.config.max_sources_per_isotope + 1)
            }
            map_cardinality = max(
                range(self.config.max_sources_per_isotope + 1),
                key=lambda k: (cardinality_distribution[str(k)], -k),
            )
            modes: list[dict[str, object]] = []
            for slot in range(map_cardinality):
                selected = (
                    (cardinalities == map_cardinality)
                    & active[:, slot]
                )
                mass = float(np.sum(weights[selected]))
                if mass <= 0.0:
                    continue
                local_weights = weights[selected] / mass
                chart = self.state.chart_ids[selected, isotope_index, slot]
                xyz = positions[chart]
                mean = np.sum(local_weights[:, None] * xyz, axis=0)
                centered = xyz - mean
                covariance = np.einsum(
                    "n,ni,nj->ij", local_weights, centered, centered, optimize=True
                )
                strength = self.state.strengths_cps_1m[selected, isotope_index, slot]
                strength_mean = float(np.sum(local_weights * strength))
                order = np.argsort(strength)
                cumulative = np.cumsum(local_weights[order])
                interval = [
                    float(strength[order[min(np.searchsorted(cumulative, p), order.size - 1)]])
                    for p in (0.05, 0.95)
                ]
                modes.append(
                    {
                        "position_mean_xyz": mean.tolist(),
                        "position_covariance_xyz": covariance.tolist(),
                        "credible_radius_m": float(
                            np.sqrt(max(float(np.max(np.linalg.eigvalsh(covariance))), 0.0))
                        ),
                        "strength_mean_cps_1m": strength_mean,
                        "strength_credible_interval_cps_1m": interval,
                        "posterior_mass": min(mass, 1.0),
                    }
                )
            result[isotope] = {
                "map_cardinality": int(map_cardinality),
                "cardinality_distribution": cardinality_distribution,
                "modes": modes,
            }
        return result

    def state_identity(self) -> str:
        payload = b"\0".join(
            (
                np.ascontiguousarray(self.state.chart_ids).tobytes(),
                np.ascontiguousarray(self.state.strengths_cps_1m).tobytes(),
                np.ascontiguousarray(self.state.log_weights).tobytes(),
                canonical_json_bytes(self.state.rng_state),
                canonical_json_bytes(list(self.state.applied_directive_ids)),
                canonical_json_bytes(list(self.state.trace)),
            )
        )
        return sha256_bytes(payload)


__all__ = ["ParticleFilter", "ParticleFilterConfig", "ParticleState"]
