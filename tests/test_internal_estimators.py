from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np
from measurement.continuous_kernels import LineTransportComponents
from measurement.model import EnvironmentConfig
from measurement.surface_charts import SurfaceChartGeometry

from orchestrator.contracts import MeasurementLogInfo
from orchestrator.estimators.context import EstimatorContext
from orchestrator.estimators.forward import predict_particle_spectra
from orchestrator.estimators.future_scoring import _shield_program_groups
from orchestrator.estimators.mle import SpectralMLE, SpectralMLEConfig
from orchestrator.estimators.pf import ParticleFilter, ParticleFilterConfig
from orchestrator.estimators.rj import (
    _birth_transition_density,
    _move_probabilities,
    _regions,
)


class FakeKernel:
    def line_transport_components_selected_pairs_for_detectors(
        self,
        *,
        isotope,
        detector_positions,
        sources,
        fe_indices,
        pb_indices,
        positive_line_indices,
    ):
        del isotope, fe_indices, pb_indices
        distance = np.linalg.norm(
            np.asarray(detector_positions)[:, None, :] - np.asarray(sources)[None, :, :],
            axis=-1,
        )[..., None]
        line_count = len(positive_line_indices)
        distance = np.broadcast_to(distance, (*distance.shape[:-1], line_count)).copy()
        kernel = 1.0 / (1.0 + np.square(distance))
        zeros = np.zeros_like(kernel)
        return LineTransportComponents(
            total_kernel=kernel,
            unattenuated_kernel=kernel,
            uncollided_kernel=kernel,
            tau_fe=zeros,
            tau_pb=zeros,
            tau_obstacle=zeros,
            tau_obstacle_compton=zeros,
            distance_m=distance,
        )

    def line_branching_weights(self, isotope, line_indices):
        del isotope
        return np.ones(len(line_indices), dtype=np.float64)


class FakeSpectralModel:
    line_identity = (
        {
            "isotope": "Cs-137",
            "transport_line_index": 0,
            "energy_keV": 100.0,
            "mu_fe_cm_inv": 0.1,
            "mu_pb_cm_inv": 0.2,
        },
        {
            "isotope": "Co-60",
            "transport_line_index": 0,
            "energy_keV": 200.0,
            "mu_fe_cm_inv": 0.1,
            "mu_pb_cm_inv": 0.2,
        },
    )
    additive_scatter_response = None
    rate_scale_nodes = np.asarray([1.0])
    rate_scale_weights = np.asarray([1.0])

    def __init__(self, *, dead_time_tau_s: float = 0.0) -> None:
        self.dead_time_tau_s = float(dead_time_tau_s)

    def pre_dead_time_components_numpy(self, total, uncollided, features, live):
        del uncollided, features, live
        source = np.sum(total, axis=2)
        background = np.full(source.shape[1:], 0.25, dtype=np.float64)
        return source, background

    def predict_mean_numpy(self, total, uncollided, features, live):
        source, background = self.pre_dead_time_components_numpy(
            total, uncollided, features, live
        )
        return source + background

    def log_likelihood_numpy(self, observed, total, uncollided, features, live):
        mean = self.predict_mean_numpy(total, uncollided, features, live)
        return np.sum(observed[None, :, :] * np.log(mean) - mean, axis=(1, 2))


def _geometry() -> SurfaceChartGeometry:
    vertices = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    return SurfaceChartGeometry(
        centers_xyz=np.mean(vertices, axis=1),
        areas_m2=np.ones(2),
        kinds=("floor", "floor"),
        face_ids=("floor", "floor"),
        normals_xyz=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        local_uv_m=np.asarray([[0.5, 0.5], [1.5, 0.5]]),
        vertices_xyz=vertices,
        adjacency_edges=np.asarray([[0, 1]], dtype=np.int64),
        shared_edge_lengths_m=np.asarray([1.0]),
    )


def _context(
    tmp_path: Path,
    *,
    station_ids: tuple[int, int] = (0, 0),
    dead_time_tau_s: float = 0.0,
) -> EstimatorContext:
    arrays = {
        "step_id": np.asarray([0, 1], dtype=np.int64),
        "station_id": np.asarray(station_ids, dtype=np.int64),
        "detector_pose_xyz": np.asarray([[0.5, 0.5, 1.0], [1.5, 0.5, 1.0]]),
        "fe_orientation_index": np.asarray([0, 1], dtype=np.int64),
        "pb_orientation_index": np.asarray([0, 1], dtype=np.int64),
        "live_time_s": np.ones(2),
        "spectrum_counts": np.asarray([[5, 1], [2, 4]], dtype=np.int64),
    }
    info = MeasurementLogInfo(
        root=tmp_path,
        manifest=MappingProxyType(
            {"schema_version": 2, "record_count": 2, "isotopes": ["Cs-137", "Co-60"]}
        ),
        forward_model_manifest=MappingProxyType({}),
        arrays=MappingProxyType(arrays),
        artifact_inventory=MappingProxyType({}),
        measurement_log_sha256="a" * 64,
    )
    geometry = _geometry()
    return EstimatorContext(
        measurement_log=info,
        runtime_config={},
        environment_payload={},
        environment=EnvironmentConfig(size_x=2.0, size_y=1.0, size_z=1.0),
        obstacle_grid=None,
        surface_geometry=geometry,
        transport_positions_xyz=geometry.centers_xyz,
        kernel=FakeKernel(),
        spectral_model=FakeSpectralModel(  # type: ignore[arg-type]
            dead_time_tau_s=dead_time_tau_s
        ),
    )


def test_local_forward_preserves_particle_action_and_bin_axes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    charts = np.asarray([[[0], [-1]], [[1], [0]]], dtype=np.int64)
    strengths = np.asarray([[[5.0], [0.0]], [[3.0], [2.0]]])

    prediction = predict_particle_spectra(
        context,
        chart_ids=charts,
        strengths_cps_1m=strengths,
        row_indices=np.asarray([0, 1], dtype=np.int64),
    )

    assert prediction.mean_spectra.shape == (2, 2, 2)
    assert np.all(prediction.mean_spectra > 0.0)
    assert prediction.mean_spectra[0, 0, 1] == 0.25


def test_future_scoring_derives_program_from_runtime_fe_pb_sequence() -> None:
    groups = _shield_program_groups(
        [10, 11],
        metadata={10: {}, 11: {}},
        step_to_index={10: 0, 11: 1},
        fe_indices=np.asarray([2, 4], dtype=np.int64),
        pb_indices=np.asarray([3, 5], dtype=np.int64),
        grouping_semantics="metadata_or_fe_pb_sequence",
    )

    assert list(groups.values()) == [[10, 11]]
    assert next(iter(groups)).startswith("fe-pb-sequence-")


def test_strict_pf_checkpoint_trace_is_prefix_causal(tmp_path: Path) -> None:
    context = _context(tmp_path)
    config = ParticleFilterConfig(num_particles=8, max_sources_per_isotope=1)
    pf = ParticleFilter(context, config, random_seed=11)

    state = pf.advance(2)

    assert state.processed_record_count == 2
    assert [row["step_id"] for row in state.trace] == [0, 1]
    assert state.trace[0]["station_update_complete"] is False
    assert state.trace[1]["station_update_complete"] is True
    assert state.prediction_cutoffs.tolist() == [-1, -1]


def test_strict_pf_resume_matches_uninterrupted_state(tmp_path: Path) -> None:
    context = _context(tmp_path, station_ids=(0, 1))
    config = ParticleFilterConfig(num_particles=8, max_sources_per_isotope=1)
    uninterrupted = ParticleFilter(context, config, random_seed=17)
    uninterrupted.advance(2)

    first_stage = ParticleFilter(context, config, random_seed=17)
    checkpoint_state = first_stage.advance(1)
    resumed = ParticleFilter(context, config, random_seed=17, state=checkpoint_state)
    resumed.advance(2)

    assert resumed.state_identity() == uninterrupted.state_identity()
    np.testing.assert_array_equal(resumed.state.chart_ids, uninterrupted.state.chart_ids)
    np.testing.assert_array_equal(
        resumed.state.strengths_cps_1m,
        uninterrupted.state.strengths_cps_1m,
    )
    np.testing.assert_array_equal(resumed.state.log_weights, uninterrupted.state.log_weights)
    assert resumed.state.rng_state == uninterrupted.state.rng_state


def test_spectral_mle_uses_complete_surface_dictionary_and_decreases_objective(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    estimator = SpectralMLE(
        context,
        SpectralMLEConfig(
            patch_edge_m=1.0,
            maximum_iterations=50,
            l1_penalty=0.0,
            tv_penalty=0.0,
        ),
    )
    response = estimator.build_response()
    initial = np.full(4, 10.0)
    initial_objective, _ = estimator._objective_and_gradient(initial, response)

    result = estimator.fit()

    assert response.source_pre_dead_time_counts.shape == (4, 2, 2)
    assert result.patch_strength_by_isotope.shape == (2, 2)
    assert result.objective_value <= initial_objective
    assert result.predicted_spectra.shape == (2, 2)


def test_spectral_mle_dead_time_gradient_matches_finite_difference(tmp_path: Path) -> None:
    context = _context(tmp_path, dead_time_tau_s=0.03)
    estimator = SpectralMLE(
        context,
        SpectralMLEConfig(l1_penalty=1.0e-3, tv_penalty=2.0e-3),
    )
    response = estimator.build_response()
    values = np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    _, analytic = estimator._objective_and_gradient(values, response)
    numeric = np.zeros_like(values)
    epsilon = 1.0e-5
    for index in range(values.size):
        plus = values.copy()
        minus = values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_value, _ = estimator._objective_and_gradient(plus, response)
        minus_value, _ = estimator._objective_and_gradient(minus, response)
        numeric[index] = (plus_value - minus_value) / (2.0 * epsilon)

    np.testing.assert_allclose(analytic, numeric, rtol=2.0e-5, atol=2.0e-5)


def test_exact_rj_birth_and_death_proposal_terms_are_paired(tmp_path: Path) -> None:
    context = _context(tmp_path)
    config = ParticleFilterConfig(num_particles=4, max_sources_per_isotope=2)
    charts = np.full((4, 2, 2), -1, dtype=np.int64)
    strengths = np.zeros_like(charts, dtype=np.float64)
    regions = _regions(
        context,
        [
            {
                "candidate_id": "candidate-cs",
                "isotope": "Cs-137",
                "candidate_weight": 1.0,
                "centroid_xyz": [0.5, 0.5, 0.0],
                "covariance_xyz": [
                    [0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.1],
                ],
                "surface_kinds": ["floor"],
                "integrated_strength_cps_1m": 1000.0,
            }
        ],
    )
    birth_probability, death_probability, _ = _move_probabilities(
        chart_ids=charts,
        strengths=strengths,
        regions=regions,
        config=config,
        strength_sigma=1.0,
    )
    assert birth_probability == 1.0
    assert death_probability == 0.0
    transition, _ = _birth_transition_density(
        chart_ids=charts,
        regions=regions,
        particle=0,
        isotope_index=0,
        chart=0,
        strength=1000.0,
        config=config,
        strength_sigma=1.0,
    )
    assert transition > 0.0

    after_charts = charts.copy()
    after_strengths = strengths.copy()
    after_charts[0, 0, 0] = 0
    after_strengths[0, 0, 0] = 1000.0
    reverse_birth_probability, reverse_death_probability, reverse_choices = (
        _move_probabilities(
            chart_ids=after_charts,
            strengths=after_strengths,
            regions=regions,
            config=config,
            strength_sigma=1.0,
        )
    )
    assert reverse_birth_probability == 0.5
    assert reverse_death_probability == 0.5
    assert len(reverse_choices) == 1
    reverse_transition, _ = _birth_transition_density(
        chart_ids=charts,
        regions=regions,
        particle=0,
        isotope_index=0,
        chart=0,
        strength=1000.0,
        config=config,
        strength_sigma=1.0,
    )
    assert reverse_transition == transition
