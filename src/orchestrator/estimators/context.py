"""Shared-runtime-backed, truth-free inference context.

The runtime remains the sole owner of detector, shield, obstacle, transport,
spectrum and surface geometry.  This module binds those immutable physical
objects to an estimator-independent MeasurementLog; all statistical state is
owned by the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from measurement.model import EnvironmentConfig
from measurement.observation_model import (
    build_runtime_observation_model,
    continuous_kernel_from_observation_model,
)
from measurement.obstacles import ObstacleGrid
from measurement.source_boundary import surface_transport_positions
from measurement.surface_charts import SurfaceChartGeometry, build_surface_chart_geometry
from spectrum.transport_spectral import (
    GeometryConditionedSpectralModel,
    geometry_conditioned_model_from_runtime_config,
)

from orchestrator.contracts import MeasurementLogInfo, validate_measurement_log
from orchestrator.errors import ContractError
from orchestrator.hashing import load_json


@dataclass(frozen=True, slots=True)
class EstimatorContext:
    """Authenticated physical and observation objects for local estimators."""

    measurement_log: MeasurementLogInfo
    runtime_config: dict[str, object]
    environment_payload: dict[str, object]
    environment: EnvironmentConfig
    obstacle_grid: ObstacleGrid | None
    surface_geometry: SurfaceChartGeometry
    transport_positions_xyz: np.ndarray
    kernel: object
    spectral_model: GeometryConditionedSpectralModel

    @property
    def isotopes(self) -> tuple[str, ...]:
        return self.measurement_log.isotopes

    @property
    def detector_positions_xyz(self) -> np.ndarray:
        return np.asarray(self.measurement_log.arrays["detector_pose_xyz"], dtype=np.float64)

    @property
    def spectrum_counts(self) -> np.ndarray:
        return np.asarray(self.measurement_log.arrays["spectrum_counts"], dtype=np.float64)


def _environment(payload: dict[str, object]) -> EnvironmentConfig:
    try:
        return EnvironmentConfig(
            size_x=float(payload["size_x"]),
            size_y=float(payload["size_y"]),
            size_z=float(payload["size_z"]),
            detector_position=(
                None
                if payload.get("detector_position") is None
                else tuple(float(value) for value in payload["detector_position"])  # type: ignore[arg-type]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("MeasurementLog environment lacks valid room dimensions.") from exc


def _obstacles(payload: dict[str, object]) -> ObstacleGrid | None:
    candidates = [
        payload[name]
        for name in ("obstacle_grid", "obstacle_layout")
        if payload.get(name) is not None
    ]
    if not candidates:
        return None
    if len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise ContractError("MeasurementLog must contain at most one embedded obstacle grid.")
    try:
        return ObstacleGrid.from_dict(dict(candidates[0]))
    except (TypeError, ValueError) as exc:
        raise ContractError("MeasurementLog embedded obstacle grid is invalid.") from exc


def _resolved_forward_config(payload: dict[str, object]) -> dict[str, object]:
    """Prefer a serialized resolved model over its original registry selectors."""
    resolved = dict(payload)
    if isinstance(resolved.get("full_spectrum_generative_model"), dict):
        # Runtime logs retain selection provenance after embedding the fully resolved
        # model. The runtime builder correctly rejects both as user input, so inference
        # consumes the embedded immutable model and drops only the now-redundant
        # selectors.
        for key in (
            "full_spectrum_generative_model_path",
            "full_spectrum_model_registry_path",
            "isotope_experiment_profile",
        ):
            resolved.pop(key, None)
    return resolved


def load_estimator_context(
    measurement_log: str | Path | MeasurementLogInfo,
    *,
    patch_edge_m: float = 1.0,
    use_gpu: bool = False,
) -> EstimatorContext:
    """Build one local inference context from a raw full-spectrum log."""
    info = (
        measurement_log
        if isinstance(measurement_log, MeasurementLogInfo)
        else validate_measurement_log(measurement_log)
    )
    if info.schema_version != 2:
        raise ContractError(
            "Internal PF/MLE requires raw full-spectrum MeasurementLog v2; "
            "v1 remains an archived external-baseline contract."
        )
    if not np.isfinite(patch_edge_m) or patch_edge_m <= 0.0:
        raise ValueError("patch_edge_m must be finite and positive.")
    runtime_config = load_json(info.root / "runtime_config.resolved.json")
    forward_config = _resolved_forward_config(runtime_config)
    environment_payload = load_json(info.root / "environment.json")
    environment = _environment(environment_payload)
    obstacle_grid = _obstacles(environment_payload)
    try:
        observation_model = build_runtime_observation_model(
            forward_config,
            isotopes=info.isotopes,
        )
        kernel = continuous_kernel_from_observation_model(
            observation_model,
            obstacle_grid=obstacle_grid,
            use_gpu=bool(use_gpu),
        )
        spectral_model = geometry_conditioned_model_from_runtime_config(
            forward_config,
            run_root=info.root,
        )
        spectral_model.require_runtime_ready()
        surfaces = build_surface_chart_geometry(
            environment,
            obstacle_grid,
            float(patch_edge_m),
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ContractError(
            f"Shared runtime could not reconstruct inference physics: {exc}"
        ) from exc
    edges = np.asarray(info.arrays["energy_bin_edges_keV"], dtype=np.float64)
    model_axis = np.asarray(spectral_model.energy_axis_keV, dtype=np.float64)
    if edges.shape != (model_axis.size + 1,) or not np.array_equal(edges[:-1], model_axis):
        raise ContractError(
            "MeasurementLog energy bins do not match the authenticated full-spectrum model."
        )
    transport = surface_transport_positions(surfaces.centers_xyz, surfaces.normals_xyz)
    return EstimatorContext(
        measurement_log=info,
        runtime_config=runtime_config,
        environment_payload=environment_payload,
        environment=environment,
        obstacle_grid=obstacle_grid,
        surface_geometry=surfaces,
        transport_positions_xyz=np.asarray(transport, dtype=np.float64),
        kernel=kernel,
        spectral_model=spectral_model,
    )
