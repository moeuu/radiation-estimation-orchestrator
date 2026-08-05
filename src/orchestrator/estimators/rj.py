"""PF-owned paired birth/death reversible-jump kernel for verified MLE regions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orchestrator.contracts import (
    PFCheckpointInfo,
    PFRJDirectiveInfo,
    PFRJReceiptInfo,
    validate_pf_checkpoint_v1,
    validate_pf_rj_receipt_v1,
)
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import (
    canonical_json_bytes,
    json_safe,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
)

from .artifacts import load_checkpoint_state, repository_commit, save_particle_state
from .context import EstimatorContext, load_estimator_context
from .pf import ParticleFilter, ParticleFilterConfig


@dataclass(frozen=True, slots=True)
class _BirthRegion:
    candidate_id: str
    isotope: str
    isotope_index: int
    weight: float
    chart_probability: np.ndarray
    strength_center: float


@dataclass(frozen=True, slots=True)
class _DeathChoice:
    particle: int
    isotope_index: int
    slot: int


def _region_chart_probabilities(
    *,
    centers: np.ndarray,
    areas: np.ndarray,
    kinds: tuple[str, ...],
    region: dict[str, object],
) -> np.ndarray:
    centroid = np.asarray(region["centroid_xyz"], dtype=np.float64)
    covariance = np.asarray(region["covariance_xyz"], dtype=np.float64)
    allowed = {str(value) for value in region["surface_kinds"]}  # type: ignore[union-attr]
    inverse = np.linalg.pinv(covariance + np.eye(3) * 1.0e-9, hermitian=True)
    delta = centers - centroid
    exponent = -0.5 * np.einsum("ci,ij,cj->c", delta, inverse, delta, optimize=True)
    exponent -= float(np.max(exponent))
    values = areas * np.exp(exponent)
    values *= np.asarray([kind in allowed for kind in kinds], dtype=np.float64)
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        compatible = np.asarray([kind in allowed for kind in kinds], dtype=bool)
        if not np.any(compatible):
            raise ContractError("RJ birth region names no compatible runtime surface.")
        distance = np.linalg.norm(delta, axis=1)
        chart = int(np.flatnonzero(compatible)[np.argmin(distance[compatible])])
        values = np.zeros_like(areas)
        values[chart] = 1.0
        return values
    return values / total


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _truncated_lognormal_log_density(
    value: float,
    location: float,
    sigma: float,
    lower: float,
    upper: float,
) -> float:
    if value < lower or value > upper or location <= 0.0 or sigma <= 0.0:
        return -math.inf
    residual = (math.log(value) - math.log(location)) / sigma
    lower_z = (math.log(lower) - math.log(location)) / sigma
    upper_z = (math.log(upper) - math.log(location)) / sigma
    normalization = _normal_cdf(upper_z) - _normal_cdf(lower_z)
    if normalization <= 0.0:
        return -math.inf
    return (
        -math.log(value * sigma * math.sqrt(2.0 * math.pi))
        - 0.5 * residual**2
        - math.log(normalization)
    )


def _regions(
    context: EstimatorContext,
    payload: list[dict[str, object]],
) -> tuple[_BirthRegion, ...]:
    geometry = context.surface_geometry
    raw_weights = np.asarray(
        [float(region["candidate_weight"]) for region in payload], dtype=np.float64
    )
    raw_weights /= float(np.sum(raw_weights))
    result: list[_BirthRegion] = []
    for raw, weight in zip(payload, raw_weights, strict=True):
        isotope = str(raw["isotope"])
        try:
            isotope_index = context.isotopes.index(isotope)
        except ValueError as exc:
            raise ContractError("RJ directive isotope is absent from the PF state.") from exc
        result.append(
            _BirthRegion(
                candidate_id=str(raw["candidate_id"]),
                isotope=isotope,
                isotope_index=isotope_index,
                weight=float(weight),
                chart_probability=_region_chart_probabilities(
                    centers=np.asarray(geometry.centers_xyz, dtype=np.float64),
                    areas=np.asarray(geometry.areas_m2, dtype=np.float64),
                    kinds=geometry.kinds,
                    region=raw,
                ),
                strength_center=float(raw["integrated_strength_cps_1m"]),
            )
        )
    return tuple(result)


def _available_regions(
    chart_ids: np.ndarray, regions: tuple[_BirthRegion, ...]
) -> tuple[_BirthRegion, ...]:
    return tuple(
        region
        for region in regions
        if np.any(np.any(chart_ids[:, region.isotope_index, :] < 0, axis=1))
    )


def _normalized_available_weights(regions: tuple[_BirthRegion, ...]) -> np.ndarray:
    values = np.asarray([region.weight for region in regions], dtype=np.float64)
    return values / float(np.sum(values))


def _birth_transition_density(
    *,
    chart_ids: np.ndarray,
    regions: tuple[_BirthRegion, ...],
    particle: int,
    isotope_index: int,
    chart: int,
    strength: float,
    config: ParticleFilterConfig,
    strength_sigma: float,
) -> tuple[float, str]:
    """Return the marginal birth proposal density and dominant region ID."""
    available = _available_regions(chart_ids, regions)
    if not available:
        return 0.0, "none"
    available_weights = _normalized_available_weights(available)
    eligible = np.flatnonzero(np.any(chart_ids[:, isotope_index, :] < 0, axis=1))
    if particle not in eligible:
        return 0.0, "none"
    contributions: list[tuple[float, str]] = []
    for region, region_weight in zip(available, available_weights, strict=True):
        if region.isotope_index != isotope_index:
            continue
        log_strength = _truncated_lognormal_log_density(
            strength,
            region.strength_center,
            strength_sigma,
            config.strength_min_cps_1m,
            config.strength_max_cps_1m,
        )
        chart_probability = float(region.chart_probability[chart])
        if chart_probability <= 0.0 or not math.isfinite(log_strength):
            continue
        value = (
            float(region_weight)
            / float(eligible.size)
            * chart_probability
            * math.exp(log_strength)
        )
        contributions.append((value, region.candidate_id))
    if not contributions:
        return 0.0, "none"
    return sum(value for value, _ in contributions), max(contributions)[1]


def _last_active_slot(slots: np.ndarray) -> int | None:
    active = np.flatnonzero(np.asarray(slots, dtype=np.int64) >= 0)
    return None if active.size == 0 else int(active[-1])


def _death_choices(
    *,
    chart_ids: np.ndarray,
    strengths: np.ndarray,
    regions: tuple[_BirthRegion, ...],
    config: ParticleFilterConfig,
    strength_sigma: float,
) -> tuple[_DeathChoice, ...]:
    """List last-slot removals whose reverse birth has nonzero density."""
    result: list[_DeathChoice] = []
    for particle in range(chart_ids.shape[0]):
        for isotope_index in range(chart_ids.shape[1]):
            slot = _last_active_slot(chart_ids[particle, isotope_index])
            if slot is None:
                continue
            chart = int(chart_ids[particle, isotope_index, slot])
            strength = float(strengths[particle, isotope_index, slot])
            reduced_charts = chart_ids.copy()
            reduced_charts[particle, isotope_index, slot] = -1
            density, _ = _birth_transition_density(
                chart_ids=reduced_charts,
                regions=regions,
                particle=particle,
                isotope_index=isotope_index,
                chart=chart,
                strength=strength,
                config=config,
                strength_sigma=strength_sigma,
            )
            if density > 0.0:
                result.append(_DeathChoice(particle, isotope_index, slot))
    return tuple(result)


def _move_probabilities(
    *,
    chart_ids: np.ndarray,
    strengths: np.ndarray,
    regions: tuple[_BirthRegion, ...],
    config: ParticleFilterConfig,
    strength_sigma: float,
) -> tuple[float, float, tuple[_DeathChoice, ...]]:
    birth_available = bool(_available_regions(chart_ids, regions))
    deaths = _death_choices(
        chart_ids=chart_ids,
        strengths=strengths,
        regions=regions,
        config=config,
        strength_sigma=strength_sigma,
    )
    death_available = bool(deaths)
    if birth_available and death_available:
        return 0.5, 0.5, deaths
    if birth_available:
        return 1.0, 0.0, deaths
    if death_available:
        return 0.0, 1.0, deaths
    raise ContractError("Exact-RJ directive has no reversible birth or death move.")


def _draw_bounded_lognormal(
    rng: np.random.Generator,
    *,
    location: float,
    sigma: float,
    lower: float,
    upper: float,
) -> float:
    for _ in range(10_000):
        draw = float(np.exp(rng.normal(math.log(location), sigma)))
        if lower <= draw <= upper:
            return draw
    raise RuntimeError("Could not draw a bounded exact-RJ strength auxiliary.")


def apply_exact_rj(
    measurement_log: str | Path,
    *,
    config_path: str | Path,
    checkpoint_in: PFCheckpointInfo,
    directive: PFRJDirectiveInfo,
    output_directory: str | Path,
) -> tuple[PFCheckpointInfo, PFRJReceiptInfo]:
    """Apply one directive-conditioned paired birth/death RJ transition."""
    if checkpoint_in.checkpoint_sha256 != directive.payload["pf_checkpoint_sha256"]:
        raise DataReuseError("Exact-RJ directive is bound to a different PF checkpoint.")
    state = load_checkpoint_state(checkpoint_in)
    directive_id = str(directive.payload["directive_id"])
    if directive_id in state.applied_directive_ids:
        raise DataReuseError("Exact-RJ directive was already applied to this PF state.")
    config_file = Path(config_path).resolve()
    config = ParticleFilterConfig.from_path(config_file)
    context = load_estimator_context(
        measurement_log,
        patch_edge_m=config.patch_edge_m,
        use_gpu=config.use_gpu,
    )
    if context.measurement_log.measurement_log_sha256 != directive.payload[
        "prefix_measurement_log_sha256"
    ]:
        raise DataReuseError("Exact-RJ directive is bound to a different log prefix.")
    estimator = ParticleFilter(
        context,
        config,
        random_seed=int(checkpoint_in.payload["random_seed"]),
        state=state,
    )
    if estimator.state.processed_record_count != context.measurement_log.record_count:
        raise DataReuseError("Exact-RJ PF state does not cover the directive target prefix.")
    raw_regions = [
        dict(value) for value in directive.payload["birth_regions"]  # type: ignore[arg-type]
    ]
    regions = _regions(context, raw_regions)
    strength_sigma = 1.0
    seed = int(
        sha256_bytes(canonical_json_bytes(json_safe(directive.payload)))[:16], 16
    )
    rng = np.random.Generator(np.random.Philox(seed))
    charts = estimator.state.chart_ids
    strengths = estimator.state.strengths_cps_1m
    birth_probability, death_probability, death_choices = _move_probabilities(
        chart_ids=charts,
        strengths=strengths,
        regions=regions,
        config=config,
        strength_sigma=strength_sigma,
    )
    move_kind = "birth" if float(rng.random()) < birth_probability else "death"

    before_charts: np.ndarray
    before_strengths: np.ndarray
    after_charts: np.ndarray
    after_strengths: np.ndarray
    candidate_id: str
    particle: int
    isotope_index: int
    slot: int
    log_forward: float
    log_reverse: float

    if move_kind == "birth":
        available = _available_regions(charts, regions)
        available_weights = _normalized_available_weights(available)
        generated_region = available[int(rng.choice(len(available), p=available_weights))]
        isotope_index = generated_region.isotope_index
        eligible = np.flatnonzero(np.any(charts[:, isotope_index, :] < 0, axis=1))
        particle = int(rng.choice(eligible))
        slot = int(np.flatnonzero(charts[particle, isotope_index] < 0)[0])
        chart = int(
            rng.choice(
                context.surface_geometry.chart_count,
                p=generated_region.chart_probability,
            )
        )
        proposed_strength = _draw_bounded_lognormal(
            rng,
            location=generated_region.strength_center,
            sigma=strength_sigma,
            lower=config.strength_min_cps_1m,
            upper=config.strength_max_cps_1m,
        )
        transition_density, candidate_id = _birth_transition_density(
            chart_ids=charts,
            regions=regions,
            particle=particle,
            isotope_index=isotope_index,
            chart=chart,
            strength=proposed_strength,
            config=config,
            strength_sigma=strength_sigma,
        )
        before_charts = charts[particle : particle + 1].copy()
        before_strengths = strengths[particle : particle + 1].copy()
        after_charts = before_charts.copy()
        after_strengths = before_strengths.copy()
        after_charts[0, isotope_index, slot] = chart
        after_strengths[0, isotope_index, slot] = proposed_strength
        ensemble_after_charts = charts.copy()
        ensemble_after_strengths = strengths.copy()
        ensemble_after_charts[particle, isotope_index, slot] = chart
        ensemble_after_strengths[particle, isotope_index, slot] = proposed_strength
        _, reverse_death_probability, reverse_choices = _move_probabilities(
            chart_ids=ensemble_after_charts,
            strengths=ensemble_after_strengths,
            regions=regions,
            config=config,
            strength_sigma=strength_sigma,
        )
        log_forward = math.log(birth_probability * transition_density)
        log_reverse = math.log(reverse_death_probability / len(reverse_choices))
    else:
        choice = death_choices[int(rng.integers(len(death_choices)))]
        particle, isotope_index, slot = (
            choice.particle,
            choice.isotope_index,
            choice.slot,
        )
        chart = int(charts[particle, isotope_index, slot])
        removed_strength = float(strengths[particle, isotope_index, slot])
        before_charts = charts[particle : particle + 1].copy()
        before_strengths = strengths[particle : particle + 1].copy()
        after_charts = before_charts.copy()
        after_strengths = before_strengths.copy()
        after_charts[0, isotope_index, slot] = -1
        after_strengths[0, isotope_index, slot] = 0.0
        ensemble_after_charts = charts.copy()
        ensemble_after_strengths = strengths.copy()
        ensemble_after_charts[particle, isotope_index, slot] = -1
        ensemble_after_strengths[particle, isotope_index, slot] = 0.0
        reverse_birth_probability, _, _ = _move_probabilities(
            chart_ids=ensemble_after_charts,
            strengths=ensemble_after_strengths,
            regions=regions,
            config=config,
            strength_sigma=strength_sigma,
        )
        transition_density, candidate_id = _birth_transition_density(
            chart_ids=ensemble_after_charts,
            regions=regions,
            particle=particle,
            isotope_index=isotope_index,
            chart=chart,
            strength=removed_strength,
            config=config,
            strength_sigma=strength_sigma,
        )
        log_forward = math.log(death_probability / len(death_choices))
        log_reverse = math.log(reverse_birth_probability * transition_density)

    target_before = float(estimator.full_log_target(before_charts, before_strengths)[0])
    target_after = float(estimator.full_log_target(after_charts, after_strengths)[0])
    cardinality_before = int(np.sum(before_charts[0, isotope_index] >= 0))
    cardinality_proposed = cardinality_before + (1 if move_kind == "birth" else -1)
    log_jacobian = 0.0
    log_alpha = min(
        0.0,
        target_after - target_before + log_reverse - log_forward + log_jacobian,
    )
    uniform = max(float(rng.random()), np.nextafter(0.0, 1.0))
    accepted = math.log(uniform) < log_alpha
    before_artifact_hash = str(checkpoint_in.payload["state_artifact_sha256"])
    if accepted:
        estimator.state.chart_ids[particle] = after_charts[0]
        estimator.state.strengths_cps_1m[particle] = after_strengths[0]
        estimator.state.applied_directive_ids = tuple(
            (*estimator.state.applied_directive_ids, directive_id)
        )
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state_path = save_particle_state(output / "pf_state.npz", estimator.state)
    after_artifact_hash = sha256_file(state_path)
    if not accepted and after_artifact_hash != before_artifact_hash:
        raise RuntimeError("Rejected exact-RJ move changed deterministic PF state bytes.")
    checkpoint_payload = dict(checkpoint_in.payload)
    checkpoint_payload["state_artifact"] = state_path.name
    checkpoint_payload["state_artifact_sha256"] = after_artifact_hash
    checkpoint_payload["pf_repository_commit"] = repository_commit()
    checkpoint_payload["checkpoint_id"] = (
        "pf-checkpoint-"
        f"{sha256_bytes(canonical_json_bytes({'state': after_artifact_hash}))[:20]}"
    )
    checkpoint_path = write_json_atomic(output / "pf_checkpoint.json", checkpoint_payload)
    output_checkpoint = validate_pf_checkpoint_v1(
        checkpoint_path,
        expected_source_run_id=str(context.measurement_log.manifest["run_id"]),
        expected_prefix_measurement_log_sha256=context.measurement_log.measurement_log_sha256,
    )
    receipt_payload = {
        "schema_version": 1,
        "receipt_family": "pf_exact_rj_move",
        "receipt_id": f"pf-rj-receipt-{directive_id}",
        "directive_id": directive_id,
        "directive_sha256": directive.directive_sha256,
        "applied_once": True,
        "data_cutoff_step": int(directive.payload["data_cutoff_step"]),
        "prefix_measurement_log_sha256": directive.payload["prefix_measurement_log_sha256"],
        "pf_checkpoint_before_sha256": checkpoint_in.checkpoint_sha256,
        "move": {
            "kind": move_kind,
            "isotope": context.isotopes[isotope_index],
            "candidate_id": candidate_id,
            "cardinality_before": cardinality_before,
            "cardinality_after": cardinality_proposed if accepted else cardinality_before,
        },
        "target": {
            "log_density_before": target_before,
            "log_density_after": target_after,
            "includes_complete_pf_target": True,
        },
        "proposal": {
            "log_forward_density": log_forward,
            "log_reverse_density": log_reverse,
            "log_abs_jacobian": log_jacobian,
            "dimension_matching_transform": directive.payload["kernel"][  # type: ignore[index]
                "dimension_matching_transform"
            ],
        },
        "acceptance": {
            "log_acceptance_ratio": log_alpha,
            "uniform_draw": uniform,
            "accepted": accepted,
        },
        "state": {
            "before_sha256": before_artifact_hash,
            "after_sha256": after_artifact_hash,
        },
        "safety": {
            "direct_weight_change": False,
            "hard_prune": False,
            "pf_target_preserved": True,
        },
    }
    receipt_path = write_json_atomic(output / "pf_rj_receipt.json", receipt_payload)
    receipt = validate_pf_rj_receipt_v1(
        receipt_path,
        expected_directive=directive,
        expected_output_checkpoint=output_checkpoint,
    )
    return output_checkpoint, receipt


__all__ = ["apply_exact_rj"]
