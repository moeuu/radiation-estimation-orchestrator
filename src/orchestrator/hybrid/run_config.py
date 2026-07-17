"""Fail-closed configuration for the executable offline PF+MLE controller."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from orchestrator.adapters.base import PRODUCTION_ALLOWED_DIRTY_PREFIXES
from orchestrator.errors import ContractError
from orchestrator.hashing import load_json

from .config import HybridConfig, HybridMode

_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ESTIMATOR_RUNS = frozenset({"pf_strict", "mle_count", "mle_spectral"})
_PLANNING_TEMPLATE_FIELDS = frozenset(
    {
        "data_cutoff_step",
        "candidate_poses_xyz",
        "candidate_attestation",
        "dsspp_config",
        "bounds_xyz",
        "continuous_height_bounds_m",
    }
)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ContractError(f"Hybrid config field {field!r} must be an object.")
    return value


def _resolve(base: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"Hybrid config field {field!r} must be a path string.")
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else base / supplied
    if candidate.is_symlink():
        raise ContractError(f"Hybrid config field {field!r} must not be a symlink.")
    return candidate.resolve()


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"Hybrid config field {field!r} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ContractError(f"Hybrid config field {field!r} must be at least {minimum}.")
    return result


def _adapter_policy(value: Mapping[str, object], *, label: str) -> dict[str, object]:
    if "command" in value:
        raise ContractError(f"adapters.{label}.command cannot override the production CLI.")
    if value.get("verify_revision", True) is not True:
        raise ContractError(f"adapters.{label}.verify_revision must be true.")
    if value.get("require_clean", True) is not True:
        raise ContractError(f"adapters.{label}.require_clean must be true.")
    raw = value.get("allowed_dirty_prefixes", PRODUCTION_ALLOWED_DIRTY_PREFIXES)
    if not isinstance(raw, list | tuple) or not all(isinstance(item, str) for item in raw):
        raise ContractError(f"adapters.{label}.allowed_dirty_prefixes must be strings.")
    approved = set(PRODUCTION_ALLOWED_DIRTY_PREFIXES)
    normalized: set[str] = set()
    for item in raw:
        prefix = item.replace("\\", "/")
        while prefix.startswith("./"):
            prefix = prefix[2:]
        if not prefix or prefix.startswith(("/", "../")) or "/../" in prefix:
            raise ContractError(f"adapters.{label} has invalid dirty prefix {item!r}.")
        normalized.add(prefix if prefix.endswith("/") else prefix + "/")
    broadened = normalized - approved
    if broadened:
        raise ContractError(
            f"adapters.{label} broadens the artifact-only allowlist: {sorted(broadened)}"
        )
    return dict(value)


@dataclass(frozen=True, slots=True)
class ProposalKernelConfig:
    """One density-defined external position proposal family."""

    position_sigma_xyz_m: tuple[float, float, float]
    defensive_weight: float
    candidate_weight_floor: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ProposalKernelConfig:
        if value.get("family", "defensive_truncated_gaussian_position") != (
            "defensive_truncated_gaussian_position"
        ):
            raise ContractError("Hybrid v1 supports only the defensive Gaussian kernel.")
        raw_sigma = value.get("position_sigma_xyz_m")
        if not isinstance(raw_sigma, list | tuple) or len(raw_sigma) != 3:
            raise ContractError("proposal_kernel.position_sigma_xyz_m must contain XYZ values.")
        sigma = tuple(float(item) for item in raw_sigma)
        defensive = float(value.get("defensive_weight", 0.1))
        floor = float(value.get("candidate_weight_floor", 1e-12))
        if not all(isfinite(item) and item > 0 for item in sigma):
            raise ContractError("Proposal XYZ sigma values must be finite and positive.")
        if not isfinite(defensive) or not 0 < defensive <= 1:
            raise ContractError("Proposal defensive_weight must lie in (0, 1].")
        if not isfinite(floor) or floor <= 0:
            raise ContractError("Proposal candidate_weight_floor must be positive.")
        return cls(
            position_sigma_xyz_m=(sigma[0], sigma[1], sigma[2]),
            defensive_weight=defensive,
            candidate_weight_floor=floor,
        )

    def for_candidate(self, strength_cps_1m: float) -> dict[str, object]:
        """Weight proposal components by MLE strength without changing the PF target."""
        return {
            "family": "defensive_truncated_gaussian_position",
            "position_sigma_xyz_m": list(self.position_sigma_xyz_m),
            "defensive_weight": self.defensive_weight,
            "candidate_weight": max(float(strength_cps_1m), self.candidate_weight_floor),
        }


@dataclass(frozen=True, slots=True)
class HybridRunConfig:
    """Resolved truth-free inputs and execution policy for one hybrid replay."""

    source_path: Path
    hybrid_run_id: str
    measurement_log_path: Path
    output_directory: Path
    pin_registry_path: Path
    pf_config_path: Path
    mle_count_config_path: Path
    mle_spectral_config_path: Path
    pf_profile: str
    random_seed: int
    relocation_seed: int
    expected_resolved_config_sha256: Mapping[str, str]
    station_end_steps: tuple[tuple[int, int], ...]
    hybrid_policy: HybridConfig
    proposal_kernel: ProposalKernelConfig
    pf_adapter: Mapping[str, object]
    mle_adapter: Mapping[str, object]
    planning_requests: Mapping[int, Mapping[str, object]]

    @classmethod
    def load(cls, path: str | Path) -> HybridRunConfig:
        supplied = Path(path)
        if supplied.is_symlink():
            raise ContractError(f"Hybrid config must not be a symlink: {supplied}")
        source = supplied.resolve()
        payload = load_json(source)
        if payload.get("schema_version") != 1:
            raise ContractError("Hybrid run config schema_version must be 1.")
        forbidden = sorted(set(payload) & {"truth", "evaluation_truth", "truth_path"})
        if forbidden:
            raise ContractError(
                f"The inference-only hybrid config may not contain truth paths: {forbidden}."
            )
        run_id = payload.get("hybrid_run_id")
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            raise ContractError("hybrid_run_id contains unsupported characters.")
        base = source.parent
        estimator_configs = _mapping(payload.get("estimator_configs"), field="estimator_configs")
        expected = _mapping(
            payload.get("expected_resolved_estimator_config_sha256"),
            field="expected_resolved_estimator_config_sha256",
        )
        if set(expected) != _ESTIMATOR_RUNS or any(
            not isinstance(value, str) or not _SHA256.fullmatch(value)
            for value in expected.values()
        ):
            raise ContractError(
                "Expected resolved hashes must exactly name pf_strict, mle_count, and mle_spectral."
            )
        raw_boundaries = payload.get("station_boundaries")
        if not isinstance(raw_boundaries, list) or not raw_boundaries:
            raise ContractError("station_boundaries must be a nonempty array.")
        boundaries: list[tuple[int, int]] = []
        for index, item in enumerate(raw_boundaries):
            row = _mapping(item, field=f"station_boundaries[{index}]")
            if set(row) != {"station_id", "terminal_step_id"}:
                raise ContractError(
                    "Every station boundary must contain only station_id and terminal_step_id."
                )
            boundaries.append(
                (
                    _integer(row["station_id"], field="station_id"),
                    _integer(row["terminal_step_id"], field="terminal_step_id"),
                )
            )
        raw_policy = _mapping(payload.get("hybrid_policy", {}), field="hybrid_policy")
        try:
            policy = HybridConfig(**dict(raw_policy))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Invalid hybrid_policy: {exc}") from exc
        if policy.mode is not HybridMode.PROPOSAL_ONLY_MH:
            raise ContractError("Executable hybrid v1 requires mode='proposal_only_mh'.")
        adapters = _mapping(payload.get("adapters", {}), field="adapters")
        pf_adapter = _adapter_policy(
            _mapping(adapters.get("particle_filter", {}), field="adapters.particle_filter"),
            label="particle_filter",
        )
        mle_adapter = _adapter_policy(
            _mapping(adapters.get("surface_mle", {}), field="adapters.surface_mle"),
            label="surface_mle",
        )
        profile = str(payload.get("pf_profile", "pf_strict"))
        if profile != "pf_strict":
            raise ContractError("Hybrid v1 requires the pf_strict base profile.")
        raw_planning = _mapping(payload.get("planning", {}), field="planning")
        unknown_planning = set(raw_planning) - {"enabled", "requests"}
        if unknown_planning:
            raise ContractError(
                f"Unknown planning configuration fields: {sorted(unknown_planning)}"
            )
        planning_enabled = raw_planning.get("enabled", False)
        if not isinstance(planning_enabled, bool):
            raise ContractError("planning.enabled must be boolean.")
        raw_requests = raw_planning.get("requests", [])
        if not isinstance(raw_requests, list):
            raise ContractError("planning.requests must be an array.")
        if not planning_enabled and raw_requests:
            raise ContractError("Disabled hybrid planning may not declare requests.")
        if planning_enabled and not raw_requests:
            raise ContractError("Enabled hybrid planning requires at least one request.")
        boundary_steps = {step for _, step in boundaries}
        planning_requests: dict[int, Mapping[str, object]] = {}
        for index, item in enumerate(raw_requests):
            row = _mapping(item, field=f"planning.requests[{index}]")
            unknown = set(row) - _PLANNING_TEMPLATE_FIELDS
            required = {
                "data_cutoff_step",
                "candidate_poses_xyz",
                "candidate_attestation",
                "dsspp_config",
            }
            if unknown or not required.issubset(row):
                raise ContractError(
                    "Planning templates must contain cutoff, candidates, attestation, and "
                    f"DSS-PP config only; unknown={sorted(unknown)}."
                )
            cutoff = _integer(
                row["data_cutoff_step"],
                field=f"planning.requests[{index}].data_cutoff_step",
            )
            if cutoff not in boundary_steps or cutoff in planning_requests:
                raise ContractError(
                    "Planning request cutoffs must be unique declared station boundaries."
                )
            planning_requests[cutoff] = dict(row)
        instance = cls(
            source_path=source,
            hybrid_run_id=run_id,
            measurement_log_path=_resolve(
                base, payload.get("measurement_log"), field="measurement_log"
            ),
            output_directory=_resolve(
                base, payload.get("output_directory"), field="output_directory"
            ),
            pin_registry_path=_resolve(base, payload.get("pin_registry"), field="pin_registry"),
            pf_config_path=_resolve(
                base, estimator_configs.get("pf"), field="estimator_configs.pf"
            ),
            mle_count_config_path=_resolve(
                base,
                estimator_configs.get("mle_count"),
                field="estimator_configs.mle_count",
            ),
            mle_spectral_config_path=_resolve(
                base,
                estimator_configs.get("mle_spectral"),
                field="estimator_configs.mle_spectral",
            ),
            pf_profile=profile,
            random_seed=_integer(payload.get("random_seed", 0), field="random_seed"),
            relocation_seed=_integer(
                payload.get("relocation_seed", payload.get("random_seed", 0)),
                field="relocation_seed",
            ),
            expected_resolved_config_sha256={
                str(name): str(value) for name, value in expected.items()
            },
            station_end_steps=tuple(boundaries),
            hybrid_policy=policy,
            proposal_kernel=ProposalKernelConfig.from_mapping(
                _mapping(payload.get("proposal_kernel", {}), field="proposal_kernel")
            ),
            pf_adapter=pf_adapter,
            mle_adapter=mle_adapter,
            planning_requests=planning_requests,
        )
        instance._validate_paths()
        return instance

    def _validate_paths(self) -> None:
        for path in (
            self.source_path,
            self.pin_registry_path,
            self.pf_config_path,
            self.mle_count_config_path,
            self.mle_spectral_config_path,
        ):
            if path.is_symlink() or not path.is_file():
                raise ContractError(f"Hybrid input must be a non-symlink file: {path}")
        if self.measurement_log_path.is_symlink() or not self.measurement_log_path.is_dir():
            raise ContractError(
                f"Hybrid MeasurementLog path is invalid: {self.measurement_log_path}"
            )
        for path in (
            self.pf_config_path,
            self.mle_count_config_path,
            self.mle_spectral_config_path,
        ):
            load_json(path)


__all__ = ["HybridRunConfig", "ProposalKernelConfig"]
