"""Fail-closed configuration for raw-spectrum offline hybrid v2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from orchestrator.errors import ContractError
from orchestrator.estimators.artifacts import (
    mle_resolved_config_sha256,
    pf_resolved_config_sha256,
)
from orchestrator.hashing import load_json

from .predictive import SpectralSchedulerPolicy
from .verification import VerificationPolicy


class SpectralHybridMode(StrEnum):
    VERIFICATION_ONLY = "spectral_verification_only"
    WITHIN_MODEL_RELOCATION = "spectral_within_model_relocation"
    EXACT_RJ = "spectral_exact_rj"


def _object(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractError(f"Hybrid-v2 field {field!r} must be an object.")
    return dict(value)


def _path(base: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"Hybrid-v2 field {field!r} must be a path string.")
    supplied = Path(value)
    resolved = (supplied if supplied.is_absolute() else base / supplied).resolve()
    if resolved.is_symlink():
        raise ContractError(f"Hybrid-v2 path {field!r} may not be a symlink.")
    return resolved


@dataclass(frozen=True, slots=True)
class SpectralHybridRunConfig:
    source_path: Path
    hybrid_run_id: str
    measurement_log_path: Path
    output_directory: Path
    pf_config_path: Path
    mle_spectral_config_path: Path
    expected_pf_resolved_config_sha256: str
    expected_mle_resolved_config_sha256: str
    station_end_steps: tuple[tuple[int, int], ...]
    mode: SpectralHybridMode
    scheduler_policy: SpectralSchedulerPolicy
    verification_policy: VerificationPolicy
    position_sigma_xyz_m: tuple[float, float, float]
    defensive_weight: float
    random_seed: int
    relocation_seed: int

    @classmethod
    def load(cls, path: str | Path) -> SpectralHybridRunConfig:
        source = Path(path).resolve()
        payload = load_json(source)
        if payload.get("schema_version") != 2 or payload.get("milestone") != (
            "pf_mle_hybrid_v2"
        ):
            raise ContractError("Raw-spectrum hybrid config must declare milestone v2.")
        forbidden = {"truth", "truth_path", "evaluation_truth"}.intersection(payload)
        if forbidden:
            raise ContractError("Hybrid-v2 inference config may not contain truth paths.")
        base = source.parent
        run_id = payload.get("hybrid_run_id")
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise ContractError("hybrid_run_id is invalid.")
        configs = _object(payload.get("estimator_configs"), field="estimator_configs")
        boundaries_raw = payload.get("station_boundaries")
        if not isinstance(boundaries_raw, list) or not boundaries_raw:
            raise ContractError("Hybrid-v2 station_boundaries must be nonempty.")
        boundaries: list[tuple[int, int]] = []
        for row in boundaries_raw:
            item = _object(row, field="station_boundaries[]")
            if set(item) != {"station_id", "terminal_step_id"}:
                raise ContractError("Station boundary rows have unexpected fields.")
            station = item["station_id"]
            step = item["terminal_step_id"]
            if (
                isinstance(station, bool)
                or not isinstance(station, int)
                or isinstance(step, bool)
                or not isinstance(step, int)
                or min(station, step) < 0
            ):
                raise ContractError("Station boundaries must contain nonnegative integers.")
            boundaries.append((station, step))
        scheduler = _object(payload.get("scheduler", {}), field="scheduler")
        verification = _object(payload.get("verification"), field="verification")
        kernel = _object(payload.get("relocation_kernel", {}), field="relocation_kernel")
        sigma_raw = kernel.get("position_sigma_xyz_m", [0.5, 0.5, 0.5])
        if not isinstance(sigma_raw, list) or len(sigma_raw) != 3:
            raise ContractError("relocation_kernel.position_sigma_xyz_m requires XYZ.")
        sigma = tuple(float(value) for value in sigma_raw)
        defensive = float(kernel.get("defensive_weight", 0.1))
        if "pin_registry" in payload or "adapters" in payload:
            raise ContractError(
                "Hybrid-v2 estimators are local; pin_registry/adapters are no longer allowed."
            )
        pf_config = _path(base, configs.get("pf_strict"), field="pf_strict")
        mle_config = _path(base, configs.get("mle_spectral"), field="mle_spectral")
        return cls(
            source_path=source,
            hybrid_run_id=run_id,
            measurement_log_path=_path(
                base, payload.get("measurement_log"), field="measurement_log"
            ),
            output_directory=_path(
                base, payload.get("output_directory"), field="output_directory"
            ),
            pf_config_path=pf_config,
            mle_spectral_config_path=mle_config,
            expected_pf_resolved_config_sha256=pf_resolved_config_sha256(pf_config),
            expected_mle_resolved_config_sha256=mle_resolved_config_sha256(mle_config),
            station_end_steps=tuple(boundaries),
            mode=SpectralHybridMode(str(payload.get("mode"))),
            scheduler_policy=SpectralSchedulerPolicy(**scheduler),
            verification_policy=VerificationPolicy(**verification),
            position_sigma_xyz_m=(sigma[0], sigma[1], sigma[2]),
            defensive_weight=defensive,
            random_seed=int(payload.get("random_seed", 0)),
            relocation_seed=int(payload.get("relocation_seed", 1)),
        )


__all__ = ["SpectralHybridMode", "SpectralHybridRunConfig"]
