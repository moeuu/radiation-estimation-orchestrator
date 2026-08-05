"""Fail-closed configuration for resumable live hybrid-v2 missions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from orchestrator.errors import ContractError
from orchestrator.estimators.artifacts import (
    mle_resolved_config_sha256,
    pf_resolved_config_sha256,
)
from orchestrator.hashing import load_json

from .mission import MissionBudget
from .predictive import SpectralSchedulerPolicy
from .run_config import (
    SpectralHybridMode,
    _object,
    _path,
)
from .verification import VerificationPolicy

_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass(frozen=True, slots=True)
class RuntimeSessionConfig:
    repository_path: Path
    revision: str
    scenario_path: Path
    session_state_directory: Path
    transcript_path: Path
    timeout_s: float
    require_3d_candidate_diversity: bool


@dataclass(frozen=True, slots=True)
class LivePlannerConfig:
    dsspp_config: dict[str, object]
    dwell_time_s: float


@dataclass(frozen=True, slots=True)
class LiveSpectralHybridRunConfig:
    """One immutable live mission definition without any truth path."""

    source_path: Path
    hybrid_run_id: str
    output_directory: Path
    pf_config_path: Path
    mle_spectral_config_path: Path
    expected_pf_resolved_config_sha256: str
    expected_mle_resolved_config_sha256: str
    mode: SpectralHybridMode
    scheduler_policy: SpectralSchedulerPolicy
    verification_policy: VerificationPolicy
    position_sigma_xyz_m: tuple[float, float, float]
    defensive_weight: float
    random_seed: int
    relocation_seed: int
    runtime: RuntimeSessionConfig
    planner: LivePlannerConfig
    budget: MissionBudget

    @classmethod
    def load(cls, path: str | Path) -> LiveSpectralHybridRunConfig:
        source = Path(path).resolve()
        payload = load_json(source)
        if payload.get("schema_version") != 2 or payload.get("milestone") != (
            "pf_mle_hybrid_live_v2"
        ):
            raise ContractError("Live hybrid config must declare live milestone v2.")
        forbidden = {
            "truth",
            "truth_path",
            "evaluation_truth",
            "measurement_log",
            "station_boundaries",
        }.intersection(payload)
        if forbidden:
            raise ContractError(
                "Live inference config may not contain truth or a prebuilt observation log."
            )
        base = source.parent
        run_id = payload.get("hybrid_run_id")
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
            raise ContractError("hybrid_run_id is invalid.")
        configs = _object(payload.get("estimator_configs"), field="estimator_configs")
        runtime_raw = _object(payload.get("runtime"), field="runtime")
        revision = runtime_raw.get("revision")
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ContractError("Live runtime must be pinned to an exact commit.")
        planner_raw = _object(payload.get("planner"), field="planner")
        dsspp = _object(planner_raw.get("dsspp_config"), field="planner.dsspp_config")
        dwell = float(planner_raw.get("dwell_time_s", 0.0))
        if dwell <= 0:
            raise ContractError("planner.dwell_time_s must be positive.")
        budget_raw = _object(payload.get("mission_budget"), field="mission_budget")
        scheduler = _object(payload.get("scheduler", {}), field="scheduler")
        verification = _object(payload.get("verification"), field="verification")
        kernel = _object(payload.get("relocation_kernel", {}), field="relocation_kernel")
        sigma_raw = kernel.get("position_sigma_xyz_m", [0.5, 0.5, 0.5])
        if not isinstance(sigma_raw, list) or len(sigma_raw) != 3:
            raise ContractError("relocation_kernel.position_sigma_xyz_m requires XYZ.")
        sigma = tuple(float(value) for value in sigma_raw)
        if any(value <= 0 for value in sigma):
            raise ContractError("relocation position sigma values must be positive.")
        defensive = float(kernel.get("defensive_weight", 0.1))
        if not 0 < defensive <= 1:
            raise ContractError("relocation defensive_weight must be in (0, 1].")
        timeout = float(runtime_raw.get("timeout_s", 300.0))
        if timeout <= 0:
            raise ContractError("runtime.timeout_s must be positive.")
        if "pin_registry" in payload or "adapters" in payload:
            raise ContractError(
                "Live-v2 estimators are local; pin_registry/adapters are not allowed."
            )
        output = _path(base, payload.get("output_directory"), field="output_directory")
        pf_config = _path(base, configs.get("pf_strict"), field="pf_strict")
        mle_config = _path(base, configs.get("mle_spectral"), field="mle_spectral")
        return cls(
            source_path=source,
            hybrid_run_id=run_id,
            output_directory=output,
            pf_config_path=pf_config,
            mle_spectral_config_path=mle_config,
            expected_pf_resolved_config_sha256=pf_resolved_config_sha256(pf_config),
            expected_mle_resolved_config_sha256=mle_resolved_config_sha256(mle_config),
            mode=SpectralHybridMode(str(payload.get("mode"))),
            scheduler_policy=SpectralSchedulerPolicy(**scheduler),
            verification_policy=VerificationPolicy(**verification),
            position_sigma_xyz_m=(sigma[0], sigma[1], sigma[2]),
            defensive_weight=defensive,
            random_seed=int(payload.get("random_seed", 0)),
            relocation_seed=int(payload.get("relocation_seed", 1)),
            runtime=RuntimeSessionConfig(
                repository_path=_path(
                    base, runtime_raw.get("repository_path"), field="runtime.repository_path"
                ),
                revision=revision,
                scenario_path=_path(
                    base, runtime_raw.get("scenario_path"), field="runtime.scenario_path"
                ),
                session_state_directory=_path(
                    base,
                    runtime_raw.get(
                        "session_state_directory",
                        str(output / "runtime_session"),
                    ),
                    field="runtime.session_state_directory",
                ),
                transcript_path=_path(
                    base,
                    runtime_raw.get("transcript_path", str(output / "runtime.jsonl")),
                    field="runtime.transcript_path",
                ),
                timeout_s=timeout,
                require_3d_candidate_diversity=(
                    runtime_raw.get("require_3d_candidate_diversity", True) is True
                ),
            ),
            planner=LivePlannerConfig(dsspp_config=dsspp, dwell_time_s=dwell),
            budget=MissionBudget(
                max_actions=int(budget_raw["max_actions"]),
                max_total_time_s=float(budget_raw["max_total_time_s"]),
                max_live_time_s=float(budget_raw["max_live_time_s"]),
                max_travel_time_s=float(budget_raw["max_travel_time_s"]),
                max_shield_actuation_time_s=float(
                    budget_raw["max_shield_actuation_time_s"]
                ),
            ),
        )


__all__ = [
    "LivePlannerConfig",
    "LiveSpectralHybridRunConfig",
    "RuntimeSessionConfig",
]
