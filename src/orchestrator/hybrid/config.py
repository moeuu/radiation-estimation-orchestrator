"""Fail-closed capabilities for the first executable hybrid milestone."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite


class HybridMode(StrEnum):
    """Only feedback modes with an explicit no-double-use interpretation."""

    VERIFICATION_ONLY = "verification_only"
    PROPOSAL_ONLY_MH = "proposal_only_mh"


@dataclass(frozen=True, slots=True)
class HybridCapabilities:
    """Resolved authority granted to a hybrid controller and PF consumer."""

    register_verification_candidates: bool
    submit_external_proposals: bool
    require_target_preserving_mh: bool
    direct_mle_objective_reweight: bool = False
    hard_prune: bool = False
    future_only_corroboration: bool = True
    once_only_directives: bool = True


def capabilities_for(mode: HybridMode) -> HybridCapabilities:
    """Resolve a mode without consulting legacy feature booleans."""
    if mode is HybridMode.VERIFICATION_ONLY:
        return HybridCapabilities(
            register_verification_candidates=True,
            submit_external_proposals=False,
            require_target_preserving_mh=False,
        )
    if mode is HybridMode.PROPOSAL_ONLY_MH:
        return HybridCapabilities(
            register_verification_candidates=True,
            submit_external_proposals=True,
            require_target_preserving_mh=True,
        )
    raise ValueError(f"Unsupported hybrid mode: {mode!r}")


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Validated controller policy; unsafe feedback cannot be enabled in v1."""

    mode: HybridMode = HybridMode.VERIFICATION_ONLY
    station_interval: int = 1
    minimum_completed_stations: int = 1
    predictive_deviance_threshold: float | None = None
    predictive_mismatch_streak: int = 2
    scheduler_cooldown_stations: int = 0
    verification_support_log_predictive_ratio: float = 3.0
    verification_reject_log_predictive_ratio: float = -3.0
    verification_min_future_observations: int = 1
    allow_direct_mle_objective_reweight: bool = False
    allow_hard_prune: bool = False
    require_station_complete_cutoff: bool = True
    require_future_only_corroboration: bool = True
    require_once_only_directives: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", HybridMode(self.mode))
        if self.station_interval < 1:
            raise ValueError("station_interval must be at least one.")
        if self.minimum_completed_stations < 1:
            raise ValueError("minimum_completed_stations must be at least one.")
        if self.predictive_mismatch_streak < 1:
            raise ValueError("predictive_mismatch_streak must be at least one.")
        if self.scheduler_cooldown_stations < 0:
            raise ValueError("scheduler_cooldown_stations must be nonnegative.")
        threshold = self.predictive_deviance_threshold
        if threshold is not None and (not isfinite(threshold) or threshold < 0):
            raise ValueError("predictive_deviance_threshold must be finite and nonnegative.")
        if self.verification_min_future_observations < 1:
            raise ValueError("verification_min_future_observations must be at least one.")
        support = self.verification_support_log_predictive_ratio
        reject = self.verification_reject_log_predictive_ratio
        if not isfinite(support) or support <= 0:
            raise ValueError("verification support threshold must be finite and positive.")
        if not isfinite(reject) or reject >= 0:
            raise ValueError("verification reject threshold must be finite and negative.")
        if self.allow_direct_mle_objective_reweight:
            raise ValueError("Direct MLE-objective reweighting is forbidden in hybrid v1.")
        if self.allow_hard_prune:
            raise ValueError("MLE-directed hard pruning is forbidden in hybrid v1.")
        if not self.require_station_complete_cutoff:
            raise ValueError("Hybrid v1 requires an explicit station-complete cutoff.")
        if not self.require_future_only_corroboration:
            raise ValueError("Hybrid v1 requires strictly future corroboration.")
        if not self.require_once_only_directives:
            raise ValueError("Hybrid v1 requires once-only directive application.")

    @property
    def capabilities(self) -> HybridCapabilities:
        """Return the immutable capability map for the selected mode."""
        return capabilities_for(self.mode)

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-safe resolved configuration."""
        payload = asdict(self)
        payload["mode"] = self.mode.value
        payload["capabilities"] = asdict(self.capabilities)
        return payload
