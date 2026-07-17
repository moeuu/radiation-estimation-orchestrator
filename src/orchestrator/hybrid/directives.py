"""Versioned safe MLE-to-PF directive and receipt construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from orchestrator.errors import ContractError
from orchestrator.hashing import canonical_json_bytes, json_safe, sha256_bytes

from .config import HybridConfig, HybridMode
from .snapshot import MLESnapshot, SnapshotCluster


def _require_sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest.")
    return value


@dataclass(frozen=True, slots=True)
class DirectiveProposal:
    """A candidate registration or a fully specified external MH proposal kernel."""

    proposal_id: str
    snapshot_candidate_id: str
    isotope: str
    candidate_mean_xyz: tuple[float, float, float]
    snapshot_strength_cps_1m_metadata: float
    proposal_kernel: Mapping[str, object] | None = None

    @classmethod
    def from_cluster(
        cls,
        cluster: SnapshotCluster,
        *,
        proposal_kernel: Mapping[str, object] | None = None,
        candidate_mean_xyz: tuple[float, float, float] | None = None,
    ) -> DirectiveProposal:
        """Create a stable proposal identity from one snapshot candidate."""
        mean_xyz = cluster.centroid_xyz if candidate_mean_xyz is None else candidate_mean_xyz
        identity = {
            "snapshot_candidate_id": cluster.snapshot_candidate_id,
            "isotope": cluster.isotope,
            "candidate_mean_xyz": list(mean_xyz),
            "snapshot_strength_cps_1m_metadata": cluster.integrated_strength_cps_1m,
        }
        return cls(
            proposal_id=f"proposal-{sha256_bytes(canonical_json_bytes(identity))[:20]}",
            snapshot_candidate_id=cluster.snapshot_candidate_id,
            isotope=cluster.isotope,
            candidate_mean_xyz=mean_xyz,
            snapshot_strength_cps_1m_metadata=cluster.integrated_strength_cps_1m,
            proposal_kernel=(
                None
                if proposal_kernel is None
                else MappingProxyType(dict(json_safe(proposal_kernel)))  # type: ignore[arg-type]
            ),
        )

    def __post_init__(self) -> None:
        if not self.proposal_id or not self.snapshot_candidate_id or not self.isotope:
            raise ContractError("Directive proposal identifiers and isotope must be nonempty.")
        if len(self.candidate_mean_xyz) != 3 or not all(
            isfinite(value) for value in self.candidate_mean_xyz
        ):
            raise ContractError("Directive proposal position must contain three finite values.")
        if (
            not isfinite(self.snapshot_strength_cps_1m_metadata)
            or self.snapshot_strength_cps_1m_metadata < 0
        ):
            raise ContractError("Directive proposal strength must be finite and nonnegative.")

    def to_dict(self) -> dict[str, object]:
        """Return PFDirective v1 proposal data."""
        return {
            "proposal_id": self.proposal_id,
            "snapshot_candidate_id": self.snapshot_candidate_id,
            "isotope": self.isotope,
            "candidate_mean_xyz": list(self.candidate_mean_xyz),
            "snapshot_strength_cps_1m_metadata": self.snapshot_strength_cps_1m_metadata,
            "proposal_kernel": (
                None if self.proposal_kernel is None else dict(self.proposal_kernel)
            ),
        }


@dataclass(frozen=True, slots=True)
class PFDirective:
    """A once-only instruction bound to one MLE snapshot cutoff."""

    directive_id: str
    kind: HybridMode
    snapshot: MLESnapshot
    pf_resolved_config_sha256: str
    proposals: tuple[DirectiveProposal, ...]
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return canonical PFDirective v1 JSON data."""
        requires_mh = self.kind is HybridMode.PROPOSAL_ONLY_MH
        prefix = self.snapshot.prefix
        return {
            "schema_version": 1,
            "directive_id": self.directive_id,
            "directive_kind": self.kind.value,
            "snapshot_id": self.snapshot.snapshot_id,
            "snapshot_sha256": self.snapshot.sha256,
            "source_run_id": prefix.source_run_id,
            "prefix_measurement_log_sha256": prefix.prefix_measurement_log_sha256,
            "covered_records_sha256": prefix.covered_records_sha256,
            "covered_station_boundaries_sha256": prefix.covered_station_boundaries_sha256,
            "pf_resolved_config_sha256": self.pf_resolved_config_sha256,
            "data_cutoff_step": prefix.data_cutoff_step,
            "data_cutoff_station": prefix.data_cutoff_station,
            "cutoff_station_complete": prefix.cutoff_station_complete,
            "covered_step_ids": list(prefix.covered_step_ids),
            "apply_after_step": prefix.data_cutoff_step,
            "corroboration_min_step": prefix.corroboration_min_step,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "safety_policy": {
                "direct_mle_objective_reweight": False,
                "hard_prune_authorized": False,
                "future_only_corroboration": True,
                "once_only_application": True,
                "requires_target_preserving_mh": requires_mh,
            },
            "provenance": dict(self.provenance),
        }

    @property
    def sha256(self) -> str:
        """Hash canonical directive semantics."""
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


def build_pf_directive(
    snapshot: MLESnapshot,
    *,
    config: HybridConfig,
    pf_resolved_config_sha256: str,
    proposals: Sequence[DirectiveProposal],
    provenance: Mapping[str, object],
) -> PFDirective:
    """Build a safe directive after resolving the selected capability profile."""
    normalized = tuple(proposals)
    if not normalized:
        raise ContractError("A PF directive must contain at least one candidate proposal.")
    available = {cluster.snapshot_candidate_id for cluster in snapshot.clusters}
    proposal_ids = [proposal.proposal_id for proposal in normalized]
    if len(set(proposal_ids)) != len(proposal_ids):
        raise ContractError("PF directive proposal IDs must be unique.")
    if any(proposal.snapshot_candidate_id not in available for proposal in normalized):
        raise ContractError("PF directive proposal is not present in its MLE snapshot.")
    if config.mode is HybridMode.VERIFICATION_ONLY:
        if any(proposal.proposal_kernel is not None for proposal in normalized):
            raise ContractError("Verification-only directives may not carry a proposal kernel.")
    else:
        if any(proposal.proposal_kernel is None for proposal in normalized):
            raise ContractError("proposal_only_mh requires a density-defined proposal kernel.")
        for proposal in normalized:
            kernel = proposal.proposal_kernel
            assert kernel is not None
            if kernel.get("family") != "defensive_truncated_gaussian_position":
                raise ContractError("Unsupported proposal_only_mh proposal kernel family.")
            sigma = kernel.get("position_sigma_xyz_m")
            if not isinstance(sigma, list | tuple) or len(sigma) != 3:
                raise ContractError("Proposal kernel position_sigma_xyz_m must have three values.")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not isfinite(float(value))
                or float(value) <= 0
                for value in sigma
            ):
                raise ContractError("Proposal kernel position sigma values must be positive.")
            for name in ("defensive_weight", "candidate_weight"):
                value = kernel.get(name)
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ContractError(f"Proposal kernel {name} must be numeric.")
                numeric = float(value)
                if not isfinite(numeric) or numeric <= 0:
                    raise ContractError(f"Proposal kernel {name} must be finite and positive.")
            if float(kernel["defensive_weight"]) > 1:
                raise ContractError("Proposal kernel defensive_weight must not exceed one.")
    _require_sha256(pf_resolved_config_sha256, label="pf_resolved_config_sha256")
    identity = {
        "schema_version": 1,
        "directive_kind": config.mode.value,
        "snapshot_sha256": snapshot.sha256,
        "pf_resolved_config_sha256": pf_resolved_config_sha256,
        "proposals": [proposal.to_dict() for proposal in normalized],
    }
    directive_id = f"directive-{sha256_bytes(canonical_json_bytes(identity))[:20]}"
    safe_provenance = MappingProxyType(dict(json_safe(provenance)))  # type: ignore[arg-type]
    return PFDirective(
        directive_id=directive_id,
        kind=config.mode,
        snapshot=snapshot,
        pf_resolved_config_sha256=pf_resolved_config_sha256,
        proposals=normalized,
        provenance=safe_provenance,
    )


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    """PF consumer evidence for one directive proposal."""

    proposal_id: str
    outcome: str
    mh_attempt_count: int = 0
    mh_accepted_count: int = 0
    mh_rejected_count: int = 0
    not_sampled_count: int = 0
    eligible_particle_count: int = 0
    mh_log_acceptance_ratio: float | None = None
    mh_log_uniform_draw: float | None = None

    def __post_init__(self) -> None:
        if not self.proposal_id:
            raise ContractError("Candidate outcome requires proposal_id.")
        allowed = {
            "registered",
            "mh_accepted",
            "mh_rejected",
            "mh_mixed",
            "not_applied",
        }
        if self.outcome not in allowed:
            raise ContractError(f"Unsupported candidate outcome: {self.outcome!r}.")
        for name, value in (
            ("mh_log_acceptance_ratio", self.mh_log_acceptance_ratio),
            ("mh_log_uniform_draw", self.mh_log_uniform_draw),
        ):
            if value is not None and not isfinite(value):
                raise ContractError(f"{name} must be finite when present.")
        counts = (
            self.mh_attempt_count,
            self.mh_accepted_count,
            self.mh_rejected_count,
            self.not_sampled_count,
            self.eligible_particle_count,
        )
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ContractError("Candidate outcome aggregate counts must be nonnegative integers.")
        if self.mh_attempt_count != self.mh_accepted_count + self.mh_rejected_count:
            raise ContractError("Candidate MH attempts must equal accepted plus rejected counts.")
        if self.eligible_particle_count != self.mh_attempt_count + self.not_sampled_count:
            raise ContractError("Eligible particles must equal attempted plus not-sampled counts.")
        expected_outcome = (
            "not_applied"
            if self.mh_attempt_count == 0
            else "mh_accepted"
            if self.mh_accepted_count == self.mh_attempt_count
            else "mh_rejected"
            if self.mh_rejected_count == self.mh_attempt_count
            else "mh_mixed"
        )
        if self.outcome not in {"registered", expected_outcome}:
            raise ContractError("Candidate outcome label differs from its aggregate MH counts.")
        has_ratio = self.mh_log_acceptance_ratio is not None
        has_draw = self.mh_log_uniform_draw is not None
        if has_ratio != has_draw or has_ratio != (self.mh_attempt_count == 1):
            raise ContractError(
                "Scalar MH evidence is required only for exactly one candidate attempt."
            )

    def to_dict(self) -> dict[str, object]:
        """Return receipt contract data."""
        return {
            "proposal_id": self.proposal_id,
            "outcome": self.outcome,
            "mh_attempt_count": self.mh_attempt_count,
            "mh_accepted_count": self.mh_accepted_count,
            "mh_rejected_count": self.mh_rejected_count,
            "not_sampled_count": self.not_sampled_count,
            "eligible_particle_count": self.eligible_particle_count,
            "mh_log_acceptance_ratio": self.mh_log_acceptance_ratio,
            "mh_log_uniform_draw": self.mh_log_uniform_draw,
        }


@dataclass(frozen=True, slots=True)
class PFDirectiveReceipt:
    """Proof that a PF consumer handled one directive without observation reuse."""

    receipt_id: str
    directive: PFDirective
    consumer_variant: str
    status: str
    pf_state_sha256_before: str
    pf_state_sha256_after: str
    outcomes: tuple[CandidateOutcome, ...]
    provenance: Mapping[str, object]

    @classmethod
    def create(
        cls,
        *,
        directive: PFDirective,
        consumer_variant: str,
        status: str,
        pf_state_sha256_before: str,
        pf_state_sha256_after: str,
        outcomes: Sequence[CandidateOutcome],
        provenance: Mapping[str, object],
    ) -> PFDirectiveReceipt:
        """Create a complete receipt while enforcing mode-specific evidence."""
        if status not in {"applied", "rejected"}:
            raise ContractError("PF directive receipt status must be applied or rejected.")
        if not consumer_variant:
            raise ContractError("PF directive receipt requires consumer_variant.")
        _require_sha256(pf_state_sha256_before, label="pf_state_sha256_before")
        _require_sha256(pf_state_sha256_after, label="pf_state_sha256_after")
        normalized = tuple(outcomes)
        proposal_ids = {proposal.proposal_id for proposal in directive.proposals}
        outcome_ids = [outcome.proposal_id for outcome in normalized]
        if set(outcome_ids) != proposal_ids or len(outcome_ids) != len(set(outcome_ids)):
            raise ContractError("Receipt must account exactly once for every directive proposal.")
        for outcome in normalized:
            if status == "rejected":
                valid = outcome.outcome == "not_applied"
            elif directive.kind is HybridMode.VERIFICATION_ONLY:
                valid = (
                    outcome.outcome == "registered"
                    and outcome.mh_attempt_count == 0
                    and outcome.eligible_particle_count == 0
                    and outcome.mh_log_acceptance_ratio is None
                    and outcome.mh_log_uniform_draw is None
                )
            else:
                sampled = outcome.mh_attempt_count > 0
                unsampled = outcome.outcome == "not_applied"
                valid = (
                    sampled and outcome.outcome in {"mh_accepted", "mh_rejected", "mh_mixed"}
                ) or (
                    unsampled
                    and outcome.mh_attempt_count == 0
                    and outcome.mh_log_acceptance_ratio is None
                    and outcome.mh_log_uniform_draw is None
                )
            if not valid:
                raise ContractError(
                    "Receipt candidate outcome is inconsistent with directive mode."
                )
        identity = {
            "schema_version": 1,
            "directive_sha256": directive.sha256,
            "pf_state_sha256_before": pf_state_sha256_before,
            "pf_state_sha256_after": pf_state_sha256_after,
            "outcomes": [outcome.to_dict() for outcome in normalized],
        }
        receipt_id = f"receipt-{sha256_bytes(canonical_json_bytes(identity))[:20]}"
        return cls(
            receipt_id=receipt_id,
            directive=directive,
            consumer_variant=consumer_variant,
            status=status,
            pf_state_sha256_before=pf_state_sha256_before,
            pf_state_sha256_after=pf_state_sha256_after,
            outcomes=normalized,
            provenance=MappingProxyType(dict(json_safe(provenance))),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        """Return canonical PFDirectiveReceipt v1 JSON data."""
        target_mh = self.directive.kind is HybridMode.PROPOSAL_ONLY_MH
        cutoff = self.directive.snapshot.prefix.data_cutoff_step
        prefix = self.directive.snapshot.prefix
        return {
            "schema_version": 1,
            "receipt_id": self.receipt_id,
            "directive_id": self.directive.directive_id,
            "directive_sha256": self.directive.sha256,
            "directive_kind": self.directive.kind.value,
            "consumer_family": "particle_filter",
            "consumer_variant": self.consumer_variant,
            "data_cutoff_step": cutoff,
            "applied_after_step": cutoff,
            "status": self.status,
            "pf_state_sha256_before": self.pf_state_sha256_before,
            "pf_state_sha256_after": self.pf_state_sha256_after,
            "candidate_outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "safety_evidence": {
                "direct_mle_objective_reweight_performed": False,
                "hard_prune_performed": False,
                "target_preserving_mh_performed": target_mh and self.status == "applied",
                "reweighted_observation_step_ids": [],
                "next_observation_min_step": cutoff + 1,
            },
            "provenance": {
                **dict(self.provenance),
                "source_run_id": prefix.source_run_id,
                "covered_records_sha256": prefix.covered_records_sha256,
                "pf_resolved_config_sha256": self.directive.pf_resolved_config_sha256,
            },
        }

    @property
    def sha256(self) -> str:
        """Hash canonical receipt semantics."""
        return sha256_bytes(canonical_json_bytes(self.to_dict()))
