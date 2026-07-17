"""Future-observation verification and non-destructive quarantine of MLE candidates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from math import isfinite

from orchestrator.errors import ContractError, DataReuseError

from .config import HybridConfig
from .directives import PFDirective


class CandidateState(StrEnum):
    """Verification states; quarantine never directly deletes PF particles."""

    PENDING = "pending"
    VERIFIED = "verified"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class VerificationCandidate:
    """Evidence state for one snapshot-derived proposal."""

    directive_id: str
    proposal_id: str
    snapshot_candidate_id: str
    data_cutoff_step: int
    corroboration_min_step: int
    state: CandidateState = CandidateState.PENDING
    future_step_ids: tuple[int, ...] = ()
    cumulative_log_predictive_likelihood_ratio: float = 0.0


class VerificationQueue:
    """A candidate queue whose only evidence comes strictly after each cutoff."""

    def __init__(self, config: HybridConfig) -> None:
        self._config = config
        self._candidates: dict[tuple[str, str], VerificationCandidate] = {}
        self._directive_cutoffs: dict[str, int] = {}

    @property
    def candidates(self) -> tuple[VerificationCandidate, ...]:
        """Return candidates in stable directive/proposal order."""
        return tuple(self._candidates[key] for key in sorted(self._candidates))

    def register(self, directive: PFDirective) -> tuple[VerificationCandidate, ...]:
        """Register all proposals once without modifying PF weights or particles."""
        if directive.directive_id in self._directive_cutoffs:
            raise DataReuseError(f"Directive {directive.directive_id} is already registered.")
        cutoff = directive.snapshot.prefix.data_cutoff_step
        self._directive_cutoffs[directive.directive_id] = cutoff
        registered: list[VerificationCandidate] = []
        for proposal in directive.proposals:
            key = (directive.directive_id, proposal.proposal_id)
            candidate = VerificationCandidate(
                directive_id=directive.directive_id,
                proposal_id=proposal.proposal_id,
                snapshot_candidate_id=proposal.snapshot_candidate_id,
                data_cutoff_step=cutoff,
                corroboration_min_step=cutoff + 1,
            )
            self._candidates[key] = candidate
            registered.append(candidate)
        return tuple(registered)

    def corroborate(
        self,
        *,
        directive_id: str,
        proposal_id: str,
        step_id: int,
        log_predictive_likelihood_ratio: float,
    ) -> VerificationCandidate:
        """Accumulate independent future evidence and update verification state."""
        key = (directive_id, proposal_id)
        try:
            candidate = self._candidates[key]
        except KeyError as exc:
            raise ContractError("Corroboration references an unknown directive proposal.") from exc
        if candidate.state is not CandidateState.PENDING:
            raise DataReuseError(
                "A resolved candidate may not consume more corroboration evidence."
            )
        if step_id < candidate.corroboration_min_step:
            raise DataReuseError(
                "Candidate corroboration must use only observations strictly after its cutoff."
            )
        if candidate.future_step_ids and step_id <= candidate.future_step_ids[-1]:
            raise DataReuseError("Candidate corroboration steps must be strictly increasing.")
        if not isfinite(log_predictive_likelihood_ratio):
            raise ContractError("Corroboration log predictive ratio must be finite.")
        steps = (*candidate.future_step_ids, int(step_id))
        cumulative = candidate.cumulative_log_predictive_likelihood_ratio + float(
            log_predictive_likelihood_ratio
        )
        state = CandidateState.PENDING
        if len(steps) >= self._config.verification_min_future_observations:
            if cumulative >= self._config.verification_support_log_predictive_ratio:
                state = CandidateState.VERIFIED
            elif cumulative <= self._config.verification_reject_log_predictive_ratio:
                state = CandidateState.QUARANTINED
        updated = replace(
            candidate,
            state=state,
            future_step_ids=steps,
            cumulative_log_predictive_likelihood_ratio=cumulative,
        )
        self._candidates[key] = updated
        return updated

    def quarantine(self, *, directive_id: str, proposal_id: str) -> VerificationCandidate:
        """Return a quarantined view; this intentionally has no PF-prune side effect."""
        key = (directive_id, proposal_id)
        try:
            candidate = self._candidates[key]
        except KeyError as exc:
            raise ContractError("Unknown candidate cannot be quarantined.") from exc
        if candidate.state is not CandidateState.QUARANTINED:
            raise ContractError("Only evidence-rejected candidates enter quarantine.")
        return candidate
