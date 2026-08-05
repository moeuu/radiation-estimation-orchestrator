"""Independent-block verification for frozen spectral-MLE candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from math import isfinite
from typing import Any

from orchestrator.errors import ContractError, DataReuseError


class CandidateState(StrEnum):
    PENDING = "pending"
    VERIFIED = "verified"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    support_log_likelihood_ratio: float
    reject_log_likelihood_ratio: float
    minimum_blocks: int = 2
    minimum_distinct_stations: int = 2
    minimum_distinct_heights: int = 1
    minimum_distinct_shield_programs: int = 1
    pending_planner_weight: float = 0.2
    verified_planner_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.reject_log_likelihood_ratio >= self.support_log_likelihood_ratio:
            raise ValueError("Reject threshold must be smaller than support threshold.")
        for name in (
            "minimum_blocks",
            "minimum_distinct_stations",
            "minimum_distinct_heights",
            "minimum_distinct_shield_programs",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if not 0 <= self.pending_planner_weight < self.verified_planner_weight:
            raise ValueError("Pending planner weight must be below verified planner weight.")


@dataclass(frozen=True, slots=True)
class BlockEvidence:
    block_id: str
    station_id: int
    height_group_id: str
    shield_program_id: str
    step_ids: tuple[int, ...]
    log_likelihood_ratio: float
    score_artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.block_id or self.station_id < 0:
            raise ContractError("Verification block identity is invalid.")
        if not self.height_group_id or not self.shield_program_id:
            raise ContractError("Verification block requires height and shield-program IDs.")
        if not self.step_ids or tuple(sorted(set(self.step_ids))) != self.step_ids:
            raise ContractError("Verification block step IDs must be sorted and unique.")
        if not isfinite(self.log_likelihood_ratio):
            raise ContractError("Verification block LLR must be finite.")
        if len(self.score_artifact_sha256) != 64:
            raise ContractError("Verification block score hash must be SHA-256 shaped.")


@dataclass(frozen=True, slots=True)
class VerificationCandidate:
    snapshot_id: str
    snapshot_sha256: str
    candidate_id: str
    data_cutoff_step: int
    state: CandidateState = CandidateState.PENDING
    blocks: tuple[BlockEvidence, ...] = ()
    cumulative_log_likelihood_ratio: float = 0.0

    @property
    def distinct_station_count(self) -> int:
        return len({block.station_id for block in self.blocks})

    @property
    def distinct_height_count(self) -> int:
        return len({block.height_group_id for block in self.blocks})

    @property
    def distinct_shield_program_count(self) -> int:
        return len({block.shield_program_id for block in self.blocks})


class BlockVerificationQueue:
    """Resolve candidates only after sufficiently distinct post-cutoff blocks."""

    def __init__(self, policy: VerificationPolicy) -> None:
        self.policy = policy
        self._candidates: dict[tuple[str, str], VerificationCandidate] = {}

    @property
    def candidates(self) -> tuple[VerificationCandidate, ...]:
        return tuple(self._candidates[key] for key in sorted(self._candidates))

    def register(
        self,
        *,
        snapshot_id: str,
        snapshot_sha256: str,
        candidate_id: str,
        data_cutoff_step: int,
    ) -> VerificationCandidate:
        key = (snapshot_id, candidate_id)
        if key in self._candidates:
            raise DataReuseError("A spectral candidate may be registered only once.")
        candidate = VerificationCandidate(
            snapshot_id=snapshot_id,
            snapshot_sha256=snapshot_sha256,
            candidate_id=candidate_id,
            data_cutoff_step=int(data_cutoff_step),
        )
        self._candidates[key] = candidate
        return candidate

    def corroborate(
        self,
        *,
        snapshot_id: str,
        candidate_id: str,
        evidence: BlockEvidence,
    ) -> VerificationCandidate:
        key = (snapshot_id, candidate_id)
        try:
            candidate = self._candidates[key]
        except KeyError as exc:
            raise ContractError("Verification evidence references an unknown candidate.") from exc
        if candidate.state is not CandidateState.PENDING:
            raise DataReuseError("A resolved candidate may not consume more evidence.")
        if any(step <= candidate.data_cutoff_step for step in evidence.step_ids):
            raise DataReuseError("Verification evidence must be strictly post-cutoff.")
        existing_blocks = {block.block_id for block in candidate.blocks}
        existing_steps = {step for block in candidate.blocks for step in block.step_ids}
        if evidence.block_id in existing_blocks or existing_steps.intersection(evidence.step_ids):
            raise DataReuseError("A block or observation may verify a candidate only once.")
        blocks = (*candidate.blocks, evidence)
        cumulative = candidate.cumulative_log_likelihood_ratio + evidence.log_likelihood_ratio
        tentative = replace(
            candidate,
            blocks=blocks,
            cumulative_log_likelihood_ratio=cumulative,
        )
        observable = (
            len(blocks) >= self.policy.minimum_blocks
            and tentative.distinct_station_count >= self.policy.minimum_distinct_stations
            and tentative.distinct_height_count >= self.policy.minimum_distinct_heights
            and tentative.distinct_shield_program_count
            >= self.policy.minimum_distinct_shield_programs
        )
        state = CandidateState.PENDING
        if observable and cumulative >= self.policy.support_log_likelihood_ratio:
            state = CandidateState.VERIFIED
        elif observable and cumulative <= self.policy.reject_log_likelihood_ratio:
            state = CandidateState.QUARANTINED
        updated = replace(tentative, state=state)
        self._candidates[key] = updated
        return updated

    def planner_weight(self, candidate: VerificationCandidate) -> float:
        if candidate.state is CandidateState.VERIFIED:
            return self.policy.verified_planner_weight
        if candidate.state is CandidateState.PENDING:
            return self.policy.pending_planner_weight
        return 0.0

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": asdict(self.policy),
            "candidates": [
                {
                    **asdict(candidate),
                    "state": candidate.state.value,
                    "blocks": [asdict(block) for block in candidate.blocks],
                }
                for candidate in self.candidates
            ],
        }

    @classmethod
    def from_state(cls, payload: dict[str, Any]) -> BlockVerificationQueue:
        if payload.get("schema_version") != 1 or not isinstance(payload.get("policy"), dict):
            raise ContractError("Verification queue state has an unsupported schema.")
        queue = cls(VerificationPolicy(**payload["policy"]))
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ContractError("Verification queue state candidates must be an array.")
        for raw in raw_candidates:
            if not isinstance(raw, dict):
                raise ContractError("Verification queue candidate state must be an object.")
            blocks = tuple(BlockEvidence(**block) for block in raw.get("blocks", ()))
            candidate = VerificationCandidate(
                snapshot_id=str(raw["snapshot_id"]),
                snapshot_sha256=str(raw["snapshot_sha256"]),
                candidate_id=str(raw["candidate_id"]),
                data_cutoff_step=int(raw["data_cutoff_step"]),
                state=CandidateState(str(raw["state"])),
                blocks=blocks,
                cumulative_log_likelihood_ratio=float(
                    raw["cumulative_log_likelihood_ratio"]
                ),
            )
            queue._candidates[(candidate.snapshot_id, candidate.candidate_id)] = candidate
        return queue


__all__ = [
    "BlockEvidence",
    "BlockVerificationQueue",
    "CandidateState",
    "VerificationCandidate",
    "VerificationPolicy",
]
