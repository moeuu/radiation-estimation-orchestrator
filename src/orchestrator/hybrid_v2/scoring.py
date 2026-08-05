"""Apply validated MLE-owned spectral score artifacts to the verification queue."""

from __future__ import annotations

from orchestrator.contracts import FutureSpectralCandidateScoreInfo, SpectralMLESnapshotInfo
from orchestrator.errors import ContractError

from .verification import BlockEvidence, BlockVerificationQueue, VerificationCandidate


def register_snapshot_candidates(
    queue: BlockVerificationQueue,
    snapshot: SpectralMLESnapshotInfo,
) -> tuple[VerificationCandidate, ...]:
    """Register every frozen MLE candidate exactly once."""
    result: list[VerificationCandidate] = []
    raw_candidates = snapshot.payload["candidates"]
    assert isinstance(raw_candidates, list)
    for raw in raw_candidates:
        assert isinstance(raw, dict)
        result.append(
            queue.register(
                snapshot_id=str(snapshot.payload["snapshot_id"]),
                snapshot_sha256=snapshot.snapshot_sha256,
                candidate_id=str(raw["candidate_id"]),
                data_cutoff_step=snapshot.cutoff_step,
            )
        )
    return tuple(result)


def apply_future_spectral_scores(
    queue: BlockVerificationQueue,
    *,
    snapshot: SpectralMLESnapshotInfo,
    score: FutureSpectralCandidateScoreInfo,
) -> tuple[VerificationCandidate, ...]:
    """Consume each independent block once for each frozen candidate."""
    if score.payload["snapshot_id"] != snapshot.payload["snapshot_id"]:
        raise ContractError("Future score and spectral snapshot identities differ.")
    raw_blocks = score.payload["blocks"]
    raw_candidates = score.payload["candidates"]
    assert isinstance(raw_blocks, list)
    assert isinstance(raw_candidates, list)
    blocks = {
        str(raw["block_id"]): raw
        for raw in raw_blocks
        if isinstance(raw, dict)
    }
    updated: list[VerificationCandidate] = []
    for raw_candidate in raw_candidates:
        assert isinstance(raw_candidate, dict)
        candidate_id = str(raw_candidate["candidate_id"])
        current = next(
            (
                candidate
                for candidate in queue.candidates
                if candidate.snapshot_id == snapshot.payload["snapshot_id"]
                and candidate.candidate_id == candidate_id
            ),
            None,
        )
        if current is None:
            raise ContractError("Future score references an unregistered candidate.")
        if current.state.value != "pending":
            continue
        raw_scores = raw_candidate["block_scores"]
        assert isinstance(raw_scores, list)
        candidate: VerificationCandidate | None = None
        for raw_score in raw_scores:
            assert isinstance(raw_score, dict)
            block_id = str(raw_score["block_id"])
            raw_block = blocks[block_id]
            candidate = queue.corroborate(
                snapshot_id=str(snapshot.payload["snapshot_id"]),
                candidate_id=candidate_id,
                evidence=BlockEvidence(
                    block_id=block_id,
                    station_id=int(raw_block["station_id"]),
                    height_group_id=str(raw_block["height_group_id"]),
                    shield_program_id=str(raw_block["shield_program_id"]),
                    step_ids=tuple(int(value) for value in raw_block["step_ids"]),
                    log_likelihood_ratio=float(raw_score["log_likelihood_ratio"]),
                    score_artifact_sha256=score.score_sha256,
                ),
            )
        if candidate is not None:
            updated.append(candidate)
    return tuple(updated)


__all__ = ["apply_future_spectral_scores", "register_snapshot_candidates"]
