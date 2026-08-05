"""Causality, resume, and independent-block tests for hybrid v2."""

from __future__ import annotations

import numpy as np
import pytest

from orchestrator.errors import DataReuseError
from orchestrator.hybrid_v2 import (
    BlockEvidence,
    BlockVerificationQueue,
    CandidateState,
    SpectralHybridScheduler,
    SpectralPredictiveMonitor,
    SpectralSchedulerPolicy,
    VerificationPolicy,
)


def _signal(
    monitor: SpectralPredictiveMonitor,
    step: int,
    station: int,
    *,
    complete: bool,
    observed: tuple[int, ...] = (5, 2, 0),
    predicted: tuple[float, ...] = (1.0, 1.0, 1.0),
):
    return monitor.record(
        step_id=step,
        station_id=station,
        prediction_data_cutoff_step=step - 1,
        station_complete=complete,
        observed_spectrum=np.asarray(observed, dtype=np.int64),
        predicted_spectrum=np.asarray(predicted, dtype=np.float64),
    )


def test_spectral_trigger_uses_preupdate_prediction_and_resumes() -> None:
    monitor = SpectralPredictiveMonitor()
    policy = SpectralSchedulerPolicy(
        station_interval=10,
        normalized_deviance_threshold=1.0,
        mismatch_streak=1,
    )
    scheduler = SpectralHybridScheduler(policy)

    assert scheduler.consider(_signal(monitor, 0, 0, complete=False)) is None
    trigger = scheduler.consider(_signal(monitor, 1, 0, complete=True))
    assert trigger is not None
    assert trigger.data_cutoff_step == 1

    resumed = SpectralHybridScheduler.from_state(scheduler.to_state())
    assert resumed.consider(_signal(monitor, 2, 1, complete=True)) is not None


def test_spectral_monitor_rejects_postupdate_prediction() -> None:
    monitor = SpectralPredictiveMonitor()
    with pytest.raises(DataReuseError, match="before"):
        monitor.record(
            step_id=3,
            station_id=1,
            prediction_data_cutoff_step=3,
            station_complete=True,
            observed_spectrum=np.asarray([1, 2], dtype=np.int64),
            predicted_spectrum=np.asarray([1.0, 2.0]),
        )


def _block(
    block_id: str,
    station: int,
    height: str,
    shield: str,
    steps: tuple[int, ...],
    llr: float,
) -> BlockEvidence:
    return BlockEvidence(
        block_id=block_id,
        station_id=station,
        height_group_id=height,
        shield_program_id=shield,
        step_ids=steps,
        log_likelihood_ratio=llr,
        score_artifact_sha256="a" * 64,
    )


def test_candidate_requires_distinct_station_and_height_blocks() -> None:
    queue = BlockVerificationQueue(
        VerificationPolicy(
            support_log_likelihood_ratio=3.0,
            reject_log_likelihood_ratio=-3.0,
            minimum_blocks=2,
            minimum_distinct_stations=2,
            minimum_distinct_heights=2,
            minimum_distinct_shield_programs=1,
        )
    )
    candidate = queue.register(
        snapshot_id="snapshot-1",
        snapshot_sha256="b" * 64,
        candidate_id="candidate-1",
        data_cutoff_step=2,
    )
    candidate = queue.corroborate(
        snapshot_id="snapshot-1",
        candidate_id="candidate-1",
        evidence=_block("block-a", 1, "low", "shield-a", (3, 4), 4.0),
    )
    assert candidate.state is CandidateState.PENDING
    assert queue.planner_weight(candidate) < queue.policy.verified_planner_weight

    candidate = queue.corroborate(
        snapshot_id="snapshot-1",
        candidate_id="candidate-1",
        evidence=_block("block-b", 2, "high", "shield-a", (5,), 1.0),
    )
    assert candidate.state is CandidateState.VERIFIED
    assert queue.planner_weight(candidate) == queue.policy.verified_planner_weight

    resumed = BlockVerificationQueue.from_state(queue.to_state())
    assert resumed.candidates == queue.candidates


def test_verification_rejects_cutoff_and_duplicate_steps() -> None:
    queue = BlockVerificationQueue(
        VerificationPolicy(
            support_log_likelihood_ratio=1.0,
            reject_log_likelihood_ratio=-1.0,
            minimum_distinct_stations=1,
        )
    )
    queue.register(
        snapshot_id="snapshot-1",
        snapshot_sha256="b" * 64,
        candidate_id="candidate-1",
        data_cutoff_step=4,
    )
    with pytest.raises(DataReuseError, match="post-cutoff"):
        queue.corroborate(
            snapshot_id="snapshot-1",
            candidate_id="candidate-1",
            evidence=_block("block-old", 1, "low", "shield-a", (4,), 0.1),
        )
