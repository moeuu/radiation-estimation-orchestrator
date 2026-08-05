from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from orchestrator.hashing import canonical_json_bytes, sha256_bytes, write_json_atomic
from orchestrator.hybrid_v2 import (
    ActionDecision,
    CandidateSnapshot,
    HybridMissionController,
    MissionBudget,
    MissionPhase,
    RealizedAction,
)


def _candidates() -> CandidateSnapshot:
    body = {
        "poses": [[1.0, 2.0, 1.5], [2.0, 2.0, 2.0]],
        "costs": [1.0, 2.0],
    }
    return CandidateSnapshot(
        snapshot_id="runtime-candidates-0",
        snapshot_sha256=sha256_bytes(canonical_json_bytes(body)),
        candidate_poses_xyz=((1.0, 2.0, 1.5), (2.0, 2.0, 2.0)),
        travel_costs_s=(1.0, 2.0),
        candidate_path_sha256=("1" * 64, "2" * 64),
        allowed_pair_ids=tuple(range(64)),
        collision_checked=True,
        reachability_filtered=True,
        path_attestation_sha256="a" * 64,
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.snapshot = _candidates()
        self.receipts: dict[str, RealizedAction] = {}
        self.executions = 0
        self.finalizations = 0

    def candidates(self) -> CandidateSnapshot:
        return self.snapshot

    def lookup_receipt(self, decision_id: str) -> RealizedAction | None:
        return self.receipts.get(decision_id)

    def execute_once(self, decision: ActionDecision) -> RealizedAction:
        self.executions += 1
        records = tuple(
            {
                "step_id": index,
                "action_id": 0,
                "station_id": decision.station_id,
                "detector_pose_xyz": list(decision.target_pose_xyz),
                "detector_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "fe_orientation_index": pair_id // 8,
                "pb_orientation_index": pair_id % 8,
                "live_time_s": decision.dwell_time_s,
                "travel_time_s": 1.0 if index == 0 else 0.0,
                "shield_actuation_time_s": 0.25,
                "energy_bin_edges_keV": [0.0, 1.0, 2.0],
                "spectrum_counts": [3, 4],
                "metadata": {
                    "runtime_durable": True,
                    "station_complete": (
                        decision.station_complete
                        and index == len(decision.shield_program_pair_ids) - 1
                    ),
                    "realized_path_sha256": decision.selected_path_sha256,
                },
            }
            for index, pair_id in enumerate(decision.shield_program_pair_ids)
        )
        receipt = RealizedAction(
            decision_id=decision.decision_id,
            runtime_receipt_id="runtime-receipt-0",
            runtime_receipt_sha256=sha256_bytes(canonical_json_bytes(records)),
            records=records,
            measurement_log_prefix_path="/truth-free/measurement-prefix",
            measurement_log_prefix_sha256="9" * 64,
            next_candidates=self.snapshot,
        )
        self.receipts[decision.decision_id] = receipt
        return receipt

    def finalize(self) -> dict[str, object]:
        self.finalizations += 1
        return {"measurement_log_sha256": "b" * 64, "record_count": 2}


class FakePlanner:
    def propose(
        self,
        *,
        data_cutoff_step: int,
        data_cutoff_station: int,
        estimator_prefix_sha256: str,
        candidates: CandidateSnapshot,
    ) -> ActionDecision:
        return ActionDecision(
            decision_id="decision-0",
            estimator_data_cutoff_step=data_cutoff_step,
            estimator_data_cutoff_station=data_cutoff_station,
            estimator_prefix_sha256=estimator_prefix_sha256,
            candidate_snapshot_id=candidates.snapshot_id,
            candidate_snapshot_sha256=candidates.snapshot_sha256,
            candidate_index=0,
            target_pose_xyz=candidates.candidate_poses_xyz[0],
            selected_path_sha256=candidates.candidate_path_sha256[0],
            shield_program_pair_ids=(1, 8),
            dwell_time_s=2.0,
            station_id=0,
            station_complete=True,
        )


class FakeUpdater:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def update_after_append(self, realized: RealizedAction) -> dict[str, object]:
        for record in realized.records:
            assert record["metadata"]["runtime_durable"] is True
            self.records.append(dict(record))
        return {
            "data_cutoff_step": int(realized.records[-1]["step_id"]),
            "estimator_prefix_sha256": "c" * 64,
            "pf_checkpoint_sha256": "d" * 64,
        }

    def finalize_after_publish(self, published: dict[str, object]) -> dict[str, object]:
        assert published["record_count"] == 2
        return {"final_spectral_mle_result_sha256": "f" * 64}


def _controller(
    tmp_path: Path,
    runtime: FakeRuntime,
    updater: FakeUpdater,
) -> HybridMissionController:
    return HybridMissionController(
        mission_id="mission-1",
        state_path=tmp_path / "mission_state.json",
        ledger_path=tmp_path / "mission_ledger.jsonl",
        budget=MissionBudget(
            max_actions=1,
            max_total_time_s=100.0,
            max_live_time_s=100.0,
            max_travel_time_s=100.0,
            max_shield_actuation_time_s=100.0,
        ),
        runtime=runtime,
        planner=FakePlanner(),
        updater=updater,
    )


def test_hybrid_mission_orders_append_before_estimator_update(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    updater = FakeUpdater()
    controller = _controller(tmp_path, runtime, updater)

    state = controller.run()

    assert MissionPhase(state.phase) is MissionPhase.COMPLETE
    assert runtime.executions == 1
    assert runtime.finalizations == 1
    assert len(updater.records) == 2
    event_types = [entry.event_type for entry in controller.ledger.entries]
    assert event_types == [
        "action_proposed",
        "action_realized",
        "observation_appended",
        "estimators_updated",
        "mission_completed",
    ]


def test_hybrid_mission_resumes_proposed_action_exactly_once(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    updater = FakeUpdater()
    first = _controller(tmp_path, runtime, updater)
    first.advance_once()
    assert first.state.phase == MissionPhase.ACTION_PROPOSED.value

    resumed = _controller(tmp_path, runtime, updater)
    state = resumed.run()

    assert state.phase == MissionPhase.COMPLETE.value
    assert runtime.executions == 1


def test_hybrid_mission_recovers_existing_runtime_receipt(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    updater = FakeUpdater()
    first = _controller(tmp_path, runtime, updater)
    first.advance_once()
    decision = ActionDecision(**first.state.proposed_action)  # type: ignore[arg-type]
    receipt = runtime.execute_once(decision)
    assert asdict(receipt)["decision_id"] == decision.decision_id

    resumed = _controller(tmp_path, runtime, updater)
    resumed.run()

    assert runtime.executions == 1


def test_hybrid_mission_recovers_fsynced_ledger_ahead_of_state(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    updater = FakeUpdater()
    first = _controller(tmp_path, runtime, updater)
    stale_state = asdict(first.state)
    first.advance_once()
    assert first.state.phase == MissionPhase.ACTION_PROPOSED.value
    write_json_atomic(
        tmp_path / "mission_state.json",
        stale_state,
        overwrite=True,
    )

    resumed = _controller(tmp_path, runtime, updater)

    assert resumed.state.phase == MissionPhase.ACTION_PROPOSED.value
    assert resumed.state.ledger_head_sha256 == resumed.ledger.head_sha256
    assert resumed.run().phase == MissionPhase.COMPLETE.value
    assert runtime.executions == 1
