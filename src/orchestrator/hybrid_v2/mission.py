"""Resumable exactly-once hybrid mission state machine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Any, Protocol

from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import canonical_json_bytes, load_json, sha256_bytes, write_json_atomic

from .mission_ledger import MissionLedger


def _sha256_shaped(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class MissionPhase(StrEnum):
    READY = "ready"
    ACTION_PROPOSED = "action_proposed"
    ACTION_REALIZED = "action_realized"
    OBSERVATION_APPENDED = "observation_appended"
    ESTIMATORS_UPDATED = "estimators_updated"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class MissionBudget:
    max_actions: int
    max_total_time_s: float
    max_live_time_s: float
    max_travel_time_s: float
    max_shield_actuation_time_s: float

    def __post_init__(self) -> None:
        if self.max_actions < 1:
            raise ValueError("Mission max_actions must be positive.")
        values = (
            self.max_total_time_s,
            self.max_live_time_s,
            self.max_travel_time_s,
            self.max_shield_actuation_time_s,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("Mission time budgets must be finite and nonnegative.")


@dataclass(frozen=True, slots=True)
class CandidateSnapshot:
    snapshot_id: str
    snapshot_sha256: str
    candidate_poses_xyz: tuple[tuple[float, float, float], ...]
    travel_costs_s: tuple[float, ...]
    candidate_path_sha256: tuple[str, ...]
    allowed_pair_ids: tuple[int, ...]
    collision_checked: bool
    reachability_filtered: bool
    path_attestation_sha256: str

    def __post_init__(self) -> None:
        if not self.candidate_poses_xyz or len(self.candidate_poses_xyz) != len(
            self.travel_costs_s
        ):
            raise ContractError("Runtime candidate poses and travel costs must align.")
        if len(self.candidate_path_sha256) != len(self.candidate_poses_xyz) or any(
            not _sha256_shaped(value) for value in self.candidate_path_sha256
        ):
            raise ContractError("Every runtime candidate requires an exact path hash.")
        if (
            not self.allowed_pair_ids
            or not self.collision_checked
            or not self.reachability_filtered
        ):
            raise ContractError(
                "Runtime candidates require collision and reachability attestations."
            )
        if any(
            len(pose) != 3 or not all(isfinite(value) for value in pose)
            for pose in self.candidate_poses_xyz
        ):
            raise ContractError("Runtime candidate XYZ values must be finite triples.")
        if any(not isfinite(cost) or cost < 0 for cost in self.travel_costs_s):
            raise ContractError("Runtime travel costs must be finite and nonnegative.")
        if (
            tuple(sorted(set(self.allowed_pair_ids))) != self.allowed_pair_ids
            or any(pair < 0 or pair >= 64 for pair in self.allowed_pair_ids)
        ):
            raise ContractError("Runtime shield-pair IDs must be sorted unique values in 0..63.")
        if not _sha256_shaped(self.snapshot_sha256) or not _sha256_shaped(
            self.path_attestation_sha256
        ):
            raise ContractError("Runtime path attestation must be SHA-256 shaped.")


@dataclass(frozen=True, slots=True)
class ActionDecision:
    decision_id: str
    estimator_data_cutoff_step: int
    estimator_data_cutoff_station: int
    estimator_prefix_sha256: str
    candidate_snapshot_id: str
    candidate_snapshot_sha256: str
    candidate_index: int
    target_pose_xyz: tuple[float, float, float]
    selected_path_sha256: str
    shield_program_pair_ids: tuple[int, ...]
    dwell_time_s: float
    station_id: int
    station_complete: bool

    def __post_init__(self) -> None:
        if not self.decision_id or not self.candidate_snapshot_id:
            raise ContractError("Mission decision identities must be nonempty.")
        if self.estimator_data_cutoff_step < -1 or self.estimator_data_cutoff_station < -1:
            raise ContractError("Mission estimator cutoffs must be at least -1.")
        if (self.estimator_data_cutoff_step == -1) != (
            self.estimator_data_cutoff_station == -1
        ):
            raise ContractError("Mission bootstrap step/station cutoffs must agree.")
        if not _sha256_shaped(self.estimator_prefix_sha256) or not _sha256_shaped(
            self.candidate_snapshot_sha256
        ):
            raise ContractError("Mission decision prefix/snapshot hashes are invalid.")
        if self.candidate_index < 0 or self.station_id < 0:
            raise ContractError("Mission candidate and station IDs must be nonnegative.")
        if len(self.target_pose_xyz) != 3 or not all(
            isfinite(value) for value in self.target_pose_xyz
        ):
            raise ContractError("Mission target pose must be a finite XYZ triple.")
        if not _sha256_shaped(self.selected_path_sha256):
            raise ContractError("Mission selected path hash is invalid.")
        if (
            not self.shield_program_pair_ids
            or any(pair < 0 or pair >= 64 for pair in self.shield_program_pair_ids)
        ):
            raise ContractError("Mission shield program pair IDs must be in 0..63.")
        if not isfinite(self.dwell_time_s) or self.dwell_time_s <= 0:
            raise ContractError("Mission dwell time must be finite and positive.")


@dataclass(frozen=True, slots=True)
class RealizedAction:
    decision_id: str
    runtime_receipt_id: str
    runtime_receipt_sha256: str
    records: tuple[dict[str, Any], ...]
    measurement_log_prefix_path: str
    measurement_log_prefix_sha256: str
    next_candidates: CandidateSnapshot


class MissionRuntime(Protocol):
    def candidates(self) -> CandidateSnapshot: ...

    def lookup_receipt(self, decision_id: str) -> RealizedAction | None: ...

    def execute_once(self, decision: ActionDecision) -> RealizedAction: ...

    def finalize(self) -> Mapping[str, object]: ...


class MissionPlanner(Protocol):
    def propose(
        self,
        *,
        data_cutoff_step: int,
        data_cutoff_station: int,
        estimator_prefix_sha256: str,
        candidates: CandidateSnapshot,
    ) -> ActionDecision: ...


class EstimatorUpdater(Protocol):
    def update_after_append(
        self, realized: RealizedAction
    ) -> Mapping[str, object]: ...

    def finalize_after_publish(
        self, published: Mapping[str, object]
    ) -> Mapping[str, object]: ...


@dataclass(slots=True)
class MissionState:
    schema_version: int
    mission_id: str
    phase: str
    completed_actions: int
    last_observed_step: int
    last_observed_station: int
    estimator_prefix_sha256: str
    live_time_s: float
    travel_time_s: float
    shield_actuation_time_s: float
    proposed_action: dict[str, Any] | None
    realized_action: dict[str, Any] | None
    ledger_head_sha256: str


class HybridMissionController:
    """Guarantee decision, actuation, append, then estimator-update ordering."""

    def __init__(
        self,
        *,
        mission_id: str,
        state_path: str | Path,
        ledger_path: str | Path,
        budget: MissionBudget,
        runtime: MissionRuntime,
        planner: MissionPlanner,
        updater: EstimatorUpdater,
    ) -> None:
        self.state_path = Path(state_path).resolve()
        self.ledger = MissionLedger(ledger_path)
        self.budget = budget
        self.runtime = runtime
        self.planner = planner
        self.updater = updater
        if self.state_path.exists():
            payload = load_json(self.state_path)
            self.state = MissionState(**payload)
            if self.state.mission_id != mission_id:
                raise ContractError("Mission state belongs to a different mission.")
            if self.state.ledger_head_sha256 != self.ledger.head_sha256:
                self._recover_state_from_ledger()
        else:
            self.state = MissionState(
                schema_version=1,
                mission_id=mission_id,
                phase=MissionPhase.READY.value,
                completed_actions=0,
                last_observed_step=-1,
                last_observed_station=-1,
                estimator_prefix_sha256="0" * 64,
                live_time_s=0.0,
                travel_time_s=0.0,
                shield_actuation_time_s=0.0,
                proposed_action=None,
                realized_action=None,
                ledger_head_sha256=self.ledger.head_sha256,
            )
            self._save()

    def _recover_state_from_ledger(self) -> None:
        """Replay fsynced events written just before a crashed state-file update."""
        heads = ["0" * 64, *(entry.entry_sha256 for entry in self.ledger.entries)]
        try:
            start = heads.index(self.state.ledger_head_sha256)
        except ValueError as exc:
            raise ContractError("Mission state ledger head is not in the durable chain.") from exc
        pending = self.ledger.entries[start:]
        if not pending:
            raise ContractError("Mission state and ledger heads differ unexpectedly.")
        for entry in pending:
            payload = entry.payload
            if entry.event_type == "action_proposed":
                if self.state.phase not in {
                    MissionPhase.READY.value,
                    MissionPhase.ESTIMATORS_UPDATED.value,
                }:
                    raise ContractError("Cannot recover action proposal from this phase.")
                self.state.phase = MissionPhase.ACTION_PROPOSED.value
                self.state.proposed_action = dict(payload)
                self.state.realized_action = None
            elif entry.event_type == "action_realized":
                if self.state.phase != MissionPhase.ACTION_PROPOSED.value:
                    raise ContractError("Cannot recover realized action from this phase.")
                self.state.phase = MissionPhase.ACTION_REALIZED.value
                self.state.realized_action = dict(payload)
            elif entry.event_type == "observation_appended":
                if self.state.phase != MissionPhase.ACTION_REALIZED.value:
                    raise ContractError("Cannot recover observation append from this phase.")
                self.state.phase = MissionPhase.OBSERVATION_APPENDED.value
            elif entry.event_type == "estimators_updated":
                if self.state.phase != MissionPhase.OBSERVATION_APPENDED.value:
                    raise ContractError("Cannot recover estimator update from this phase.")
                realized = self._require_realized()
                records = realized["records"]
                if not isinstance(records, list) or not records:
                    raise ContractError("Recovered realized action lacks its records.")
                final_record = records[-1]
                if not isinstance(final_record, dict):
                    raise ContractError("Recovered final runtime record is invalid.")
                step = int(final_record["step_id"])
                if int(payload.get("data_cutoff_step", -2)) != step:
                    raise DataReuseError("Recovered estimator cutoff differs from its record.")
                prefix_hash = str(payload.get("estimator_prefix_sha256", ""))
                if len(prefix_hash) != 64:
                    raise ContractError("Recovered estimator prefix hash is invalid.")
                self.state.completed_actions += 1
                self.state.last_observed_step = step
                self.state.last_observed_station = int(final_record["station_id"])
                self.state.estimator_prefix_sha256 = prefix_hash
                self.state.live_time_s += sum(
                    float(record["live_time_s"]) for record in records
                )
                self.state.travel_time_s += sum(
                    float(record["travel_time_s"]) for record in records
                )
                self.state.shield_actuation_time_s += sum(
                    float(record["shield_actuation_time_s"]) for record in records
                )
                self.state.phase = MissionPhase.ESTIMATORS_UPDATED.value
                self.state.proposed_action = None
                self.state.realized_action = None
            elif entry.event_type == "mission_completed":
                if self.state.phase not in {
                    MissionPhase.READY.value,
                    MissionPhase.ESTIMATORS_UPDATED.value,
                }:
                    raise ContractError("Cannot recover mission completion from this phase.")
                self.state.phase = MissionPhase.COMPLETE.value
            else:
                raise ContractError(
                    f"Cannot recover unknown mission ledger event {entry.event_type!r}."
                )
            self.state.ledger_head_sha256 = entry.entry_sha256
        self._save()

    def run(self) -> MissionState:
        """Advance safely from any durable phase until fixed-budget completion."""
        while self.state.phase != MissionPhase.COMPLETE.value:
            if self._budget_exhausted() and self.state.phase in {
                MissionPhase.READY.value,
                MissionPhase.ESTIMATORS_UPDATED.value,
            }:
                self._complete()
                break
            self.advance_once()
        return self.state

    def advance_once(self) -> MissionState:
        phase = MissionPhase(self.state.phase)
        if phase in {MissionPhase.READY, MissionPhase.ESTIMATORS_UPDATED}:
            self._propose()
        elif phase is MissionPhase.ACTION_PROPOSED:
            self._execute_or_recover()
        elif phase is MissionPhase.ACTION_REALIZED:
            self._record_append()
        elif phase is MissionPhase.OBSERVATION_APPENDED:
            self._update_estimators()
        elif phase is MissionPhase.COMPLETE:
            return self.state
        else:  # pragma: no cover - enum exhaustiveness guard
            raise RuntimeError(f"Unsupported mission phase {phase!r}.")
        return self.state

    def _propose(self) -> None:
        candidates = self.runtime.candidates()
        decision = self.planner.propose(
            data_cutoff_step=self.state.last_observed_step,
            data_cutoff_station=self.state.last_observed_station,
            estimator_prefix_sha256=self.state.estimator_prefix_sha256,
            candidates=candidates,
        )
        self._validate_decision(decision, candidates)
        prior_decisions = {
            str(entry.payload.get("decision_id"))
            for entry in self.ledger.entries
            if entry.event_type == "action_proposed"
        }
        if decision.decision_id in prior_decisions:
            raise DataReuseError("A mission action decision ID may be proposed only once.")
        self.ledger.append("action_proposed", asdict(decision))
        self.state.phase = MissionPhase.ACTION_PROPOSED.value
        self.state.proposed_action = asdict(decision)
        self.state.realized_action = None
        self._save()

    def _execute_or_recover(self) -> None:
        decision = ActionDecision(**self._require_proposed())
        realized = self.runtime.lookup_receipt(decision.decision_id)
        if realized is None:
            realized = self.runtime.execute_once(decision)
        self._validate_realized(decision, realized)
        payload = {
            "decision_id": realized.decision_id,
            "runtime_receipt_id": realized.runtime_receipt_id,
            "runtime_receipt_sha256": realized.runtime_receipt_sha256,
            "records": list(realized.records),
            "measurement_log_prefix_path": realized.measurement_log_prefix_path,
            "measurement_log_prefix_sha256": realized.measurement_log_prefix_sha256,
            "next_candidates": asdict(realized.next_candidates),
        }
        self.ledger.append("action_realized", payload)
        self.state.phase = MissionPhase.ACTION_REALIZED.value
        self.state.realized_action = payload
        self._save()

    def _record_append(self) -> None:
        realized = self._require_realized()
        records = realized["records"]
        if not isinstance(records, list) or not records or not all(
            isinstance(record, dict) for record in records
        ):
            raise ContractError("Runtime receipt lacks a nonempty record sequence.")
        step_ids = tuple(int(record["step_id"]) for record in records)
        first_step = self.state.last_observed_step + 1
        expected = tuple(
            range(first_step, first_step + len(records))
        )
        if step_ids != expected:
            raise DataReuseError("Runtime records do not append the next causal steps.")
        records_sha = sha256_bytes(canonical_json_bytes(records))
        self.ledger.append(
            "observation_appended",
            {
                "decision_id": realized["decision_id"],
                "step_ids": list(step_ids),
                "records_sha256": records_sha,
                "runtime_receipt_sha256": realized["runtime_receipt_sha256"],
            },
        )
        self.state.phase = MissionPhase.OBSERVATION_APPENDED.value
        self._save()

    def _update_estimators(self) -> None:
        realized = self._require_realized()
        records = realized["records"]
        assert isinstance(records, list)
        realized_action = RealizedAction(
            decision_id=str(realized["decision_id"]),
            runtime_receipt_id=str(realized["runtime_receipt_id"]),
            runtime_receipt_sha256=str(realized["runtime_receipt_sha256"]),
            records=tuple(dict(record) for record in records),
            measurement_log_prefix_path=str(realized["measurement_log_prefix_path"]),
            measurement_log_prefix_sha256=str(
                realized["measurement_log_prefix_sha256"]
            ),
            next_candidates=self._candidate_from_state(realized["next_candidates"]),
        )
        update = dict(self.updater.update_after_append(realized_action))
        final_record = records[-1]
        assert isinstance(final_record, dict)
        step = int(final_record["step_id"])
        if int(update.get("data_cutoff_step", -2)) != step:
            raise DataReuseError("Estimator update did not consume the newly appended step.")
        prefix_hash = str(update.get("estimator_prefix_sha256", ""))
        if len(prefix_hash) != 64:
            raise ContractError("Estimator update must return its exact prefix SHA-256.")
        self.ledger.append(
            "estimators_updated",
            {**update, "decision_id": realized["decision_id"]},
        )
        self.state.completed_actions += 1
        self.state.last_observed_step = step
        self.state.last_observed_station = int(final_record["station_id"])
        self.state.estimator_prefix_sha256 = prefix_hash
        self.state.live_time_s += sum(float(record["live_time_s"]) for record in records)
        self.state.travel_time_s += sum(float(record["travel_time_s"]) for record in records)
        self.state.shield_actuation_time_s += sum(
            float(record["shield_actuation_time_s"]) for record in records
        )
        self.state.phase = MissionPhase.ESTIMATORS_UPDATED.value
        self.state.proposed_action = None
        self.state.realized_action = None
        self._save()

    def _complete(self) -> None:
        published = dict(self.runtime.finalize())
        final_estimate = dict(self.updater.finalize_after_publish(published))
        self.ledger.append(
            "mission_completed",
            {"runtime": published, "final_estimate": final_estimate},
        )
        self.state.phase = MissionPhase.COMPLETE.value
        self._save()

    def _validate_decision(
        self,
        decision: ActionDecision,
        candidates: CandidateSnapshot,
    ) -> None:
        if decision.estimator_data_cutoff_step != self.state.last_observed_step:
            raise DataReuseError("Planner decision is not bound to the current PF/MLE cutoff.")
        if decision.estimator_prefix_sha256 != self.state.estimator_prefix_sha256:
            raise DataReuseError("Planner decision uses a stale estimator prefix.")
        if (
            decision.candidate_snapshot_id != candidates.snapshot_id
            or decision.candidate_snapshot_sha256 != candidates.snapshot_sha256
        ):
            raise ContractError("Planner decision uses a different runtime candidate snapshot.")
        if not 0 <= decision.candidate_index < len(candidates.candidate_poses_xyz):
            raise ContractError("Planner selected an invalid candidate index.")
        if tuple(decision.target_pose_xyz) != tuple(
            candidates.candidate_poses_xyz[decision.candidate_index]
        ):
            raise ContractError("Planner target XYZ differs from the selected safe candidate.")
        if decision.selected_path_sha256 != candidates.candidate_path_sha256[
            decision.candidate_index
        ]:
            raise ContractError("Planner selected-path identity differs from the runtime path.")
        if any(
            pair_id not in candidates.allowed_pair_ids
            for pair_id in decision.shield_program_pair_ids
        ):
            raise ContractError("Planner selected a disallowed shield pair.")
        if not isfinite(decision.dwell_time_s) or decision.dwell_time_s <= 0:
            raise ContractError("Planner dwell time must be finite and positive.")

    @staticmethod
    def _validate_realized(decision: ActionDecision, realized: RealizedAction) -> None:
        if realized.decision_id != decision.decision_id:
            raise ContractError("Runtime receipt belongs to a different decision.")
        if len(realized.records) != len(decision.shield_program_pair_ids):
            raise ContractError("Runtime realized a different shield-program length.")
        for index, (record, expected_pair) in enumerate(
            zip(realized.records, decision.shield_program_pair_ids, strict=True)
        ):
            if tuple(float(value) for value in record["detector_pose_xyz"]) != tuple(
                decision.target_pose_xyz
            ):
                raise ContractError("Realized detector pose differs from the selected action.")
            pair_id = int(record["fe_orientation_index"]) * 8 + int(
                record["pb_orientation_index"]
            )
            if pair_id != expected_pair:
                raise ContractError("Realized shield program differs from the decision.")
            if int(record["station_id"]) != decision.station_id:
                raise ContractError("Realized station differs from the selected action.")
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                raise ContractError("Realized action lacks runtime metadata.")
            expected_complete = decision.station_complete and index == len(realized.records) - 1
            if metadata.get("station_complete") is not expected_complete:
                raise ContractError(
                    "Realized station-complete marker differs from the shield program."
                )
            if metadata.get("realized_path_sha256") != decision.selected_path_sha256:
                raise ContractError("Runtime realized a different collision-attested path.")

    def _budget_exhausted(self) -> bool:
        total = (
            self.state.live_time_s
            + self.state.travel_time_s
            + self.state.shield_actuation_time_s
        )
        return (
            self.state.completed_actions >= self.budget.max_actions
            or total >= self.budget.max_total_time_s
            or self.state.live_time_s >= self.budget.max_live_time_s
            or self.state.travel_time_s >= self.budget.max_travel_time_s
            or self.state.shield_actuation_time_s
            >= self.budget.max_shield_actuation_time_s
        )

    def _require_proposed(self) -> dict[str, Any]:
        if self.state.proposed_action is None:
            raise ContractError("Mission phase requires a durable proposed action.")
        return self.state.proposed_action

    def _require_realized(self) -> dict[str, Any]:
        if self.state.realized_action is None:
            raise ContractError("Mission phase requires a durable realized action.")
        return self.state.realized_action

    @staticmethod
    def _candidate_from_state(value: object) -> CandidateSnapshot:
        if not isinstance(value, dict):
            raise ContractError("Mission state contains invalid runtime candidates.")
        try:
            return CandidateSnapshot(
                snapshot_id=str(value["snapshot_id"]),
                snapshot_sha256=str(value["snapshot_sha256"]),
                candidate_poses_xyz=tuple(
                    tuple(float(component) for component in pose)
                    for pose in value["candidate_poses_xyz"]
                ),
                travel_costs_s=tuple(float(cost) for cost in value["travel_costs_s"]),
                candidate_path_sha256=tuple(
                    str(item) for item in value["candidate_path_sha256"]
                ),
                allowed_pair_ids=tuple(int(pair) for pair in value["allowed_pair_ids"]),
                collision_checked=value["collision_checked"] is True,
                reachability_filtered=value["reachability_filtered"] is True,
                path_attestation_sha256=str(value["path_attestation_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("Mission state runtime candidates are malformed.") from exc

    def _save(self) -> None:
        self.state.ledger_head_sha256 = self.ledger.head_sha256
        write_json_atomic(self.state_path, asdict(self.state), overwrite=True)


__all__ = [
    "ActionDecision",
    "CandidateSnapshot",
    "EstimatorUpdater",
    "HybridMissionController",
    "MissionBudget",
    "MissionPhase",
    "MissionPlanner",
    "MissionRuntime",
    "MissionState",
    "RealizedAction",
]
