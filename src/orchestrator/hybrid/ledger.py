"""Hash-chained append-only observation-use ledger for hybrid feedback."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import canonical_json_bytes, json_safe, sha256_bytes

from .directives import PFDirective, PFDirectiveReceipt
from .snapshot import MLESnapshot

GENESIS_EVENT_SHA256 = "0" * 64


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    """One immutable event whose hash commits to the entire preceding chain."""

    event_index: int
    event_type: str
    event_id: str
    previous_event_sha256: str
    payload: Mapping[str, object]
    event_sha256: str

    @classmethod
    def create(
        cls,
        *,
        event_index: int,
        event_type: str,
        event_id: str,
        previous_event_sha256: str,
        payload: Mapping[str, object],
    ) -> LedgerEvent:
        """Create a canonical event hash from every non-hash event field."""
        safe_payload = MappingProxyType(dict(json_safe(payload)))  # type: ignore[arg-type]
        body = {
            "schema_version": 1,
            "event_index": int(event_index),
            "event_type": event_type,
            "event_id": event_id,
            "previous_event_sha256": previous_event_sha256,
            "payload": dict(safe_payload),
        }
        return cls(
            event_index=int(event_index),
            event_type=event_type,
            event_id=event_id,
            previous_event_sha256=previous_event_sha256,
            payload=safe_payload,
            event_sha256=sha256_bytes(canonical_json_bytes(body)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the ledger-summary event representation."""
        return {
            "schema_version": 1,
            "event_index": self.event_index,
            "event_type": self.event_type,
            "event_id": self.event_id,
            "previous_event_sha256": self.previous_event_sha256,
            "payload": dict(self.payload),
            "event_sha256": self.event_sha256,
        }


class ObservationUseLedger:
    """Record snapshot, directive, receipt, and future corroboration use exactly once."""

    def __init__(
        self,
        *,
        source_run_id: str,
        station_boundary_schedule_sha256: str,
        ledger_id: str | None = None,
    ) -> None:
        if not source_run_id or len(station_boundary_schedule_sha256) != 64:
            raise ContractError("Ledger requires a run ID and station-boundary schedule hash.")
        self.source_run_id = source_run_id
        self.station_boundary_schedule_sha256 = station_boundary_schedule_sha256
        identity = sha256_bytes(
            canonical_json_bytes(
                {"source_run_id": source_run_id, "schedule": station_boundary_schedule_sha256}
            )
        )
        self.ledger_id = ledger_id or f"ledger-{identity[:20]}"
        self._events: list[LedgerEvent] = []
        self._event_ids: set[str] = set()
        self._snapshots: dict[str, MLESnapshot] = {}
        self._directives: dict[str, PFDirective] = {}
        self._receipts: dict[str, PFDirectiveReceipt] = {}
        self._corroboration_steps: dict[tuple[str, str], set[int]] = {}

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        """Return the immutable event sequence."""
        return tuple(self._events)

    def _append(self, *, event_type: str, event_id: str, payload: Mapping[str, object]) -> None:
        if event_id in self._event_ids:
            raise DataReuseError(f"Ledger event ID {event_id!r} was already appended.")
        previous = self._events[-1].event_sha256 if self._events else GENESIS_EVENT_SHA256
        event = LedgerEvent.create(
            event_index=len(self._events),
            event_type=event_type,
            event_id=event_id,
            previous_event_sha256=previous,
            payload=payload,
        )
        self._events.append(event)
        self._event_ids.add(event_id)

    def register_snapshot(self, snapshot: MLESnapshot) -> None:
        """Record one exact-prefix MLE snapshot."""
        if snapshot.prefix.source_run_id != self.source_run_id:
            raise DataReuseError("Snapshot and ledger are bound to different source runs.")
        if (
            snapshot.prefix.station_boundary_schedule_sha256
            != self.station_boundary_schedule_sha256
        ):
            raise DataReuseError("Snapshot and ledger use different station-boundary schedules.")
        if snapshot.snapshot_id in self._snapshots:
            raise DataReuseError(f"Snapshot {snapshot.snapshot_id} is already registered.")
        self._snapshots[snapshot.snapshot_id] = snapshot
        self._append(
            event_type="snapshot_registered",
            event_id=f"snapshot-event-{snapshot.snapshot_id}",
            payload={
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_sha256": snapshot.sha256,
                "data_cutoff_step": snapshot.prefix.data_cutoff_step,
                "data_cutoff_station": snapshot.prefix.data_cutoff_station,
                "covered_step_ids": list(snapshot.prefix.covered_step_ids),
                "cutoff_station_complete": True,
            },
        )

    def issue_directive(self, directive: PFDirective) -> None:
        """Record one directive after its source snapshot has been registered."""
        if directive.snapshot.snapshot_id not in self._snapshots:
            raise DataReuseError("A directive cannot precede its registered snapshot.")
        if directive.directive_id in self._directives:
            raise DataReuseError(f"Directive {directive.directive_id} is already issued.")
        self._directives[directive.directive_id] = directive
        self._append(
            event_type="directive_issued",
            event_id=f"directive-event-{directive.directive_id}",
            payload={
                "directive_id": directive.directive_id,
                "directive_sha256": directive.sha256,
                "directive_kind": directive.kind.value,
                "snapshot_id": directive.snapshot.snapshot_id,
                "data_cutoff_step": directive.snapshot.prefix.data_cutoff_step,
                "corroboration_min_step": directive.snapshot.prefix.corroboration_min_step,
                "proposal_ids": [proposal.proposal_id for proposal in directive.proposals],
                "direct_mle_objective_reweight": False,
                "hard_prune_authorized": False,
            },
        )

    def record_receipt(self, receipt: PFDirectiveReceipt) -> None:
        """Record application once and reject a second receipt for the same directive."""
        directive_id = receipt.directive.directive_id
        expected = self._directives.get(directive_id)
        if expected is None:
            raise DataReuseError("A directive receipt cannot precede directive issuance.")
        if receipt.directive.sha256 != expected.sha256:
            raise DataReuseError("Receipt is bound to different directive semantics.")
        if directive_id in self._receipts:
            raise DataReuseError(f"Directive {directive_id} already has an application receipt.")
        self._receipts[directive_id] = receipt
        self._append(
            event_type="directive_receipt",
            event_id=f"receipt-event-{receipt.receipt_id}",
            payload=receipt.to_dict(),
        )

    def record_corroboration(
        self,
        *,
        directive_id: str,
        proposal_id: str,
        snapshot_id: str,
        snapshot_candidate_id: str,
        step_id: int,
        station_id: int,
        log_predictive_likelihood_ratio: float,
        future_score_sha256: str,
        current_covered_records_sha256: str,
        state: str,
    ) -> None:
        """Record independent evidence only when ``step_id`` is strictly post-cutoff."""
        directive = self._directives.get(directive_id)
        if directive is None:
            raise DataReuseError("Corroboration references an unissued directive.")
        if directive_id not in self._receipts:
            raise DataReuseError("Corroboration cannot precede directive application receipt.")
        if self._receipts[directive_id].status != "applied":
            raise DataReuseError("A rejected directive cannot consume corroboration evidence.")
        proposal_ids = {proposal.proposal_id for proposal in directive.proposals}
        if proposal_id not in proposal_ids:
            raise DataReuseError("Corroboration references an unknown directive proposal.")
        if snapshot_id != directive.snapshot.snapshot_id:
            raise DataReuseError("Corroboration snapshot differs from its directive.")
        proposal = next(item for item in directive.proposals if item.proposal_id == proposal_id)
        if snapshot_candidate_id != proposal.snapshot_candidate_id:
            raise DataReuseError("Corroboration candidate differs from its directive proposal.")
        cutoff = directive.snapshot.prefix.data_cutoff_step
        if step_id <= cutoff:
            raise DataReuseError("Corroboration must use a step strictly after the MLE cutoff.")
        if station_id < 0 or not isfinite(log_predictive_likelihood_ratio):
            raise ContractError("Corroboration station and log predictive ratio are invalid.")
        for label, digest in (
            ("future_score_sha256", future_score_sha256),
            ("current_covered_records_sha256", current_covered_records_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ContractError(f"Corroboration {label} is invalid.")
        key = (directive_id, proposal_id)
        seen = self._corroboration_steps.setdefault(key, set())
        if step_id in seen:
            raise DataReuseError("An observation may corroborate a candidate at most once.")
        seen.add(step_id)
        identity = {
            "directive_id": directive_id,
            "proposal_id": proposal_id,
            "step_id": step_id,
        }
        event_id = f"corroboration-{sha256_bytes(canonical_json_bytes(identity))[:20]}"
        self._append(
            event_type="corroboration",
            event_id=event_id,
            payload={
                **identity,
                "snapshot_id": snapshot_id,
                "snapshot_candidate_id": snapshot_candidate_id,
                "station_id": int(station_id),
                "log_predictive_likelihood_ratio": float(log_predictive_likelihood_ratio),
                "evidence_family": "frozen_count_snapshot_cluster_log_predictive_ratio",
                "future_score_sha256": future_score_sha256,
                "current_covered_records_sha256": current_covered_records_sha256,
                "candidate_state": state,
                "data_cutoff_step": cutoff,
                "future_only": True,
            },
        )

    def summary(self) -> dict[str, object]:
        """Return a self-verifying hash-chain summary contract."""
        last_hash = self._events[-1].event_sha256 if self._events else GENESIS_EVENT_SHA256
        return {
            "schema_version": 1,
            "ledger_id": self.ledger_id,
            "source_run_id": self.source_run_id,
            "station_boundary_schedule_sha256": self.station_boundary_schedule_sha256,
            "genesis_event_sha256": GENESIS_EVENT_SHA256,
            "event_count": len(self._events),
            "last_event_sha256": last_hash,
            "events": [event.to_dict() for event in self._events],
            "safety_summary": {
                "all_directives_once_only": True,
                "all_corroboration_future_only": True,
                "direct_mle_objective_reweight_performed": False,
                "hard_prune_performed": False,
            },
        }
