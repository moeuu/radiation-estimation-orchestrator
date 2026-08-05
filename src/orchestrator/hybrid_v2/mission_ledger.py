"""Durable hash-chained ledger for resumable hybrid missions."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.errors import ContractError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes

_GENESIS = "0" * 64


@dataclass(frozen=True, slots=True)
class MissionLedgerEntry:
    sequence: int
    event_type: str
    payload: dict[str, Any]
    previous_sha256: str
    entry_sha256: str


class MissionLedger:
    """Append durable events and validate the complete chain on resume."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries = list(self._load()) if self.path.exists() else []

    @property
    def entries(self) -> tuple[MissionLedgerEntry, ...]:
        return tuple(self._entries)

    @property
    def head_sha256(self) -> str:
        return self._entries[-1].entry_sha256 if self._entries else _GENESIS

    def append(self, event_type: str, payload: dict[str, Any]) -> MissionLedgerEntry:
        if not event_type or not isinstance(payload, dict):
            raise ContractError("Mission ledger requires an event type and object payload.")
        sequence = len(self._entries)
        body = {
            "schema_version": 1,
            "sequence": sequence,
            "event_type": event_type,
            "payload": payload,
            "previous_sha256": self.head_sha256,
        }
        digest = sha256_bytes(canonical_json_bytes(body))
        row = {**body, "entry_sha256": digest}
        encoded = (
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        entry = MissionLedgerEntry(
            sequence=sequence,
            event_type=event_type,
            payload=payload,
            previous_sha256=body["previous_sha256"],
            entry_sha256=digest,
        )
        self._entries.append(entry)
        return entry

    def _load(self) -> tuple[MissionLedgerEntry, ...]:
        result: list[MissionLedgerEntry] = []
        previous = _GENESIS
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"Invalid mission ledger line {line_number}.") from exc
            if not isinstance(row, dict) or row.get("schema_version") != 1:
                raise ContractError("Mission ledger has an unsupported row schema.")
            body = {key: value for key, value in row.items() if key != "entry_sha256"}
            digest = sha256_bytes(canonical_json_bytes(body))
            if row.get("entry_sha256") != digest or row.get("previous_sha256") != previous:
                raise ContractError("Mission ledger hash chain is invalid.")
            if row.get("sequence") != len(result):
                raise ContractError("Mission ledger sequence is not contiguous.")
            payload = row.get("payload")
            if not isinstance(payload, dict) or not isinstance(row.get("event_type"), str):
                raise ContractError("Mission ledger event is malformed.")
            result.append(
                MissionLedgerEntry(
                    sequence=len(result),
                    event_type=row["event_type"],
                    payload=payload,
                    previous_sha256=previous,
                    entry_sha256=digest,
                )
            )
            previous = digest
        return tuple(result)


__all__ = ["MissionLedger", "MissionLedgerEntry"]
