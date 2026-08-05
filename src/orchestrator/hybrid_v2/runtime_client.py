"""Strict JSON-lines client for a resumable shared-runtime session v2."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from orchestrator.adapters.base import (
    PRODUCTION_ALLOWED_DIRTY_PREFIXES,
    EstimatorPin,
    verify_repository_revision,
)
from orchestrator.errors import ContractError
from orchestrator.hashing import (
    canonical_json_bytes,
    directory_inventory,
    inventory_digest,
    sha256_bytes,
    sha256_file,
)

from .mission import ActionDecision, CandidateSnapshot, RealizedAction

RUNTIME_EVENT_PREFIX = "resumable-adaptive-session "
DEFAULT_RESUMABLE_RUNTIME_COMMAND = (
    "uv",
    "run",
    "--project",
    "{repository}",
    "python",
    "-m",
    "runtime.cli",
    "run-resumable-adaptive-session",
    "{scenario}",
    "--session-state-dir",
    "{session_state_dir}",
)


class ResumableAdaptiveRuntimeClient:
    """Execute actions only through an exactly-once, receipt-queryable runtime."""

    def __init__(
        self,
        *,
        repository: str | Path,
        revision: str,
        scenario: str | Path,
        session_state_dir: str | Path,
        transcript_path: str | Path,
        timeout_s: float = 300.0,
        require_3d_candidate_diversity: bool = True,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.scenario = Path(scenario).resolve()
        self.session_state_dir = Path(session_state_dir).resolve()
        self.transcript_path = Path(transcript_path).resolve()
        self.timeout_s = float(timeout_s)
        self.require_3d_candidate_diversity = bool(require_3d_candidate_diversity)
        pin = EstimatorPin(
            name="simulation_runtime",
            repository="shared-runtime",
            revision=revision,
            revision_type="commit",
            release_tag=None,
            local_path_hint=None,
            expected_measurement_log_schema_version=2,
            expected_result_schema_version=1,
        )
        observed_revision, dirty = verify_repository_revision(
            self.repository,
            pin,
            require_clean=True,
            allowed_dirty_prefixes=PRODUCTION_ALLOWED_DIRTY_PREFIXES,
        )
        self.requested_revision = revision
        self.observed_revision = observed_revision
        self.dirty_worktree = dirty
        if self.scenario.is_symlink() or not self.scenario.is_file():
            raise ContractError("Resumable runtime scenario is missing or a symlink.")
        self.session_state_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        command = tuple(
            token.format(
                repository=self.repository,
                scenario=self.scenario,
                session_state_dir=self.session_state_dir,
            )
            for token in DEFAULT_RESUMABLE_RUNTIME_COMMAND
        )
        self.command = command
        environment = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "LANG", "LC_ALL", "UV_CACHE_DIR")
            if key in os.environ
        }
        environment.update(
            {
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "RSE_TRUTH_ACCESS": "runtime-private",
            }
        )
        self._stderr_path = self.session_state_dir / "runtime.stderr.log"
        self._stderr = self._stderr_path.open("ab")
        self._process = subprocess.Popen(
            command,
            cwd=self.repository,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("Could not open resumable runtime protocol pipes.")
        self._input: TextIO = self._process.stdin
        self._output: TextIO = self._process.stdout
        try:
            ready = self._read_event()
            if (
                ready.get("type") != "ready"
                or ready.get("schema_version") != 2
                or ready.get("resume_capability")
                != "exactly_once_receipt_lookup"
            ):
                raise ContractError(
                    "Live hybrid requires resumable adaptive runtime protocol v2."
                )
            self.session_id = str(ready["session_id"])
            self._candidates = self._parse_candidates(ready["candidates"])
            if self.require_3d_candidate_diversity and len(
                {pose[2] for pose in self._candidates.candidate_poses_xyz}
            ) < 2:
                raise ContractError(
                    "Live 3-D hybrid requires runtime candidates at multiple detector "
                    "heights."
                )
        except BaseException:
            self.close(abort=True)
            raise

    def candidates(self) -> CandidateSnapshot:
        return self._candidates

    def lookup_receipt(self, decision_id: str) -> RealizedAction | None:
        event = self._request({"type": "lookup_receipt", "decision_id": decision_id})
        if event.get("type") == "receipt_missing":
            if event.get("decision_id") != decision_id:
                raise ContractError("Runtime receipt lookup response has wrong decision ID.")
            return None
        realized = self._parse_receipt(event, expected_decision_id=decision_id)
        self._candidates = realized.next_candidates
        return realized

    def execute_once(self, decision: ActionDecision) -> RealizedAction:
        event = self._request(
            {
                "type": "execute_once",
                "session_id": self.session_id,
                "decision": asdict(decision),
            }
        )
        realized = self._parse_receipt(event, expected_decision_id=decision.decision_id)
        self._candidates = realized.next_candidates
        return realized

    def finalize(self) -> dict[str, object]:
        event = self._request({"type": "finalize", "session_id": self.session_id})
        if event.get("type") != "published" or event.get("schema_version") != 2:
            raise ContractError("Runtime finalize did not publish MeasurementLog v2.")
        self.close(abort=False)
        return event

    def audit_record(self) -> dict[str, object]:
        """Return command, revision, transcript, and durable session hashes."""
        inventory = directory_inventory(self.session_state_dir)
        return {
            "repository_path": self.repository.as_posix(),
            "requested_revision": self.requested_revision,
            "observed_revision": self.observed_revision,
            "dirty_worktree": self.dirty_worktree,
            "command": list(self.command),
            "scenario_sha256": sha256_file(self.scenario),
            "session_id": self.session_id,
            "transcript_path": self.transcript_path.as_posix(),
            "transcript_sha256": sha256_file(self.transcript_path),
            "stderr_path": self._stderr_path.as_posix(),
            "stderr_sha256": sha256_file(self._stderr_path),
            "session_artifact_inventory": inventory,
            "session_artifact_sha256": inventory_digest(inventory),
        }

    def close(self, *, abort: bool = False) -> None:
        if self._process.poll() is None:
            if abort:
                try:
                    self._write_request({"type": "abort", "session_id": self.session_id})
                except (AttributeError, BrokenPipeError, OSError):
                    pass
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self._process.kill()
                self._process.wait()
        self._stderr.close()

    def _request(self, payload: dict[str, object]) -> dict[str, Any]:
        self._write_request(payload)
        return self._read_event()

    def _write_request(self, payload: dict[str, object]) -> None:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._input.write(encoded + "\n")
        self._input.flush()
        self._transcript("request", payload)

    def _read_event(self) -> dict[str, Any]:
        selector = selectors.DefaultSelector()
        selector.register(self._output, selectors.EVENT_READ)
        try:
            if not selector.select(self.timeout_s):
                raise TimeoutError("Timed out waiting for resumable runtime event.")
            line = self._output.readline()
        finally:
            selector.close()
        if not line:
            raise ContractError("Resumable runtime exited before returning an event.")
        if not line.startswith(RUNTIME_EVENT_PREFIX):
            raise ContractError("Resumable runtime emitted an unframed protocol line.")
        try:
            event = json.loads(line[len(RUNTIME_EVENT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise ContractError("Resumable runtime event is invalid JSON.") from exc
        if not isinstance(event, dict):
            raise ContractError("Resumable runtime event must be an object.")
        self._transcript("event", event)
        return event

    def _parse_receipt(
        self,
        event: dict[str, Any],
        *,
        expected_decision_id: str,
    ) -> RealizedAction:
        if event.get("type") != "action_receipt" or event.get("schema_version") != 2:
            raise ContractError("Runtime did not return an action receipt v2.")
        if event.get("decision_id") != expected_decision_id:
            raise ContractError("Runtime action receipt has wrong decision ID.")
        records = event.get("records")
        if not isinstance(records, list) or not records or not all(
            isinstance(record, dict) for record in records
        ):
            raise ContractError("Runtime action receipt lacks durable station records.")
        prefix_path = event.get("measurement_log_prefix_path")
        prefix_sha = event.get("measurement_log_prefix_sha256")
        if not isinstance(prefix_path, str) or not prefix_path:
            raise ContractError("Runtime receipt lacks its durable MeasurementLog prefix path.")
        if (
            not isinstance(prefix_sha, str)
            or len(prefix_sha) != 64
            or any(character not in "0123456789abcdef" for character in prefix_sha)
        ):
            raise ContractError("Runtime receipt lacks its MeasurementLog prefix hash.")
        candidates = self._parse_candidates(event.get("candidates"))
        receipt_body = {
            "decision_id": expected_decision_id,
            "runtime_receipt_id": event.get("runtime_receipt_id"),
            "records": records,
            "measurement_log_prefix_path": prefix_path,
            "measurement_log_prefix_sha256": prefix_sha,
            "candidate_snapshot_sha256": candidates.snapshot_sha256,
        }
        receipt_sha = sha256_bytes(canonical_json_bytes(receipt_body))
        if event.get("runtime_receipt_sha256") != receipt_sha:
            raise ContractError("Runtime action receipt hash is invalid.")
        return RealizedAction(
            decision_id=expected_decision_id,
            runtime_receipt_id=str(event["runtime_receipt_id"]),
            runtime_receipt_sha256=receipt_sha,
            records=tuple(dict(record) for record in records),
            measurement_log_prefix_path=prefix_path,
            measurement_log_prefix_sha256=prefix_sha,
            next_candidates=candidates,
        )

    @staticmethod
    def _parse_candidates(value: object) -> CandidateSnapshot:
        if not isinstance(value, dict):
            raise ContractError("Runtime candidates must be an object.")
        body = {
            "candidate_poses_xyz": value.get("candidate_poses_xyz"),
            "travel_costs_s": value.get("travel_costs_s"),
            "candidate_path_sha256": value.get("candidate_path_sha256"),
            "allowed_pair_ids": value.get("allowed_pair_ids"),
            "collision_checked": value.get("collision_checked"),
            "reachability_filtered": value.get("reachability_filtered"),
            "path_attestation_sha256": value.get("path_attestation_sha256"),
        }
        expected_hash = sha256_bytes(canonical_json_bytes(body))
        if value.get("snapshot_sha256") != expected_hash:
            raise ContractError("Runtime candidate snapshot hash is invalid.")
        try:
            if value["collision_checked"] is not True or value[
                "reachability_filtered"
            ] is not True:
                raise ContractError(
                    "Runtime candidates lack collision/reachability authority."
                )
            return CandidateSnapshot(
                snapshot_id=str(value["snapshot_id"]),
                snapshot_sha256=expected_hash,
                candidate_poses_xyz=tuple(
                    tuple(float(component) for component in pose)
                    for pose in value["candidate_poses_xyz"]
                ),
                travel_costs_s=tuple(float(cost) for cost in value["travel_costs_s"]),
                candidate_path_sha256=tuple(
                    str(item) for item in value["candidate_path_sha256"]
                ),
                allowed_pair_ids=tuple(int(pair) for pair in value["allowed_pair_ids"]),
                collision_checked=True,
                reachability_filtered=True,
                path_attestation_sha256=str(value["path_attestation_sha256"]),
            )
        except ContractError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("Runtime candidate snapshot is malformed.") from exc

    def _transcript(self, direction: str, payload: dict[str, object]) -> None:
        row = {
            "direction": direction,
            "payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
            "payload": payload,
        }
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())


__all__ = [
    "DEFAULT_RESUMABLE_RUNTIME_COMMAND",
    "RUNTIME_EVENT_PREFIX",
    "ResumableAdaptiveRuntimeClient",
]
