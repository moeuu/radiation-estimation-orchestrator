"""Exact causal MeasurementLog prefix and station-boundary declarations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from orchestrator.contracts import MeasurementLogInfo
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes

_SHA256_LENGTH = 64


def _require_sha256(value: str, *, label: str) -> str:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest.")
    return value


@dataclass(frozen=True, slots=True)
class StationBoundarySchedule:
    """Predeclared, hash-bound station ends for causal offline replay.

    The schedule is part of the estimator input contract. A controller must not infer a
    completed station by peeking at a later MeasurementLog record.
    """

    source_run_id: str
    station_end_steps: tuple[tuple[int, int], ...]
    schedule_sha256: str

    @classmethod
    def create(
        cls,
        *,
        source_run_id: str,
        station_end_steps: tuple[tuple[int, int], ...],
    ) -> StationBoundarySchedule:
        """Create and hash a monotone station-to-terminal-step declaration."""
        if not source_run_id:
            raise ContractError("Station-boundary schedule requires source_run_id.")
        if not station_end_steps:
            raise ContractError("Station-boundary schedule must not be empty.")
        normalized = tuple((int(station), int(step)) for station, step in station_end_steps)
        stations = tuple(station for station, _ in normalized)
        steps = tuple(step for _, step in normalized)
        if any(value < 0 for value in (*stations, *steps)):
            raise ContractError("Station-boundary schedule values must be nonnegative.")
        if any(right <= left for left, right in pairwise(stations)):
            raise ContractError("Schedule station IDs must be strictly increasing.")
        if any(right <= left for left, right in pairwise(steps)):
            raise ContractError("Schedule terminal step IDs must be strictly increasing.")
        body = {
            "schema_version": 1,
            "source_run_id": source_run_id,
            "station_end_steps": [
                {"station_id": station, "terminal_step_id": step} for station, step in normalized
            ],
        }
        return cls(
            source_run_id=source_run_id,
            station_end_steps=normalized,
            schedule_sha256=sha256_bytes(canonical_json_bytes(body)),
        )

    @classmethod
    def from_measurement_log(cls, log: MeasurementLogInfo) -> StationBoundarySchedule:
        """Build a declaration during log production or trusted preflight, not PF replay."""
        boundaries: list[tuple[int, int]] = []
        for index, (step, station) in enumerate(zip(log.step_ids, log.station_ids, strict=True)):
            at_end = index + 1 == log.record_count
            if at_end or log.station_ids[index + 1] != station:
                boundaries.append((station, step))
        return cls.create(
            source_run_id=str(log.manifest["run_id"]),
            station_end_steps=tuple(boundaries),
        )

    def terminal_step(self, station_id: int) -> int:
        """Return the declared terminal step for a station."""
        for station, step in self.station_end_steps:
            if station == station_id:
                return step
        raise DataReuseError(f"Station {station_id} is absent from the boundary schedule.")

    def asserts_complete(self, *, station_id: int, step_id: int) -> bool:
        """Return whether an observed marker exactly matches the declaration."""
        return self.terminal_step(station_id) == step_id

    def covered_sha256(self, *, cutoff_step: int) -> str:
        """Hash only station declarations visible through a cutoff."""
        covered = tuple(pair for pair in self.station_end_steps if pair[1] <= cutoff_step)
        if not covered or covered[-1][1] != cutoff_step:
            raise DataReuseError("Cutoff is absent from the covered station-boundary schedule.")
        payload = {
            "schema_version": 1,
            "source_run_id": self.source_run_id,
            "station_end_steps": [
                {"station_id": station, "terminal_step_id": step} for station, step in covered
            ],
        }
        return sha256_bytes(canonical_json_bytes(payload))


@dataclass(frozen=True, slots=True)
class MeasurementPrefix:
    """An exact, station-complete prefix presented to one batch MLE fit."""

    source_run_id: str
    prefix_measurement_log_sha256: str
    covered_records_sha256: str
    station_boundary_schedule_sha256: str
    covered_station_boundaries_sha256: str
    covered_step_ids: tuple[int, ...]
    covered_station_ids: tuple[int, ...]
    data_cutoff_step: int
    data_cutoff_station: int
    cutoff_station_complete: bool

    @classmethod
    def from_measurement_log(
        cls,
        log: MeasurementLogInfo,
        *,
        cutoff_step: int,
        prefix_measurement_log_sha256: str,
        covered_records_sha256: str,
        station_boundaries: StationBoundarySchedule,
        station_complete_marker: bool,
    ) -> MeasurementPrefix:
        """Bind an exact leading slice to an independently declared station boundary."""
        source_run_id = str(log.manifest["run_id"])
        if station_boundaries.source_run_id != source_run_id:
            raise DataReuseError("Station-boundary schedule is bound to a different source run.")
        try:
            cutoff_index = log.step_ids.index(int(cutoff_step))
        except ValueError as exc:
            raise ContractError(
                f"Cutoff step {cutoff_step} is absent from MeasurementLog."
            ) from exc
        covered_steps = log.step_ids[: cutoff_index + 1]
        covered_stations = log.station_ids[: cutoff_index + 1]
        cutoff_station = covered_stations[-1]
        if not station_complete_marker:
            raise DataReuseError("A station-complete marker is required at every MLE cutoff.")
        if not station_boundaries.asserts_complete(
            station_id=cutoff_station, step_id=covered_steps[-1]
        ):
            raise DataReuseError("Cutoff does not match the declared station terminal step.")
        return cls(
            source_run_id=source_run_id,
            prefix_measurement_log_sha256=_require_sha256(
                prefix_measurement_log_sha256, label="prefix_measurement_log_sha256"
            ),
            covered_records_sha256=_require_sha256(
                covered_records_sha256, label="covered_records_sha256"
            ),
            station_boundary_schedule_sha256=station_boundaries.schedule_sha256,
            covered_station_boundaries_sha256=station_boundaries.covered_sha256(
                cutoff_step=covered_steps[-1]
            ),
            covered_step_ids=covered_steps,
            covered_station_ids=covered_stations,
            data_cutoff_step=covered_steps[-1],
            data_cutoff_station=cutoff_station,
            cutoff_station_complete=True,
        )

    def assert_exact_coverage(self, steps: tuple[int, ...]) -> None:
        """Reject a missing, extra, reordered, or non-prefix observation set."""
        if tuple(int(step) for step in steps) != self.covered_step_ids:
            raise DataReuseError("Batch MLE coverage is not the exact declared prefix.")

    @property
    def corroboration_min_step(self) -> int:
        """First step ID that may provide independent verification evidence."""
        return self.data_cutoff_step + 1
