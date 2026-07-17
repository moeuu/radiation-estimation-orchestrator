"""Deterministic station-complete and PF-mismatch MLE scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import fsum

from orchestrator.errors import DataReuseError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes

from .config import HybridConfig
from .predictive_monitor import PredictiveSignal


class TriggerReason(StrEnum):
    """Causal reasons that may request a prefix MLE fit."""

    STATION_INTERVAL = "station_interval"
    PREDICTIVE_MISMATCH = "predictive_mismatch"


@dataclass(frozen=True, slots=True)
class HybridTrigger:
    """One once-only request to fit through a station-complete cutoff."""

    trigger_id: str
    data_cutoff_step: int
    data_cutoff_station: int
    station_complete: bool
    reasons: tuple[TriggerReason, ...]
    normalized_predictive_deviance: float


class HybridScheduler:
    """Stateful causal scheduler; only explicit station-end signals are eligible."""

    def __init__(self, config: HybridConfig) -> None:
        self._config = config
        self._last_seen_step = -1
        self._completed_stations = 0
        self._last_completed_station: int | None = None
        self._last_trigger_completed_index: int | None = None
        self._mismatch_streak = 0
        self._triggered_cutoffs: set[int] = set()
        self._open_station_id: int | None = None
        self._station_deviances: list[float] = []
        self._station_comparison_count = 0

    def consider(self, signal: PredictiveSignal) -> HybridTrigger | None:
        """Return a trigger using only diagnostics available through ``signal.step_id``."""
        if signal.step_id <= self._last_seen_step:
            raise DataReuseError("Scheduler inputs must be strictly increasing in step order.")
        self._last_seen_step = signal.step_id
        if self._open_station_id is None:
            self._open_station_id = signal.station_id
        elif signal.station_id != self._open_station_id:
            raise DataReuseError(
                "A station must reach its explicit terminal signal before another begins."
            )
        self._station_deviances.append(signal.poisson_deviance)
        self._station_comparison_count += len(signal.observed_counts)
        if not signal.station_complete:
            return None
        if self._station_comparison_count <= 0:
            raise DataReuseError("A completed station must contain predictive comparisons.")
        station_normalized_deviance = fsum(sorted(self._station_deviances)) / (
            self._station_comparison_count
        )
        self._open_station_id = None
        self._station_deviances.clear()
        self._station_comparison_count = 0
        if (
            self._last_completed_station is not None
            and signal.station_id <= self._last_completed_station
        ):
            raise DataReuseError("Each station may have only one terminal scheduling signal.")
        self._last_completed_station = signal.station_id
        self._completed_stations += 1

        threshold = self._config.predictive_deviance_threshold
        mismatch = threshold is not None and station_normalized_deviance >= threshold
        self._mismatch_streak = self._mismatch_streak + 1 if mismatch else 0
        if self._completed_stations < self._config.minimum_completed_stations:
            return None
        if self._last_trigger_completed_index is not None:
            elapsed = self._completed_stations - self._last_trigger_completed_index
            if elapsed <= self._config.scheduler_cooldown_stations:
                return None

        reasons: list[TriggerReason] = []
        if self._completed_stations % self._config.station_interval == 0:
            reasons.append(TriggerReason.STATION_INTERVAL)
        if mismatch and self._mismatch_streak >= self._config.predictive_mismatch_streak:
            reasons.append(TriggerReason.PREDICTIVE_MISMATCH)
        if not reasons:
            return None
        if signal.step_id in self._triggered_cutoffs:
            raise DataReuseError("A cutoff may schedule at most one MLE snapshot.")
        self._triggered_cutoffs.add(signal.step_id)
        self._last_trigger_completed_index = self._completed_stations
        body = {
            "schema_version": 1,
            "data_cutoff_step": signal.step_id,
            "data_cutoff_station": signal.station_id,
            "reasons": [reason.value for reason in reasons],
        }
        identifier = f"trigger-{sha256_bytes(canonical_json_bytes(body))[:20]}"
        return HybridTrigger(
            trigger_id=identifier,
            data_cutoff_step=signal.step_id,
            data_cutoff_station=signal.station_id,
            station_complete=True,
            reasons=tuple(reasons),
            normalized_predictive_deviance=station_normalized_deviance,
        )
