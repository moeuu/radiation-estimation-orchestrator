"""Pre-update full-spectrum diagnostics and causal hybrid-v2 scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes


@dataclass(frozen=True, slots=True)
class SpectralPredictiveSignal:
    """One raw-spectrum diagnostic frozen before its observation was assimilated."""

    step_id: int
    station_id: int
    prediction_data_cutoff_step: int
    station_complete: bool
    energy_bin_count: int
    poisson_deviance: float
    normalized_deviance: float
    observed_spectrum_sha256: str
    predicted_spectrum_sha256: str


def _spectrum_digest(values: np.ndarray) -> str:
    descriptor = {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "bytes_sha256": sha256_bytes(values.tobytes(order="C")),
    }
    return sha256_bytes(canonical_json_bytes(descriptor))


class SpectralPredictiveMonitor:
    """Append pre-update Poisson deviances without retaining spectrum values."""

    def __init__(self, *, expectation_floor: float = 1e-12) -> None:
        if not isfinite(expectation_floor) or expectation_floor <= 0:
            raise ValueError("expectation_floor must be finite and positive.")
        self._floor = float(expectation_floor)
        self._signals: list[SpectralPredictiveSignal] = []

    @property
    def signals(self) -> tuple[SpectralPredictiveSignal, ...]:
        return tuple(self._signals)

    def record(
        self,
        *,
        step_id: int,
        station_id: int,
        prediction_data_cutoff_step: int,
        station_complete: bool,
        observed_spectrum: ArrayLike,
        predicted_spectrum: ArrayLike,
    ) -> SpectralPredictiveSignal:
        """Record a diagnostic only when the prediction predates this observation."""
        if step_id < 0 or station_id < 0:
            raise ContractError("Spectral predictive step and station IDs must be nonnegative.")
        if prediction_data_cutoff_step < -1 or prediction_data_cutoff_step >= step_id:
            raise DataReuseError("Spectral prediction must be frozen before its observation.")
        if self._signals and step_id <= self._signals[-1].step_id:
            raise DataReuseError("Spectral predictive signals must be strictly causal.")
        observed = np.asarray(observed_spectrum)
        predicted = np.asarray(predicted_spectrum, dtype=np.float64)
        if observed.ndim != 1 or predicted.shape != observed.shape or observed.size == 0:
            raise ContractError("Observed and predicted spectra must be equal nonempty vectors.")
        if not np.issubdtype(observed.dtype, np.integer) or np.any(observed < 0):
            raise ContractError("Observed spectrum must contain nonnegative integer counts.")
        if not np.all(np.isfinite(predicted)) or np.any(predicted < 0):
            raise ContractError("Predicted spectrum must contain finite nonnegative means.")
        observed_float = observed.astype(np.float64, copy=False)
        expected = np.maximum(predicted, self._floor)
        terms = expected - observed_float
        positive = observed_float > 0
        terms[positive] += observed_float[positive] * np.log(
            observed_float[positive] / expected[positive]
        )
        deviance = float(2.0 * np.sum(terms, dtype=np.float64))
        signal = SpectralPredictiveSignal(
            step_id=int(step_id),
            station_id=int(station_id),
            prediction_data_cutoff_step=int(prediction_data_cutoff_step),
            station_complete=bool(station_complete),
            energy_bin_count=int(observed.size),
            poisson_deviance=deviance,
            normalized_deviance=deviance / float(observed.size),
            observed_spectrum_sha256=_spectrum_digest(np.ascontiguousarray(observed)),
            predicted_spectrum_sha256=_spectrum_digest(np.ascontiguousarray(predicted)),
        )
        self._signals.append(signal)
        return signal


@dataclass(frozen=True, slots=True)
class SpectralSchedulerPolicy:
    """Station-boundary trigger policy for spectral prefix MLE."""

    station_interval: int = 2
    minimum_completed_stations: int = 1
    cooldown_stations: int = 0
    normalized_deviance_threshold: float | None = None
    mismatch_streak: int = 1

    def __post_init__(self) -> None:
        for name in (
            "station_interval",
            "minimum_completed_stations",
            "mismatch_streak",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.cooldown_stations < 0:
            raise ValueError("cooldown_stations must be nonnegative.")
        threshold = self.normalized_deviance_threshold
        if threshold is not None and (not isfinite(threshold) or threshold < 0):
            raise ValueError("normalized_deviance_threshold must be finite and nonnegative.")


class SpectralTriggerReason(StrEnum):
    STATION_INTERVAL = "station_interval"
    PREDICTIVE_MISMATCH = "spectral_predictive_mismatch"


@dataclass(frozen=True, slots=True)
class SpectralTrigger:
    trigger_id: str
    data_cutoff_step: int
    data_cutoff_station: int
    reasons: tuple[SpectralTriggerReason, ...]
    normalized_predictive_deviance: float


class SpectralHybridScheduler:
    """Causal, serializable scheduler consuming station-complete spectral diagnostics."""

    def __init__(self, policy: SpectralSchedulerPolicy) -> None:
        self.policy = policy
        self._last_step = -1
        self._open_station: int | None = None
        self._station_deviance = 0.0
        self._station_bins = 0
        self._completed_stations = 0
        self._last_trigger_completed_station: int | None = None
        self._mismatch_streak = 0
        self._cutoffs: set[int] = set()

    def consider(self, signal: SpectralPredictiveSignal) -> SpectralTrigger | None:
        if signal.step_id <= self._last_step:
            raise DataReuseError("Scheduler signals must be strictly increasing.")
        self._last_step = signal.step_id
        if self._open_station is None:
            self._open_station = signal.station_id
        elif self._open_station != signal.station_id:
            raise DataReuseError("A station must complete before the next station begins.")
        self._station_deviance += signal.poisson_deviance
        self._station_bins += signal.energy_bin_count
        if not signal.station_complete:
            return None
        normalized = self._station_deviance / float(self._station_bins)
        self._station_deviance = 0.0
        self._station_bins = 0
        self._open_station = None
        self._completed_stations += 1
        threshold = self.policy.normalized_deviance_threshold
        mismatch = threshold is not None and normalized >= threshold
        self._mismatch_streak = self._mismatch_streak + 1 if mismatch else 0
        if self._completed_stations < self.policy.minimum_completed_stations:
            return None
        if self._last_trigger_completed_station is not None:
            elapsed = self._completed_stations - self._last_trigger_completed_station
            if elapsed <= self.policy.cooldown_stations:
                return None
        reasons: list[SpectralTriggerReason] = []
        if self._completed_stations % self.policy.station_interval == 0:
            reasons.append(SpectralTriggerReason.STATION_INTERVAL)
        if mismatch and self._mismatch_streak >= self.policy.mismatch_streak:
            reasons.append(SpectralTriggerReason.PREDICTIVE_MISMATCH)
        if not reasons:
            return None
        if signal.step_id in self._cutoffs:
            raise DataReuseError("A spectral cutoff may trigger at most once.")
        self._cutoffs.add(signal.step_id)
        self._last_trigger_completed_station = self._completed_stations
        identity = {
            "schema_version": 2,
            "cutoff_step": signal.step_id,
            "cutoff_station": signal.station_id,
            "reasons": [reason.value for reason in reasons],
        }
        return SpectralTrigger(
            trigger_id=f"spectral-trigger-{sha256_bytes(canonical_json_bytes(identity))[:20]}",
            data_cutoff_step=signal.step_id,
            data_cutoff_station=signal.station_id,
            reasons=tuple(reasons),
            normalized_predictive_deviance=normalized,
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": asdict(self.policy),
            "last_step": self._last_step,
            "open_station": self._open_station,
            "station_deviance": self._station_deviance,
            "station_bins": self._station_bins,
            "completed_stations": self._completed_stations,
            "last_trigger_completed_station": self._last_trigger_completed_station,
            "mismatch_streak": self._mismatch_streak,
            "cutoffs": sorted(self._cutoffs),
        }

    @classmethod
    def from_state(cls, payload: dict[str, Any]) -> SpectralHybridScheduler:
        if payload.get("schema_version") != 1 or not isinstance(payload.get("policy"), dict):
            raise ContractError("Spectral scheduler state has an unsupported schema.")
        scheduler = cls(SpectralSchedulerPolicy(**payload["policy"]))
        scheduler._last_step = int(payload["last_step"])
        scheduler._open_station = (
            None if payload["open_station"] is None else int(payload["open_station"])
        )
        scheduler._station_deviance = float(payload["station_deviance"])
        scheduler._station_bins = int(payload["station_bins"])
        scheduler._completed_stations = int(payload["completed_stations"])
        scheduler._last_trigger_completed_station = (
            None
            if payload["last_trigger_completed_station"] is None
            else int(payload["last_trigger_completed_station"])
        )
        scheduler._mismatch_streak = int(payload["mismatch_streak"])
        scheduler._cutoffs = {int(value) for value in payload["cutoffs"]}
        return scheduler


__all__ = [
    "SpectralHybridScheduler",
    "SpectralPredictiveMonitor",
    "SpectralPredictiveSignal",
    "SpectralSchedulerPolicy",
    "SpectralTrigger",
    "SpectralTriggerReason",
]
