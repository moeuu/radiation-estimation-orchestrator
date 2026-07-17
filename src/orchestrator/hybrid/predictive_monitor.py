"""Causal PF predictive diagnostics used only to schedule batch MLE work."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log
from types import MappingProxyType

from orchestrator.errors import ContractError, DataReuseError


@dataclass(frozen=True, slots=True)
class PredictiveSignal:
    """One diagnostic computed from the current observation and prior prediction."""

    step_id: int
    station_id: int
    prediction_data_cutoff_step: int
    station_complete: bool
    poisson_deviance: float
    normalized_deviance: float
    observed_counts: Mapping[str, float]
    predicted_counts: Mapping[str, float]


class PredictiveMonitor:
    """Append-only monitor that never inspects records after the current step."""

    def __init__(self, *, expected_count_floor: float = 1e-12) -> None:
        if not isfinite(expected_count_floor) or expected_count_floor <= 0:
            raise ValueError("expected_count_floor must be finite and positive.")
        self._floor = float(expected_count_floor)
        self._signals: list[PredictiveSignal] = []

    @property
    def signals(self) -> tuple[PredictiveSignal, ...]:
        """Return the causal diagnostic history."""
        return tuple(self._signals)

    def record(
        self,
        *,
        step_id: int,
        station_id: int,
        prediction_data_cutoff_step: int,
        station_complete_marker: bool,
        observed_counts: Mapping[str, float],
        predicted_counts: Mapping[str, float],
    ) -> PredictiveSignal:
        """Compute Poisson deviance without reading an MLE objective or future row."""
        if step_id < 0 or station_id < 0:
            raise ContractError("Predictive signal step and station IDs must be nonnegative.")
        if prediction_data_cutoff_step < -1 or prediction_data_cutoff_step >= step_id:
            raise DataReuseError(
                "PF prediction must be frozen before the observation used for deviance."
            )
        if self._signals and step_id <= self._signals[-1].step_id:
            raise DataReuseError("Predictive signals must be appended in strict causal order.")
        if set(observed_counts) != set(predicted_counts) or not observed_counts:
            raise ContractError("Observed and predicted isotope keys must match and be nonempty.")
        observed: dict[str, float] = {}
        predicted: dict[str, float] = {}
        deviance = 0.0
        for isotope in sorted(observed_counts):
            value = float(observed_counts[isotope])
            expectation = float(predicted_counts[isotope])
            if not isfinite(value) or value < 0:
                raise ContractError(f"Observed count for {isotope} must be finite and nonnegative.")
            if not isfinite(expectation) or expectation < 0:
                raise ContractError(
                    f"Predicted count for {isotope} must be finite and nonnegative."
                )
            safe_expectation = max(expectation, self._floor)
            term = safe_expectation - value
            if value > 0:
                term += value * log(value / safe_expectation)
            deviance += 2.0 * term
            observed[isotope] = value
            predicted[isotope] = expectation
        signal = PredictiveSignal(
            step_id=int(step_id),
            station_id=int(station_id),
            prediction_data_cutoff_step=int(prediction_data_cutoff_step),
            station_complete=bool(station_complete_marker),
            poisson_deviance=float(deviance),
            normalized_deviance=float(deviance / len(observed)),
            observed_counts=MappingProxyType(observed),
            predicted_counts=MappingProxyType(predicted),
        )
        self._signals.append(signal)
        return signal
