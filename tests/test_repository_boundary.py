"""Repository ownership tests for the estimator orchestrator."""

from __future__ import annotations

from pathlib import Path

from orchestrator.acquisition import acquire_measurement_log
from runtime.session import run_acquisition_plan


ROOT = Path(__file__).resolve().parents[1]


def test_orchestrator_contains_no_physical_runtime_tree() -> None:
    """Physics and MeasurementLog production code must stay in the shared runtime."""
    forbidden = (
        ROOT / "native",
        ROOT / "src" / "measurement",
        ROOT / "src" / "sim",
        ROOT / "src" / "spectrum",
        ROOT / "src" / "runtime",
        ROOT / "configs" / "geant4",
        ROOT / "configs" / "isaacsim",
    )
    assert all(not path.exists() for path in forbidden)


def test_acquisition_adapter_uses_shared_runtime_owner() -> None:
    """The public acquisition adapter must bind the imported shared implementation."""
    assert acquire_measurement_log.__module__ == "orchestrator.acquisition"
    assert run_acquisition_plan.__module__ == "runtime.session"
