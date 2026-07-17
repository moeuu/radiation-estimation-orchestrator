"""Subprocess-only adapters for pinned estimator repositories."""

from .base import AdapterExecution, AdapterSettings, EstimatorPin, load_estimator_pins
from .future_score_cli import DEFAULT_FUTURE_SCORE_COMMAND, FutureScoreCLIAdapter
from .hybrid_mle_cli import DEFAULT_WARM_MLE_COMMAND, WarmMLECLIAdapter
from .hybrid_pf_cli import DEFAULT_HYBRID_PF_COMMAND, HybridPFCLIAdapter
from .hybrid_planning_cli import DEFAULT_HYBRID_PLANNING_COMMAND, HybridPlanningCLIAdapter
from .mle_cli import MLECLIAdapter
from .pf_cli import PFCLIAdapter

__all__ = [
    "DEFAULT_FUTURE_SCORE_COMMAND",
    "DEFAULT_HYBRID_PF_COMMAND",
    "DEFAULT_HYBRID_PLANNING_COMMAND",
    "DEFAULT_WARM_MLE_COMMAND",
    "AdapterExecution",
    "AdapterSettings",
    "EstimatorPin",
    "FutureScoreCLIAdapter",
    "HybridPFCLIAdapter",
    "HybridPlanningCLIAdapter",
    "MLECLIAdapter",
    "PFCLIAdapter",
    "WarmMLECLIAdapter",
    "load_estimator_pins",
]
