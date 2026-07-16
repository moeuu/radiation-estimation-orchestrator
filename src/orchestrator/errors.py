"""Typed failures surfaced by contracts, adapters, and benchmark orchestration."""


class OrchestratorError(RuntimeError):
    """Base class for expected orchestration failures."""


class ContractError(OrchestratorError, ValueError):
    """A versioned input or result contract is invalid."""


class RevisionError(OrchestratorError):
    """A local estimator checkout does not match its pinned revision."""


class AdapterExecutionError(OrchestratorError):
    """An estimator subprocess failed or produced an invalid result."""


class TruthIsolationError(OrchestratorError):
    """Evaluation truth crossed the estimator execution boundary."""


class DataReuseError(OrchestratorError):
    """A hybrid action would reuse an observation across a declared cutoff."""
