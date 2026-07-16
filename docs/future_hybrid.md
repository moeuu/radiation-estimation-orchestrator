# Future PF+MLE milestone (not implemented)

The eventual PF+MLE controller belongs only in this repository. Neither pure estimator
repository should acquire cross-estimator feedback.

A later, separately versioned milestone may add predictive-deviance monitoring,
scheduled MLE snapshots, verification queues, quarantined candidates, target-preserving
rejuvenation proposals, independently corroborated pruning, conflict-hypothesis
measurement design, and a final full-history MLE.

Every MLE snapshot must declare the last covered observation and station. A feedback
consumer must bind every derived action to that snapshot, apply it at most once, and
resume sequential PF processing strictly after `data_cutoff_step`. It must never turn a
snapshot derived from observations `1:t` into a second likelihood update for those same
observations. Proposal-only moves would additionally require a target-preserving
Metropolis-Hastings or importance correction.

No scheduler, queue, feedback directive, rejuvenation, pruning, or hybrid planner is
present in v1. The current `mle_snapshot_schema.json` is a reserved interoperability
contract, not an active runtime.
