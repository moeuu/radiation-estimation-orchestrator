# Work beyond hybrid v1

The first causal PF+MLE hybrid milestone is described in
[`hybrid_v1.md`](hybrid_v1.md). It implements exact-prefix MLE snapshots, once-only
directives, corrected fixed-cardinality PF position proposals, future-only candidate
verification, an observation-use ledger, and a cold full-history MLE final report.

The following capabilities remain future work and must not be inferred from the v1
name or result:

- reversible-jump birth/death or another target-preserving trans-dimensional move;
- evidence-backed pruning with an explicit statistical correction;
- live closed-loop detector XYZ, height, and shield-program actuation;
- a conflict-hypothesis planner that affects measurement acquisition;
- a resumable PF checkpoint contract instead of deterministic prefix replay;
- direct online spectral-MLE snapshots, if scientifically justified;
- a mesh-surface extension beyond the current standalone MLE domain.

In particular, an MLE objective must never be added to PF log weights for observations
the PF already processed. Any future cardinality-changing feedback needs a separately
reviewed proposal density, reverse move, Jacobian where applicable, acceptance rule,
and an extended observation-use ledger contract. Hard pruning cannot be introduced as
a configuration shortcut.

Live planning also requires a new milestone: the controller must choose an action
before its observation exists, bind that decision to the exact PF/MLE information
cutoff, execute through the measurement runtime, and record the proposed and realized
action. Offline replay scores or recommendations are not live planner control.
