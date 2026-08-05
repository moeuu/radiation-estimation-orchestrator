# Work beyond hybrid v2

Hybrid v2 already owns local checkpoint PF, full-surface spectral MLE, future block
scoring, paired exact RJ, runtime-attested 3-D planning, and resumable live execution.
Remaining work is primarily scale, calibration, and scientific validation:

- replace materialized spectral response columns with a matrix-free or disk-backed
  operator for fine grids and long missions;
- add PF resample-move kernels that preserve more spatial/cardinality diversity without
  depending on MLE proposals;
- extend runtime-published surface charts to meshes or volumes before adding matching
  local MLE/RJ parameterizations;
- calibrate verification and planner weights on station-block holdouts across many
  environments rather than one seed;
- add bootstrap/stability uncertainty for MLE maps and clusters;
- run fixed-budget pure-PF, pure-MLE, and hybrid ablations on new random Geant4 scenes
  and then on the real system;
- quantify candidate-density convergence and the cost/benefit of vertical motion.

MLE-directed hard pruning remains forbidden. Source removal must remain a reversible
PF transition with the same current target, reverse proposal, Jacobian, once-only
ledger, and receipt checks as source birth. An MLE objective must never be added to PF
weights for observations already processed by the PF.
