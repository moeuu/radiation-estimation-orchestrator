# Forward-model conformance

The two estimators must independently implement a CLI accepting the neutral axes file
and emitting an NPZ with exactly:

- `case_ids`: one-dimensional Unicode strings;
- `unit_response`: nonnegative finite float64 values with the same leading dimension.

Axis order is isotope, detector pose, Fe orientation, Pb orientation, source point,
obstacle. Case IDs are:

```text
{isotope}|pose={pose_id}|fe={fe:02d}|pb={pb:02d}|source={source_id}|obstacle={obstacle_id}
```

The v1 fixture has 4,608 cases. Scalar response is expected total isotope counts from
unit `detector_cps_1m` strength for the case live time. The comparison tolerance is
`rtol=1e-9`, `atol=1e-12`.

Production provider JSON is an execution lock, not only a command shortcut. Each file
must name its provider, declare `revision_type: "commit"`, pin an exact 40-character
lowercase Git commit in `revision`, and set `require_clean: true`. Before creating an
artifact or starting the subprocess, the orchestrator requires that the provider
checkout's `HEAD` is that exact commit object and that `git status` is completely
clean. Unlike benchmark result/cache handling, conformance has no dirty-path
allowlist.

Commands remain arrays of arguments and are expanded token by token without a shell.
For both providers, `conformance_report.json` records the observed provider revision,
the fully expanded argument array, the provider JSON SHA-256, stdout and stderr
SHA-256 values, and the response-artifact SHA-256. This binds every numerical
comparison to the code, invocation, configuration, process output, and NPZ bytes that
produced it. In-process analytic test doubles use explicit neutral (`null`/empty)
execution provenance while retaining their artifact hash.

This checks units, distance scaling, detector geometry, all 64 shield pairs, obstacle
attenuation, live time, and isotope response without sharing physics code.

The scalar comparison is paired with an exact line-model identity check in both
estimator repositories. Each side must derive the same production
`line_mu_by_isotope` table and reproduce the shield and spectrum hashes documented in
the MeasurementLog contract. This prevents a scalar-total agreement from hiding a
different spectral line basis.

The next conformance-contract revision should add a per-energy-bin response vector.
That extension is deliberately not smuggled into v1: v1 remains the stable 4,608-case
scalar artifact plus exact line-table/hash validation.
