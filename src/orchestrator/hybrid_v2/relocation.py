"""Build spectral-snapshot fixed-cardinality PF relocation directives."""

from __future__ import annotations

from pathlib import Path

from orchestrator.contracts import (
    PFDirectiveInfo,
    SpectralMLESnapshotInfo,
    validate_pf_directive,
)
from orchestrator.errors import ContractError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes, write_json_idempotent


def build_spectral_relocation_directive(
    *,
    output_path: str | Path,
    snapshot: SpectralMLESnapshotInfo,
    pf_resolved_config_sha256: str,
    covered_station_boundaries_sha256: str,
    position_sigma_xyz_m: tuple[float, float, float],
    defensive_weight: float,
) -> PFDirectiveInfo:
    """Translate complete-surface candidates into corrected MH proposal metadata."""
    if any(value <= 0 for value in position_sigma_xyz_m):
        raise ContractError("Relocation proposal sigma values must be positive.")
    if not 0 < defensive_weight <= 1:
        raise ContractError("Relocation defensive weight must lie in (0, 1].")
    raw_candidates = snapshot.payload["candidates"]
    assert isinstance(raw_candidates, list)
    if not raw_candidates:
        raise ContractError("Spectral relocation requires at least one MLE candidate.")
    proposals = []
    for raw in raw_candidates:
        assert isinstance(raw, dict)
        identity = {
            "snapshot": snapshot.snapshot_sha256,
            "candidate": raw["candidate_id"],
            "kernel": "defensive_truncated_gaussian_position",
        }
        proposals.append(
            {
                "proposal_id": (
                    f"proposal-{sha256_bytes(canonical_json_bytes(identity))[:20]}"
                ),
                "snapshot_candidate_id": raw["candidate_id"],
                "isotope": raw["isotope"],
                "candidate_mean_xyz": raw["centroid_xyz"],
                "snapshot_strength_cps_1m_metadata": raw[
                    "integrated_strength_cps_1m"
                ],
                "proposal_kernel": {
                    "family": "defensive_truncated_gaussian_position",
                    "position_sigma_xyz_m": list(position_sigma_xyz_m),
                    "defensive_weight": defensive_weight,
                    "candidate_weight": max(
                        float(raw["integrated_strength_cps_1m"]), 1e-12
                    ),
                },
            }
        )
    prefix = snapshot.payload["prefix"]
    assert isinstance(prefix, dict)
    identity = {
        "snapshot": snapshot.snapshot_sha256,
        "pf_config": pf_resolved_config_sha256,
        "proposals": proposals,
    }
    payload = {
        "schema_version": 1,
        "directive_id": f"directive-{sha256_bytes(canonical_json_bytes(identity))[:20]}",
        "directive_kind": "proposal_only_mh",
        "snapshot_id": snapshot.payload["snapshot_id"],
        "snapshot_sha256": snapshot.snapshot_sha256,
        "source_run_id": snapshot.payload["source_run_id"],
        "prefix_measurement_log_sha256": prefix["prefix_measurement_log_sha256"],
        "covered_records_sha256": prefix["covered_records_sha256"],
        "covered_station_boundaries_sha256": covered_station_boundaries_sha256,
        "pf_resolved_config_sha256": pf_resolved_config_sha256,
        "data_cutoff_step": snapshot.cutoff_step,
        "data_cutoff_station": snapshot.cutoff_station,
        "cutoff_station_complete": True,
        "covered_step_ids": prefix["covered_step_ids"],
        "apply_after_step": snapshot.cutoff_step,
        "corroboration_min_step": snapshot.cutoff_step + 1,
        "proposals": proposals,
        "safety_policy": {
            "direct_mle_objective_reweight": False,
            "hard_prune_authorized": False,
            "future_only_corroboration": True,
            "once_only_application": True,
            "requires_target_preserving_mh": True,
        },
        "provenance": {
            "milestone": "pf_mle_hybrid_v2",
            "spectral_snapshot_sha256": snapshot.snapshot_sha256,
        },
    }
    path = write_json_idempotent(output_path, payload)
    return validate_pf_directive(path)


__all__ = ["build_spectral_relocation_directive"]
