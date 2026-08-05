"""Build proposal-only exact-RJ directives from verified spectral candidates."""

from __future__ import annotations

from pathlib import Path

from orchestrator.contracts import (
    PFCheckpointInfo,
    PFRJDirectiveInfo,
    SpectralMLESnapshotInfo,
    validate_pf_rj_directive_v1,
)
from orchestrator.errors import ContractError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes, write_json_idempotent

from .verification import CandidateState, VerificationCandidate


def build_pf_rj_directive_v1(
    *,
    output_path: str | Path,
    snapshot: SpectralMLESnapshotInfo,
    pf_checkpoint: PFCheckpointInfo,
    verification_candidates: tuple[VerificationCandidate, ...],
    data_cutoff_station: int,
    prefix_measurement_log_sha256: str,
    covered_records_sha256: str,
    dimension_matching_transform: str,
) -> PFRJDirectiveInfo:
    """Describe verified birth regions; PF remains owner of all RJ mathematics."""
    verified = tuple(
        candidate
        for candidate in verification_candidates
        if candidate.snapshot_id == snapshot.payload["snapshot_id"]
        and candidate.state is CandidateState.VERIFIED
    )
    if not verified:
        raise ContractError("An RJ directive requires at least one verified candidate.")
    if pf_checkpoint.cutoff_step <= snapshot.cutoff_step:
        raise ContractError(
            "An RJ birth may run only after future observations verified its snapshot."
        )
    if (
        pf_checkpoint.payload["prefix_measurement_log_sha256"]
        != prefix_measurement_log_sha256
    ):
        raise ContractError("RJ target prefix differs from its PF checkpoint.")
    raw_candidates = snapshot.payload["candidates"]
    assert isinstance(raw_candidates, list)
    by_id = {
        str(candidate["candidate_id"]): candidate
        for candidate in raw_candidates
        if isinstance(candidate, dict)
    }
    verified_ids = tuple(sorted(candidate.candidate_id for candidate in verified))
    if any(candidate_id not in by_id for candidate_id in verified_ids):
        raise ContractError("Verified candidate is absent from its frozen spectral snapshot.")
    strengths = [
        float(by_id[candidate_id]["integrated_strength_cps_1m"])
        for candidate_id in verified_ids
    ]
    total = sum(strengths)
    if total <= 0:
        raise ContractError("Verified RJ candidates require positive integrated strength.")
    regions = []
    for candidate_id, strength in zip(verified_ids, strengths, strict=True):
        source = by_id[candidate_id]
        regions.append(
            {
                "candidate_id": candidate_id,
                "isotope": source["isotope"],
                "centroid_xyz": source["centroid_xyz"],
                "covariance_xyz": source["covariance_xyz"],
                "integrated_strength_cps_1m": strength,
                "surface_kinds": source["surface_kinds"],
                "candidate_weight": strength / total,
            }
        )
    identity = {
        "snapshot_sha256": snapshot.snapshot_sha256,
        "pf_checkpoint_sha256": pf_checkpoint.checkpoint_sha256,
        "pf_state_before_sha256": pf_checkpoint.payload["state_artifact_sha256"],
        "candidate_ids": list(verified_ids),
        "covered_records_sha256": covered_records_sha256,
        "dimension_matching_transform": dimension_matching_transform,
    }
    prefix = snapshot.payload["prefix"]
    assert isinstance(prefix, dict)
    payload = {
        "schema_version": 1,
        "directive_family": "mle_informed_exact_rj_kernel",
        "directive_id": f"pf-rj-{sha256_bytes(canonical_json_bytes(identity))[:20]}",
        "source_run_id": snapshot.payload["source_run_id"],
        "spectral_snapshot_id": snapshot.payload["snapshot_id"],
        "spectral_snapshot_sha256": snapshot.snapshot_sha256,
        "proposal_data_cutoff_step": snapshot.cutoff_step,
        "proposal_data_cutoff_station": snapshot.cutoff_station,
        "data_cutoff_step": pf_checkpoint.cutoff_step,
        "data_cutoff_station": data_cutoff_station,
        "prefix_measurement_log_sha256": prefix_measurement_log_sha256,
        "covered_records_sha256": covered_records_sha256,
        "pf_checkpoint_sha256": pf_checkpoint.checkpoint_sha256,
        "pf_state_before_sha256": pf_checkpoint.payload["state_artifact_sha256"],
        "verified_candidate_ids": list(verified_ids),
        "birth_regions": regions,
        "kernel": {
            "family": "paired_birth_death_reversible_jump",
            "forward_birth_density": "surface_region_strength_gaussian_mixture",
            "reverse_death_selection": "pf_state_normalized_source_selection",
            "dimension_matching_transform": dimension_matching_transform,
            "jacobian_owner": "particle_filter",
            "acceptance_owner": "particle_filter",
        },
        "safety": {
            "proposal_only": True,
            "direct_pf_weight_increment": False,
            "hard_prune_requested": False,
            "same_observation_reweight": False,
            "once_only": True,
        },
    }
    path = write_json_idempotent(output_path, payload)
    return validate_pf_rj_directive_v1(path, expected_snapshot=snapshot)


__all__ = ["build_pf_rj_directive_v1"]
