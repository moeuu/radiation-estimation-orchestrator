"""Deterministic local-estimator artifacts and resumable checkpoints."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orchestrator.contracts import (
    MLEResultInfo,
    PFCheckpointInfo,
    PFResultInfo,
    validate_mle_result,
    validate_pf_checkpoint_v1,
    validate_pf_result,
)
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import (
    canonical_json_bytes,
    canonical_json_line,
    sha256_bytes,
    sha256_file,
    write_json_atomic,
    write_npz_atomic,
)
from orchestrator.hybrid.prefix_log import measurement_records_sha256

from .context import EstimatorContext, load_estimator_context
from .mle import SpectralMLE, SpectralMLEConfig, SpectralMLEResult
from .pf import ParticleFilter, ParticleFilterConfig, ParticleState


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def repository_commit() -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository_root()), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError("Could not resolve the local estimator revision.") from exc
    value = completed.stdout.strip()
    if len(value) != 40:
        raise ContractError("Local estimator revision is not a full Git commit.")
    return value


def _resolved_hash(payload: dict[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def pf_resolved_config_sha256(path: str | Path) -> str:
    return _resolved_hash(ParticleFilterConfig.from_path(path).to_dict())


def mle_resolved_config_sha256(path: str | Path) -> str:
    return _resolved_hash(SpectralMLEConfig.from_path(path).to_dict())


def save_particle_state(path: str | Path, state: ParticleState) -> Path:
    return write_npz_atomic(
        path,
        {
            "schema_version": np.asarray(1, dtype=np.int64),
            "chart_ids": np.asarray(state.chart_ids, dtype=np.int64),
            "strengths_cps_1m": np.asarray(state.strengths_cps_1m, dtype=np.float64),
            "log_weights": np.asarray(state.log_weights, dtype=np.float64),
            "processed_record_count": np.asarray(
                state.processed_record_count, dtype=np.int64
            ),
            "predicted_spectra": np.asarray(state.predicted_spectra, dtype=np.float64),
            "prediction_cutoffs": np.asarray(state.prediction_cutoffs, dtype=np.int64),
            "rng_state_json": np.asarray(
                canonical_json_bytes(state.rng_state).decode("utf-8")
            ),
            "applied_directive_ids": np.asarray(state.applied_directive_ids, dtype=np.str_),
            "trace_json": np.asarray(
                canonical_json_bytes(list(state.trace)).decode("utf-8")
            ),
        },
    )


def load_particle_state(path: str | Path) -> ParticleState:
    source = Path(path).resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "schema_version",
                "chart_ids",
                "strengths_cps_1m",
                "log_weights",
                "processed_record_count",
                "predicted_spectra",
                "prediction_cutoffs",
                "rng_state_json",
                "applied_directive_ids",
                "trace_json",
            }
            if set(archive.files) != required:
                raise ContractError("PF state artifact has an unexpected array set.")
            if int(np.asarray(archive["schema_version"]).item()) != 1:
                raise ContractError("PF state artifact has an unsupported schema.")
            return ParticleState(
                chart_ids=np.asarray(archive["chart_ids"], dtype=np.int64),
                strengths_cps_1m=np.asarray(
                    archive["strengths_cps_1m"], dtype=np.float64
                ),
                log_weights=np.asarray(archive["log_weights"], dtype=np.float64),
                processed_record_count=int(
                    np.asarray(archive["processed_record_count"]).item()
                ),
                predicted_spectra=np.asarray(
                    archive["predicted_spectra"], dtype=np.float64
                ),
                prediction_cutoffs=np.asarray(
                    archive["prediction_cutoffs"], dtype=np.int64
                ),
                rng_state=json.loads(str(np.asarray(archive["rng_state_json"]).item())),
                applied_directive_ids=tuple(
                    str(value) for value in archive["applied_directive_ids"].tolist()
                ),
                trace=tuple(
                    json.loads(str(np.asarray(archive["trace_json"]).item()))
                ),
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"Could not load PF state artifact: {exc}") from exc


def load_checkpoint_state(checkpoint: PFCheckpointInfo) -> ParticleState:
    artifact = checkpoint.path.parent / str(checkpoint.payload["state_artifact"])
    if sha256_file(artifact) != checkpoint.payload["state_artifact_sha256"]:
        raise ContractError("PF checkpoint state hash changed after validation.")
    return load_particle_state(artifact)


@dataclass(frozen=True, slots=True)
class PFRunArtifacts:
    result: PFResultInfo
    checkpoint: PFCheckpointInfo
    context: EstimatorContext


def run_pf_checkpoint(
    measurement_log: str | Path,
    *,
    config_path: str | Path,
    output_directory: str | Path,
    random_seed: int,
    checkpoint_in: PFCheckpointInfo | None = None,
) -> PFRunArtifacts:
    """Run or resume the local strict PF and publish one authenticated checkpoint."""
    config_file = Path(config_path).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = ParticleFilterConfig.from_path(config_file)
    context = load_estimator_context(
        measurement_log,
        patch_edge_m=config.patch_edge_m,
        use_gpu=config.use_gpu,
    )
    initial_state = None if checkpoint_in is None else load_checkpoint_state(checkpoint_in)
    if checkpoint_in is not None:
        if checkpoint_in.payload["source_run_id"] != context.measurement_log.manifest["run_id"]:
            raise ContractError("PF checkpoint source run differs from the resumed log.")
        if int(checkpoint_in.payload["random_seed"]) != int(random_seed):
            raise ContractError("PF checkpoint random seed differs from the resume request.")
        if checkpoint_in.payload["resolved_config_sha256"] != _resolved_hash(config.to_dict()):
            raise ContractError("PF checkpoint config differs from the resume request.")
        assert initial_state is not None
        processed = initial_state.processed_record_count
        if processed < 1:
            raise ContractError("A resumed PF checkpoint must cover a nonempty prefix.")
        expected_steps = list(context.measurement_log.step_ids[:processed])
        if checkpoint_in.payload["covered_step_ids"] != expected_steps:
            raise DataReuseError("PF checkpoint does not cover the current log prefix.")
        if int(checkpoint_in.payload["data_cutoff_step"]) != expected_steps[-1]:
            raise DataReuseError("PF checkpoint cutoff is inconsistent with its state.")
        if checkpoint_in.payload["covered_records_sha256"] != measurement_records_sha256(
            context.measurement_log,
            record_count=processed,
        ):
            raise DataReuseError("PF checkpoint observation prefix changed before resume.")
    estimator = ParticleFilter(
        context,
        config,
        random_seed=random_seed,
        state=initial_state,
    )
    start = estimator.state.processed_record_count
    stations = np.asarray(context.measurement_log.arrays["station_id"], dtype=np.int64)
    cursor = start
    while cursor < context.measurement_log.record_count:
        station = int(stations[cursor])
        end = cursor + 1
        while end < context.measurement_log.record_count and int(stations[end]) == station:
            end += 1
        estimator.advance(end)
        cursor = end
    if len(estimator.state.trace) != context.measurement_log.record_count:
        raise ContractError("PF checkpoint trace does not cover the exact causal prefix.")
    resolved_hash = _resolved_hash(config.to_dict())
    commit = repository_commit()
    posterior = {
        "schema_version": 1,
        "estimator_family": "particle_filter",
        "estimator_variant": "pf_strict",
        "final_estimate_source": "pf_posterior",
        "uses_all_history_batch_fit": False,
        "uses_surface_map": False,
        "uses_batch_model_order": False,
        "isotopes": estimator.posterior_summary(),
        "provenance": {
            "estimator_repository": "radiation-estimation-orchestrator",
            "estimator_commit": commit,
            "measurement_log_schema_version": 2,
            "measurement_log_sha256": context.measurement_log.measurement_log_sha256,
            "resolved_config_sha256": resolved_hash,
            "config_sha256": sha256_file(config_file),
            "random_seed": int(random_seed),
            "planner_belief_sources": ["joint_pf_particles", "pf_posterior"],
            "batch_feedback_applied": False,
            "command": ["internal", "pf-checkpoint"],
        },
    }
    write_json_atomic(output / "pf_posterior.json", posterior)
    write_json_atomic(
        output / "pf_diagnostics.json",
        {
            "schema_version": 2,
            "measurement_log_schema_version": 2,
            "record_count": context.measurement_log.record_count,
            "processed_record_count": estimator.state.processed_record_count,
            "effective_sample_size": float(
                1.0 / np.sum(np.square(estimator.weights))
            ),
            "state_identity_sha256": estimator.state_identity(),
            "batch_methods_invoked": [],
            "truth_read": False,
        },
    )
    (output / "pf_trace.jsonl").write_text(
        "".join(canonical_json_line(row) + "\n" for row in estimator.state.trace),
        encoding="utf-8",
    )
    write_npz_atomic(
        output / "pf_spectral_predictions.npz",
        {
            "step_id": np.asarray(context.measurement_log.step_ids, dtype=np.int64),
            "prediction_data_cutoff_step": np.asarray(
                estimator.state.prediction_cutoffs, dtype=np.int64
            ),
            "predicted_spectra": np.asarray(
                estimator.state.predicted_spectra, dtype=np.float64
            ),
        },
    )
    state_path = save_particle_state(output / "pf_state.npz", estimator.state)
    rng_hash = sha256_bytes(canonical_json_bytes(estimator.state.rng_state))
    cutoff_step = int(context.measurement_log.step_ids[-1])
    checkpoint_identity = {
        "run": context.measurement_log.manifest["run_id"],
        "cutoff": cutoff_step,
        "state": sha256_file(state_path),
        "config": resolved_hash,
    }
    checkpoint_payload = {
        "schema_version": 1,
        "checkpoint_family": "pure_pf_causal_state",
        "checkpoint_id": (
            "pf-checkpoint-"
            f"{sha256_bytes(canonical_json_bytes(checkpoint_identity))[:20]}"
        ),
        "source_run_id": context.measurement_log.manifest["run_id"],
        "measurement_log_schema_version": 2,
        "data_cutoff_step": cutoff_step,
        "data_cutoff_station": int(context.measurement_log.station_ids[-1]),
        "covered_step_ids": list(context.measurement_log.step_ids),
        "covered_records_sha256": measurement_records_sha256(context.measurement_log),
        "prefix_measurement_log_sha256": context.measurement_log.measurement_log_sha256,
        "pf_repository_commit": commit,
        "resolved_config_sha256": resolved_hash,
        "random_seed": int(random_seed),
        "state_artifact": state_path.name,
        "state_artifact_sha256": sha256_file(state_path),
        "rng_state_sha256": rng_hash,
        "safety": {
            "prefix_causal": True,
            "truth_read": False,
            "batch_feedback_applied": False,
        },
    }
    checkpoint_path = write_json_atomic(output / "pf_checkpoint.json", checkpoint_payload)
    result = validate_pf_result(
        output,
        expected_variant="pf_strict",
        expected_isotopes=context.isotopes,
        expected_log_sha256=context.measurement_log.measurement_log_sha256,
        expected_commit=commit,
        expected_config_sha256=sha256_file(config_file),
        expected_resolved_config_sha256=resolved_hash,
        expected_record_count=context.measurement_log.record_count,
        expected_step_ids=context.measurement_log.step_ids,
    )
    checkpoint = validate_pf_checkpoint_v1(
        checkpoint_path,
        expected_source_run_id=str(context.measurement_log.manifest["run_id"]),
        expected_prefix_measurement_log_sha256=(
            context.measurement_log.measurement_log_sha256
        ),
    )
    return PFRunArtifacts(result=result, checkpoint=checkpoint, context=context)


def _write_mle_bundle(
    *,
    context: EstimatorContext,
    config: SpectralMLEConfig,
    config_path: Path,
    result: SpectralMLEResult,
    output: Path,
    fit_kind: str,
    warm_start_result_sha256: str | None,
) -> MLEResultInfo:
    output.mkdir(parents=True, exist_ok=False)
    commit = repository_commit()
    resolved_hash = _resolved_hash(config.to_dict())
    provenance = {
        "estimator_family": "surface_mle",
        "estimator_variant": "spectral",
        "estimator_repository": "radiation-estimation-orchestrator",
        "estimator_commit": commit,
        "measurement_log_schema_version": 2,
        "measurement_log_sha256": context.measurement_log.measurement_log_sha256,
        "config_sha256": sha256_file(config_path),
        "resolved_estimator_config_sha256": resolved_hash,
        "uses_pf_state": False,
        "uses_pf_candidates": False,
        "candidate_domain": "complete_surface_dictionary",
        "command": ["internal", "spectral-mle"],
    }
    diagnostics = {
        "schema_version": 1,
        "isotope_names": list(context.isotopes),
        "patch_count": context.surface_geometry.chart_count,
        "predicted_spectra_present": True,
        "predicted_isotope_counts_present": False,
        "objective_value": result.objective_value,
        "poisson_deviance": result.poisson_deviance,
        "iterations": result.iterations,
        "converged": result.converged,
        "diagnostics": {
            "mode": "spectral",
            "hotspot_clusters": list(result.hotspot_clusters),
            "optimizer_message": result.optimizer_message,
            "warm_start_used": result.warm_start_used,
            "response_semantics": "exact_runtime_pre_dead_time_columns_with_dead_time_mean",
            "likelihood": "spectral_poisson",
            "causal_lineage": {
                "fit_kind": fit_kind,
                "warm_start": warm_start_result_sha256,
                "covered_step_ids": list(context.measurement_log.step_ids),
                "record_count": context.measurement_log.record_count,
            },
            "provenance": provenance,
        },
        "config": config.to_dict(),
        "provenance": provenance,
    }
    diagnostics_path = write_json_atomic(output / "mle_diagnostics.json", diagnostics)
    write_json_atomic(
        output / "hotspot_clusters.json",
        {"schema_version": 1, "hotspot_clusters": list(result.hotspot_clusters)},
    )
    geometry = context.surface_geometry
    write_npz_atomic(
        output / "mle_estimate.npz",
        {
            "schema_version": np.asarray(1, dtype=np.int64),
            "diagnostics_sha256": np.asarray(sha256_file(diagnostics_path)),
            "isotope_names": np.asarray(context.isotopes, dtype=np.str_),
            "patch_ids": np.arange(geometry.chart_count, dtype=np.int64),
            "patch_centroids_xyz": np.asarray(geometry.centers_xyz, dtype=np.float64),
            "patch_surface_kinds": np.asarray(geometry.kinds, dtype=np.str_),
            "patch_strength_by_isotope": np.asarray(
                result.patch_strength_by_isotope, dtype=np.float64
            ),
            "density_by_isotope": np.asarray(result.density_by_isotope, dtype=np.float64),
            "predicted_spectra": np.asarray(result.predicted_spectra, dtype=np.float64),
            "objective_value": np.asarray(result.objective_value, dtype=np.float64),
            "poisson_deviance": np.asarray(result.poisson_deviance, dtype=np.float64),
            "iterations": np.asarray(result.iterations, dtype=np.int64),
            "converged": np.asarray(result.converged, dtype=np.bool_),
            "patch_count": np.asarray(geometry.chart_count, dtype=np.int64),
        },
    )
    return validate_mle_result(
        output,
        expected_mode="spectral",
        expected_isotopes=context.isotopes,
        expected_log_sha256=context.measurement_log.measurement_log_sha256,
        expected_commit=commit,
        expected_config_sha256=sha256_file(config_path),
        expected_resolved_config_sha256=resolved_hash,
    )


def run_spectral_mle(
    measurement_log: str | Path,
    *,
    config_path: str | Path,
    output_directory: str | Path,
    warm_start_result: MLEResultInfo | None = None,
    fit_kind: str | None = None,
) -> MLEResultInfo:
    config_file = Path(config_path).resolve()
    config = SpectralMLEConfig.from_path(config_file)
    context = load_estimator_context(
        measurement_log,
        patch_edge_m=config.patch_edge_m,
        use_gpu=config.use_gpu,
    )
    warm_array = None
    warm_hash = None
    if warm_start_result is not None:
        warm_isotopes = tuple(
            str(value) for value in warm_start_result.arrays["isotope_names"].tolist()
        )
        if warm_isotopes != context.isotopes:
            raise ContractError("Warm spectral MLE isotope order is incompatible.")
        if warm_start_result.arrays["patch_strength_by_isotope"].shape != (
            len(context.isotopes),
            context.surface_geometry.chart_count,
        ):
            raise ContractError("Warm spectral MLE surface dictionary is incompatible.")
        warm_array = np.asarray(
            warm_start_result.arrays["patch_strength_by_isotope"], dtype=np.float64
        )
        warm_hash = warm_start_result.result_sha256
    estimator = SpectralMLE(context, config)
    result = estimator.fit(warm_start=warm_array)
    lineage = fit_kind or (
        "warm_start_prefix" if warm_start_result is not None else "cold_start_all_history"
    )
    return _write_mle_bundle(
        context=context,
        config=config,
        config_path=config_file,
        result=result,
        output=Path(output_directory).resolve(),
        fit_kind=lineage,
        warm_start_result_sha256=warm_hash,
    )


__all__ = [
    "PFRunArtifacts",
    "load_checkpoint_state",
    "load_particle_state",
    "mle_resolved_config_sha256",
    "pf_resolved_config_sha256",
    "repository_commit",
    "run_pf_checkpoint",
    "run_spectral_mle",
    "save_particle_state",
]
