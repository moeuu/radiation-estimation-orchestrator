"""Future-only spectral scoring of frozen surface-MLE candidates."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from orchestrator.contracts import (
    FutureSpectralCandidateScoreInfo,
    FutureSpectralScoreRequestInfo,
    MLEResultInfo,
    SpectralMLESnapshotInfo,
    validate_future_spectral_candidate_score_v2,
)
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import canonical_json_bytes, sha256_bytes, write_json_atomic

from .context import load_estimator_context
from .mle import SpectralMLE, SpectralMLEConfig


def _deviance(observed: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(observed, dtype=np.float64)
    mean = np.maximum(np.asarray(predicted, dtype=np.float64), np.finfo(np.float64).tiny)
    terms = mean - y
    positive = y > 0.0
    terms[positive] += y[positive] * np.log(y[positive] / mean[positive])
    return float(2.0 * np.sum(terms))


def _metadata_by_step(root: Path) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for line in (root / "observation_metadata.jsonl").read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        step = int(payload["step_id"])
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ContractError("Measurement metadata row lacks a metadata object.")
        result[step] = metadata
    return result


def _shield_program_groups(
    pose_steps: list[int],
    *,
    metadata: dict[int, dict[str, object]],
    step_to_index: dict[int, int],
    fe_indices: np.ndarray,
    pb_indices: np.ndarray,
    grouping_semantics: object,
) -> dict[str, list[int]]:
    """Group one station/height block by explicit or measured shield program."""
    explicit = [metadata[step].get("shield_program_id") for step in pose_steps]
    present = [isinstance(value, str) and bool(value) for value in explicit]
    if any(present) and not all(present):
        raise ContractError(
            "A station/height block may not mix present and absent shield-program IDs."
        )
    if all(present):
        result: dict[str, list[int]] = defaultdict(list)
        for step, program in zip(pose_steps, explicit, strict=True):
            result[str(program)].append(step)
        return dict(result)
    if grouping_semantics == "metadata_shield_program_id":
        raise ContractError("Future scoring requires metadata.shield_program_id.")
    pair_sequence = [
        [
            int(fe_indices[step_to_index[step]]),
            int(pb_indices[step_to_index[step]]),
        ]
        for step in pose_steps
    ]
    program = (
        "fe-pb-sequence-"
        f"{sha256_bytes(canonical_json_bytes(pair_sequence))[:20]}"
    )
    return {program: list(pose_steps)}


def score_future_spectra(
    measurement_log: str | Path,
    *,
    config_path: str | Path,
    snapshot_result: MLEResultInfo,
    snapshot: SpectralMLESnapshotInfo,
    request: FutureSpectralScoreRequestInfo,
    output_path: str | Path,
) -> FutureSpectralCandidateScoreInfo:
    """Score full frozen MLE versus each candidate-zero counterfactual by block."""
    if request.payload["snapshot_id"] != snapshot.payload["snapshot_id"]:
        raise DataReuseError("Future score request and snapshot differ.")
    fit = snapshot.payload["fit"]
    assert isinstance(fit, dict)
    if fit["mle_result_sha256"] != snapshot_result.result_sha256:
        raise DataReuseError(
            "Future scorer received an estimate other than the frozen snapshot fit."
        )
    config = SpectralMLEConfig.from_path(config_path)
    context = load_estimator_context(
        measurement_log,
        patch_edge_m=config.patch_edge_m,
        use_gpu=config.use_gpu,
    )
    if context.measurement_log.measurement_log_sha256 != request.payload[
        "current_measurement_log_sha256"
    ]:
        raise DataReuseError("Future score request is bound to a different MeasurementLog.")
    strengths = np.asarray(
        snapshot_result.arrays["patch_strength_by_isotope"], dtype=np.float64
    )
    result_isotopes = tuple(
        str(value) for value in snapshot_result.arrays["isotope_names"].tolist()
    )
    if result_isotopes != context.isotopes:
        raise ContractError("Frozen MLE isotope order differs from the current runtime model.")
    expected_shape = (len(context.isotopes), context.surface_geometry.chart_count)
    if strengths.shape != expected_shape:
        raise ContractError("Frozen MLE estimate and current surface dictionary differ.")
    estimator = SpectralMLE(context, config)
    full_prediction = estimator.predict_from_strengths(strengths)
    steps = tuple(int(value) for value in request.payload["requested_future_step_ids"])
    cutoff = snapshot.cutoff_step
    if any(step <= cutoff for step in steps):
        raise DataReuseError("Future spectral scorer received a non-future row.")
    step_to_index = {
        int(step): index
        for index, step in enumerate(context.measurement_log.step_ids)
    }
    if any(step not in step_to_index for step in steps):
        raise DataReuseError("Future score request names a row absent from the current log.")
    metadata = _metadata_by_step(context.measurement_log.root)
    tolerance = float(request.payload["grouping"]["height_tolerance_m"])  # type: ignore[index]
    grouped_by_pose: dict[tuple[int, int], list[int]] = defaultdict(list)
    positions = np.asarray(context.measurement_log.arrays["detector_pose_xyz"], dtype=np.float64)
    stations = np.asarray(context.measurement_log.arrays["station_id"], dtype=np.int64)
    fe_indices = np.asarray(
        context.measurement_log.arrays["fe_orientation_index"], dtype=np.int64
    )
    pb_indices = np.asarray(
        context.measurement_log.arrays["pb_orientation_index"], dtype=np.int64
    )
    for step in steps:
        index = step_to_index[step]
        height_group = int(np.rint(positions[index, 2] / tolerance))
        grouped_by_pose[(int(stations[index]), height_group)].append(step)
    grouped: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    grouping_semantics = request.payload["grouping"]["shield_program"]  # type: ignore[index]
    for (station, height), pose_steps in sorted(grouped_by_pose.items()):
        program_groups = _shield_program_groups(
            pose_steps,
            metadata=metadata,
            step_to_index=step_to_index,
            fe_indices=fe_indices,
            pb_indices=pb_indices,
            grouping_semantics=grouping_semantics,
        )
        for program, program_steps in program_groups.items():
            grouped[(station, height, program)].extend(program_steps)
    blocks: list[dict[str, object]] = []
    block_rows: dict[str, np.ndarray] = {}
    for (station, height, program), block_steps in sorted(grouped.items()):
        identity = {
            "station": station,
            "height": height,
            "program": program,
            "steps": block_steps,
        }
        block_id = f"spectral-block-{sha256_bytes(canonical_json_bytes(identity))[:20]}"
        blocks.append(
            {
                "block_id": block_id,
                "station_id": station,
                "height_group_id": f"height-{height}",
                "shield_program_id": program,
                "step_ids": block_steps,
            }
        )
        block_rows[block_id] = np.asarray(
            [step_to_index[step] for step in block_steps], dtype=np.int64
        )
    raw_candidates = snapshot.payload["candidates"]
    assert isinstance(raw_candidates, list)
    candidate_scores: list[dict[str, object]] = []
    observed = context.spectrum_counts
    for raw in raw_candidates:
        assert isinstance(raw, dict)
        isotope = str(raw["isotope"])
        try:
            isotope_index = context.isotopes.index(isotope)
        except ValueError as exc:
            raise ContractError("Snapshot candidate isotope is absent from current log.") from exc
        candidate_zero = strengths.copy()
        patch_ids = np.asarray(raw["patch_ids"], dtype=np.int64)
        if np.any(patch_ids < 0) or np.any(patch_ids >= expected_shape[1]):
            raise ContractError("Snapshot candidate patch is outside the surface dictionary.")
        candidate_zero[isotope_index, patch_ids] = 0.0
        zero_prediction = estimator.predict_from_strengths(candidate_zero)
        scores: list[dict[str, object]] = []
        for block in blocks:
            block_id = str(block["block_id"])
            rows = block_rows[block_id]
            full_deviance = _deviance(observed[rows], full_prediction[rows])
            zero_deviance = _deviance(observed[rows], zero_prediction[rows])
            scores.append(
                {
                    "block_id": block_id,
                    "full_model_poisson_deviance": full_deviance,
                    "candidate_zero_poisson_deviance": zero_deviance,
                    "log_likelihood_ratio": 0.5 * (zero_deviance - full_deviance),
                    "energy_bin_count": int(observed.shape[1]),
                }
            )
        candidate_scores.append(
            {
                "candidate_id": raw["candidate_id"],
                "cumulative_log_likelihood_ratio": float(
                    sum(float(score["log_likelihood_ratio"]) for score in scores)
                ),
                "block_scores": scores,
            }
        )
    payload = {
        "schema_version": 2,
        "score_family": "frozen_spectral_full_vs_candidate_zero_log_likelihood_ratio",
        "snapshot_id": snapshot.payload["snapshot_id"],
        "snapshot_sha256": snapshot.snapshot_sha256,
        "snapshot_data_cutoff_step": cutoff,
        "current_measurement_log_sha256": context.measurement_log.measurement_log_sha256,
        "future_step_ids": list(steps),
        "blocks": blocks,
        "candidates": candidate_scores,
        "safety": {
            "future_only": True,
            "snapshot_frozen": True,
            "same_observation_reweight": False,
        },
    }
    path = write_json_atomic(output_path, payload)
    return validate_future_spectral_candidate_score_v2(
        path,
        expected_snapshot=snapshot,
        expected_request=request,
    )


__all__ = ["score_future_spectra"]
