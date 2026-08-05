"""PF-owned DSS-PP planning adapter for live hybrid mission decisions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orchestrator.contracts import (
    validate_hybrid_planning_recommendation,
    validate_hybrid_planning_request,
    validate_measurement_log,
    validate_pf_checkpoint_v1,
)
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.estimators.artifacts import repository_commit
from orchestrator.estimators.planning import plan_from_checkpoint
from orchestrator.hashing import canonical_json_bytes, sha256_bytes, write_json_idempotent
from orchestrator.hybrid.prefix_log import measurement_records_sha256

from .mission import ActionDecision, CandidateSnapshot
from .run_config import SpectralHybridRunConfig


class LivePFHybridPlanner:
    """Turn a non-actuating PF recommendation into one attested runtime decision."""

    def __init__(
        self,
        *,
        config: SpectralHybridRunConfig,
        estimator_context_provider,
        work_directory: str | Path,
        dsspp_config: dict[str, object],
        dwell_time_s: float,
    ) -> None:
        self.config = config
        self.context_provider = estimator_context_provider
        self.work = Path(work_directory).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.dsspp_config = dict(dsspp_config)
        self.dsspp_config["augment_candidates"] = False
        self.dsspp_config["include_runtime_rescue_modes"] = False
        self.dsspp_config["include_global_surface_rescue_modes"] = False
        self.dwell_time_s = float(dwell_time_s)
        if self.dwell_time_s <= 0:
            raise ValueError("Live planning dwell time must be positive.")
        self.executions: dict[str, object] = {}

    def propose(
        self,
        *,
        data_cutoff_step: int,
        data_cutoff_station: int,
        estimator_prefix_sha256: str,
        candidates: CandidateSnapshot,
    ) -> ActionDecision:
        if data_cutoff_step == -1:
            return self._bootstrap(
                estimator_prefix_sha256=estimator_prefix_sha256,
                candidates=candidates,
            )
        context = dict(self.context_provider())
        log_path = context.get("measurement_log_path")
        if not isinstance(log_path, str):
            raise ContractError("Live planner lacks a durable MeasurementLog prefix.")
        log = validate_measurement_log(log_path)
        if log.measurement_log_sha256 != estimator_prefix_sha256:
            raise DataReuseError("Live planner context is stale relative to mission state.")
        if log.step_ids[-1] != data_cutoff_step:
            raise DataReuseError("Live planner log cutoff differs from mission state.")
        checkpoint_path = context.get("pf_checkpoint_path")
        if not isinstance(checkpoint_path, str):
            raise ContractError("Live planner lacks the current PF checkpoint.")
        checkpoint = validate_pf_checkpoint_v1(
            checkpoint_path,
            expected_source_run_id=str(log.manifest["run_id"]),
            expected_prefix_measurement_log_sha256=log.measurement_log_sha256,
        )
        external_raw = context.get("external_candidates", [])
        if not isinstance(external_raw, list):
            raise ContractError("Live planner external candidates must be an array.")
        external_modes = [
            {
                "mode_id": f"{raw['snapshot_id']}.{raw['candidate_id']}",
                "isotope": raw["isotope"],
                "position_xyz": raw["position_xyz"],
                "strength_cps_1m": raw["strength_cps_1m"],
                "weight": max(float(raw["planner_weight"]), 1e-12),
                "spread_m": raw["spread_m"],
                "verification_state": raw["state"],
                "source_snapshot_id": raw["snapshot_id"],
            }
            for raw in external_raw
        ]
        poses = [list(pose) for pose in candidates.candidate_poses_xyz]
        request_identity = {
            "cutoff": data_cutoff_step,
            "prefix": estimator_prefix_sha256,
            "candidates": candidates.snapshot_sha256,
            "external_modes": external_modes,
        }
        current_pose = np.asarray(log.arrays["detector_pose_xyz"][-1], dtype=float).tolist()
        current_pair = int(log.arrays["fe_orientation_index"][-1]) * 8 + int(
            log.arrays["pb_orientation_index"][-1]
        )
        dsspp_hash = sha256_bytes(canonical_json_bytes(self.dsspp_config))
        request_payload = {
            "schema_version": 1,
            "request_id": (
                f"live-plan-{sha256_bytes(canonical_json_bytes(request_identity))[:20]}"
            ),
            "source_run_id": log.manifest["run_id"],
            "data_cutoff_step": data_cutoff_step,
            "data_cutoff_station": data_cutoff_station,
            "covered_records_sha256": measurement_records_sha256(log),
            "pf_resolved_config_sha256": (
                self.config.expected_pf_resolved_config_sha256
            ),
            "current_pose_xyz": current_pose,
            "current_pair_id": current_pair,
            "visited_poses_xyz": np.asarray(
                log.arrays["detector_pose_xyz"], dtype=float
            ).tolist(),
            "candidate_poses_xyz": poses,
            "candidate_attestation": {
                "candidate_poses_sha256": sha256_bytes(canonical_json_bytes(poses)),
                "workspace_sha256": candidates.path_attestation_sha256,
                "planning_config_sha256": dsspp_hash,
                "collision_checked": True,
                "reachability_filtered": True,
            },
            "dsspp_config": self.dsspp_config,
            "external_modes": external_modes,
            "bounds_xyz": {
                "min": np.min(np.asarray(poses), axis=0).tolist(),
                "max": np.max(np.asarray(poses), axis=0).tolist(),
            },
            "continuous_height_bounds_m": [
                float(min(pose[2] for pose in candidates.candidate_poses_xyz)),
                float(max(pose[2] for pose in candidates.candidate_poses_xyz)),
            ],
            "allowed_pair_ids": list(candidates.allowed_pair_ids),
        }
        name = f"live-plan-after-step-{data_cutoff_step}"
        request_path = write_json_idempotent(
            self.work / "requests" / f"{name}.json",
            request_payload,
        )
        request = validate_hybrid_planning_request(request_path)
        output = self.work / "results" / name
        recommendation_path = output / "hybrid_planning_recommendation.json"
        if recommendation_path.exists():
            recommendation = validate_hybrid_planning_recommendation(
                recommendation_path,
                expected_request=request,
            )
        else:
            output.mkdir(parents=True, exist_ok=False)
            recommendation = plan_from_checkpoint(
                log,
                config_path=self.config.pf_config_path,
                checkpoint=checkpoint,
                request=request,
                output_path=recommendation_path,
            )
        self.executions[name] = {
            "estimator": "local:particle_filter:checkpoint_planning",
            "revision": repository_commit(),
            "measurement_log_sha256": log.measurement_log_sha256,
            "recommendation_sha256": recommendation.recommendation_sha256,
        }
        selected = recommendation.payload["selected_action"]
        assert isinstance(selected, dict)
        shield_program = selected["shield_program"]
        assert isinstance(shield_program, dict)
        pair_ids = shield_program["pair_ids"]
        if not isinstance(pair_ids, list) or not pair_ids:
            raise ContractError("DSS-PP returned an empty shield program.")
        program = tuple(int(pair_id) for pair_id in pair_ids)
        candidate_index = int(selected["candidate_index"])
        identity = {
            "recommendation": recommendation.recommendation_sha256,
            "candidate_snapshot": candidates.snapshot_sha256,
        }
        return ActionDecision(
            decision_id=f"decision-{sha256_bytes(canonical_json_bytes(identity))[:20]}",
            estimator_data_cutoff_step=data_cutoff_step,
            estimator_data_cutoff_station=data_cutoff_station,
            estimator_prefix_sha256=estimator_prefix_sha256,
            candidate_snapshot_id=candidates.snapshot_id,
            candidate_snapshot_sha256=candidates.snapshot_sha256,
            candidate_index=candidate_index,
            target_pose_xyz=candidates.candidate_poses_xyz[candidate_index],
            selected_path_sha256=candidates.candidate_path_sha256[candidate_index],
            shield_program_pair_ids=program,
            dwell_time_s=self.dwell_time_s,
            station_id=data_cutoff_station + 1,
            station_complete=True,
        )

    def _bootstrap(
        self,
        *,
        estimator_prefix_sha256: str,
        candidates: CandidateSnapshot,
    ) -> ActionDecision:
        index = min(
            range(len(candidates.travel_costs_s)),
            key=candidates.travel_costs_s.__getitem__,
        )
        pair_id = candidates.allowed_pair_ids[0]
        identity = {
            "bootstrap": True,
            "candidate_snapshot": candidates.snapshot_sha256,
            "candidate_index": index,
            "pair_id": pair_id,
        }
        return ActionDecision(
            decision_id=f"decision-{sha256_bytes(canonical_json_bytes(identity))[:20]}",
            estimator_data_cutoff_step=-1,
            estimator_data_cutoff_station=-1,
            estimator_prefix_sha256=estimator_prefix_sha256,
            candidate_snapshot_id=candidates.snapshot_id,
            candidate_snapshot_sha256=candidates.snapshot_sha256,
            candidate_index=index,
            target_pose_xyz=candidates.candidate_poses_xyz[index],
            selected_path_sha256=candidates.candidate_path_sha256[index],
            shield_program_pair_ids=(pair_id,),
            dwell_time_s=self.dwell_time_s,
            station_id=0,
            station_complete=True,
        )


__all__ = ["LivePFHybridPlanner"]
