"""Composition root for the resumable shared-runtime hybrid-v2 mission."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from orchestrator.errors import ContractError
from orchestrator.estimators.artifacts import repository_commit
from orchestrator.hashing import load_json, sha256_file, write_json_atomic
from orchestrator.manifests import orchestrator_provenance

from .live_config import LiveSpectralHybridRunConfig
from .live_planner import LivePFHybridPlanner
from .live_updater import LiveHybridEstimatorUpdater
from .mission import HybridMissionController, MissionPhase
from .runtime_client import ResumableAdaptiveRuntimeClient


class LiveSpectralHybridRunner:
    """Run or resume one exact-config, fixed-budget live hybrid mission."""

    def __init__(self, config: LiveSpectralHybridRunConfig) -> None:
        self.config = config

    def run(self) -> Path:
        config = self.config
        output = config.output_directory
        output.mkdir(parents=True, exist_ok=True)
        identity_path = output / "live_run_identity.json"
        identity = {
            "schema_version": 1,
            "milestone": "pf_mle_hybrid_live_v2",
            "hybrid_run_id": config.hybrid_run_id,
            "config_path": config.source_path.as_posix(),
            "config_sha256": sha256_file(config.source_path),
            "runtime_revision": config.runtime.revision,
            "runtime_scenario_sha256": sha256_file(config.runtime.scenario_path),
        }
        if identity_path.exists():
            if load_json(identity_path) != identity:
                raise ContractError(
                    "A live mission may resume only with its exact original config."
                )
        else:
            write_json_atomic(identity_path, identity)

        runtime = ResumableAdaptiveRuntimeClient(
            repository=config.runtime.repository_path,
            revision=config.runtime.revision,
            scenario=config.runtime.scenario_path,
            session_state_dir=config.runtime.session_state_directory,
            transcript_path=config.runtime.transcript_path,
            timeout_s=config.runtime.timeout_s,
            require_3d_candidate_diversity=(
                config.runtime.require_3d_candidate_diversity
            ),
        )
        try:
            updater = LiveHybridEstimatorUpdater(
                config=config,  # type: ignore[arg-type]
                work_directory=output / "estimation",
            )
            planner = LivePFHybridPlanner(
                config=config,  # type: ignore[arg-type]
                estimator_context_provider=updater.planning_context,
                work_directory=output / "planning",
                dsspp_config=config.planner.dsspp_config,
                dwell_time_s=config.planner.dwell_time_s,
            )
            controller = HybridMissionController(
                mission_id=config.hybrid_run_id,
                state_path=output / "mission_state.json",
                ledger_path=output / "mission_ledger.jsonl",
                budget=config.budget,
                runtime=runtime,
                planner=planner,
                updater=updater,
            )
            state = controller.run()
            if state.phase != MissionPhase.COMPLETE.value:
                raise ContractError("Live mission returned before fixed-budget completion.")
            completion = controller.ledger.entries[-1]
            if completion.event_type != "mission_completed":
                raise ContractError("Live mission ledger lacks its completion record.")
            final_estimate = completion.payload["final_estimate"]
            if not isinstance(final_estimate, dict):
                raise ContractError("Live mission completion lacks its final estimate.")
            execution_times = [
                float(execution["runtime_s"])
                for execution in updater.state["executions"].values()
            ]
            manifest = {
                "schema_version": 1,
                "milestone": "pf_mle_hybrid_live_v2",
                "hybrid_run_id": config.hybrid_run_id,
                "mode": config.mode.value,
                "mission": asdict(state),
                "mission_budget": asdict(config.budget),
                "mission_time_s": (
                    state.live_time_s
                    + state.travel_time_s
                    + state.shield_actuation_time_s
                ),
                "estimator_runtime_s": sum(execution_times),
                "estimator_revisions": {
                    "particle_filter": repository_commit(),
                    "surface_mle": repository_commit(),
                },
                "executions": {
                    "estimation": dict(sorted(updater.state["executions"].items())),
                    "planning": dict(sorted(planner.executions.items())),
                },
                "authoritative_result": final_estimate,
                "runtime": runtime.audit_record(),
                "artifacts": {
                    "mission_state_sha256": sha256_file(output / "mission_state.json"),
                    "mission_ledger_sha256": sha256_file(output / "mission_ledger.jsonl"),
                    "estimator_state_sha256": sha256_file(updater.state_path),
                },
                "orchestrator_provenance": orchestrator_provenance(
                    Path(__file__).resolve().parents[3]
                ),
                "safety": {
                    "truth_read": False,
                    "runtime_owns_observation_generation": True,
                    "runtime_receipt_lookup_required": True,
                    "observation_published_before_estimator_update": True,
                    "future_only_verification": True,
                    "mle_feedback_uses_raw_spectra_only": True,
                    "hard_prune": False,
                    "final_authoritative_estimator": "cold_full_log_spectral_mle",
                },
            }
            write_json_atomic(
                output / "live_hybrid_manifest.json", manifest, overwrite=True
            )
            return output
        finally:
            runtime.close(abort=False)


__all__ = ["LiveSpectralHybridRunner"]
