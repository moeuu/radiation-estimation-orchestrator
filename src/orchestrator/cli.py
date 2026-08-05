"""Public runtime acquisition, local inference, hybrid, and validation CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .acquisition import acquire_measurement_log
from .benchmark import BenchmarkConfig, BenchmarkRunner
from .contracts import (
    validate_future_spectral_score_request_v1,
    validate_hybrid_planning_request,
    validate_measurement_log,
    validate_mle_result,
    validate_mle_snapshot,
    validate_pf_checkpoint_v1,
    validate_pf_result,
    validate_pf_rj_directive_v1,
    validate_pf_rj_receipt_v1,
    validate_spectral_mle_snapshot_v3,
)
from .errors import ContractError
from .estimators.artifacts import run_pf_checkpoint, run_spectral_mle
from .estimators.future_scoring import score_future_spectra
from .estimators.planning import plan_from_checkpoint
from .estimators.rj import apply_exact_rj
from .evaluation import evaluate_spectral_mission
from .hashing import load_json, write_json_atomic
from .hybrid_v2.live import LiveSpectralHybridRunner
from .hybrid_v2.live_config import LiveSpectralHybridRunConfig
from .hybrid_v2.offline import SpectralOfflineHybridController
from .hybrid_v2.run_config import SpectralHybridRunConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radiation-estimation-orchestrator",
        description=(
            "Run in-repository PF/MLE/hybrid inference over MeasurementLogs produced "
            "by the shared simulation runtime."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser(
        "acquire",
        help="Run a private plan through the shared simulation runtime.",
    )
    acquire.add_argument("--plan", type=Path, required=True)
    benchmark = commands.add_parser(
        "benchmark", help="Run a versioned pure-estimator same-log benchmark."
    )
    benchmark.add_argument("--config", type=Path, required=True)
    hybrid_v2 = commands.add_parser(
        "hybrid-v2-replay",
        help="Run raw-spectrum causal PF plus spectral-MLE replay.",
    )
    hybrid_v2.add_argument("--config", type=Path, required=True)
    hybrid_v2_live = commands.add_parser(
        "hybrid-v2-live",
        help="Run or resume collision-attested live 3-D PF plus spectral-MLE.",
    )
    hybrid_v2_live.add_argument("--config", type=Path, required=True)
    evaluate_live = commands.add_parser(
        "evaluate-live",
        help="Open separate truth only after a completed live hybrid mission.",
    )
    evaluate_live.add_argument("--manifest", type=Path, required=True)
    evaluate_live.add_argument("--truth", type=Path, required=True)
    evaluate_live.add_argument("--output", type=Path, required=True)

    pf_checkpoint = commands.add_parser(
        "pf-checkpoint",
        help="Run or resume the local strict full-spectrum PF.",
    )
    pf_checkpoint.add_argument("--measurement-log", type=Path, required=True)
    pf_checkpoint.add_argument("--config", type=Path, required=True)
    pf_checkpoint.add_argument("--output-dir", type=Path, required=True)
    pf_checkpoint.add_argument("--seed", type=int, default=0)
    pf_checkpoint.add_argument("--checkpoint-in", type=Path, default=None)
    spectral_mle = commands.add_parser(
        "spectral-mle",
        help="Run the local all-surface raw-spectrum MLE.",
    )
    spectral_mle.add_argument("--measurement-log", type=Path, required=True)
    spectral_mle.add_argument("--config", type=Path, required=True)
    spectral_mle.add_argument("--output-dir", type=Path, required=True)
    spectral_mle.add_argument("--warm-start", type=Path, default=None)
    spectral_score = commands.add_parser(
        "future-spectral-score",
        help="Score frozen MLE candidates on once-only future spectral blocks.",
    )
    spectral_score.add_argument("--measurement-log", type=Path, required=True)
    spectral_score.add_argument("--config", type=Path, required=True)
    spectral_score.add_argument("--snapshot-result", type=Path, required=True)
    spectral_score.add_argument("--snapshot", type=Path, required=True)
    spectral_score.add_argument("--request", type=Path, required=True)
    spectral_score.add_argument("--output", type=Path, required=True)
    exact_rj = commands.add_parser(
        "exact-rj",
        help="Apply one local PF-owned exact reversible-jump directive.",
    )
    exact_rj.add_argument("--measurement-log", type=Path, required=True)
    exact_rj.add_argument("--config", type=Path, required=True)
    exact_rj.add_argument("--checkpoint-in", type=Path, required=True)
    exact_rj.add_argument("--directive", type=Path, required=True)
    exact_rj.add_argument("--output-dir", type=Path, required=True)
    checkpoint_plan = commands.add_parser(
        "checkpoint-plan",
        help="Plan the next attested 3-D action from a local PF checkpoint.",
    )
    checkpoint_plan.add_argument("--measurement-log", type=Path, required=True)
    checkpoint_plan.add_argument("--config", type=Path, required=True)
    checkpoint_plan.add_argument("--checkpoint-in", type=Path, required=True)
    checkpoint_plan.add_argument("--request", type=Path, required=True)
    checkpoint_plan.add_argument("--output", type=Path, required=True)

    validate_log = commands.add_parser(
        "validate-log",
        help="Validate MeasurementLog v1 or raw full-spectrum v2.",
    )
    validate_log.add_argument("--run-dir", type=Path, required=True)
    validate_pf = commands.add_parser("validate-pf-result", help="Validate PFResult v1.")
    validate_pf.add_argument("--result-dir", type=Path, required=True)
    validate_mle = commands.add_parser("validate-mle-result", help="Validate MLEResult v1.")
    validate_mle.add_argument("--result-dir", type=Path, required=True)
    validate_mle.add_argument("--mode", choices=("count", "spectral"), default=None)
    validate_snapshot = commands.add_parser(
        "validate-mle-snapshot",
        help="Validate the reserved future MLESnapshot/data_cutoff contract.",
    )
    validate_snapshot.add_argument("--snapshot", type=Path, required=True)
    validate_spectral_snapshot = commands.add_parser(
        "validate-spectral-snapshot",
        help="Validate a raw-spectrum hybrid-v2 MLE snapshot.",
    )
    validate_spectral_snapshot.add_argument("--snapshot", type=Path, required=True)
    validate_checkpoint = commands.add_parser(
        "validate-pf-checkpoint",
        help="Validate an opaque causal PF checkpoint bundle.",
    )
    validate_checkpoint.add_argument("--checkpoint", type=Path, required=True)
    validate_rj_directive = commands.add_parser(
        "validate-rj-directive",
        help="Validate an MLE-informed exact-RJ proposal directive.",
    )
    validate_rj_directive.add_argument("--directive", type=Path, required=True)
    validate_rj_receipt = commands.add_parser(
        "validate-rj-receipt",
        help="Validate and recompute one PF exact-RJ receipt.",
    )
    validate_rj_receipt.add_argument("--receipt", type=Path, required=True)

    return parser


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    """Run one orchestrator command."""
    args = _parser().parse_args(None if argv is None else list(argv))
    if args.command == "acquire":
        result = acquire_measurement_log(args.plan)
        _print(
            {
                "status": "complete",
                "measurement_log_path": result.measurement_log_path.as_posix(),
                "measurement_log_sha256": result.measurement_log_sha256,
                "record_count": result.record_count,
                "run_id": result.run_id,
            }
        )
        return 0
    if args.command == "benchmark":
        config = BenchmarkConfig.load(args.config)
        if config.schema_version != 2:
            raise ContractError(
                "Benchmark v1 is archived; the active CLI accepts runtime-only v2 configs."
            )
        output = BenchmarkRunner(config).run()
        _print({"status": "complete", "output_directory": output.as_posix()})
        return 0
    if args.command == "hybrid-v2-replay":
        output = SpectralOfflineHybridController(
            SpectralHybridRunConfig.load(args.config)
        ).run()
        _print({"status": "inference_complete", "output_directory": output.as_posix()})
        return 0
    if args.command == "hybrid-v2-live":
        output = LiveSpectralHybridRunner(
            LiveSpectralHybridRunConfig.load(args.config)
        ).run()
        _print({"status": "mission_complete", "output_directory": output.as_posix()})
        return 0
    if args.command == "evaluate-live":
        manifest = load_json(args.manifest)
        if manifest.get("milestone") != "pf_mle_hybrid_live_v2":
            raise ContractError("Live evaluation requires a completed live-v2 manifest.")
        authoritative = manifest.get("authoritative_result")
        if not isinstance(authoritative, dict):
            raise ContractError("Live manifest lacks its authoritative result.")
        log_path = authoritative.get("measurement_log_path")
        result_path = authoritative.get("result_path")
        if not isinstance(log_path, str) or not isinstance(result_path, str):
            raise ContractError("Live authoritative result paths are invalid.")
        log = validate_measurement_log(log_path)
        result = validate_mle_result(
            result_path,
            expected_mode="spectral",
            expected_log_sha256=log.measurement_log_sha256,
        )
        if result.result_sha256 != authoritative.get("result_sha256"):
            raise ContractError("Live final MLE differs from the mission manifest.")
        metrics = evaluate_spectral_mission(
            measurement_log=log,
            truth_path=args.truth,
            mle_spectral_result=result,
            estimator_runtime_s=float(manifest.get("estimator_runtime_s", 0.0)),
        )
        output = write_json_atomic(args.output, metrics)
        _print({"status": "evaluation_complete", "output": output.as_posix()})
        return 0
    if args.command == "pf-checkpoint":
        checkpoint = (
            None
            if args.checkpoint_in is None
            else validate_pf_checkpoint_v1(args.checkpoint_in)
        )
        artifacts = run_pf_checkpoint(
            args.measurement_log,
            config_path=args.config,
            output_directory=args.output_dir,
            random_seed=args.seed,
            checkpoint_in=checkpoint,
        )
        _print(
            {
                "status": "complete",
                "result_sha256": artifacts.result.result_sha256,
                "checkpoint_sha256": artifacts.checkpoint.checkpoint_sha256,
            }
        )
        return 0
    if args.command == "spectral-mle":
        warm = None if args.warm_start is None else validate_mle_result(args.warm_start)
        result = run_spectral_mle(
            args.measurement_log,
            config_path=args.config,
            output_directory=args.output_dir,
            warm_start_result=warm,
        )
        _print(
            {
                "status": "complete",
                "converged": result.diagnostics["converged"],
                "result_sha256": result.result_sha256,
            }
        )
        return 0
    if args.command == "future-spectral-score":
        result = score_future_spectra(
            args.measurement_log,
            config_path=args.config,
            snapshot_result=validate_mle_result(args.snapshot_result, expected_mode="spectral"),
            snapshot=validate_spectral_mle_snapshot_v3(args.snapshot),
            request=validate_future_spectral_score_request_v1(args.request),
            output_path=args.output,
        )
        _print({"status": "complete", "score_sha256": result.score_sha256})
        return 0
    if args.command == "exact-rj":
        checkpoint, receipt = apply_exact_rj(
            args.measurement_log,
            config_path=args.config,
            checkpoint_in=validate_pf_checkpoint_v1(args.checkpoint_in),
            directive=validate_pf_rj_directive_v1(args.directive),
            output_directory=args.output_dir,
        )
        _print(
            {
                "status": "complete",
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "receipt_sha256": receipt.receipt_sha256,
            }
        )
        return 0
    if args.command == "checkpoint-plan":
        recommendation = plan_from_checkpoint(
            args.measurement_log,
            config_path=args.config,
            checkpoint=validate_pf_checkpoint_v1(args.checkpoint_in),
            request=validate_hybrid_planning_request(args.request),
            output_path=args.output,
        )
        _print(
            {
                "status": "complete",
                "recommendation_sha256": recommendation.recommendation_sha256,
            }
        )
        return 0
    if args.command == "validate-log":
        info = validate_measurement_log(args.run_dir)
        _print(
            {
                "status": "valid",
                "schema_version": info.schema_version,
                "record_count": info.record_count,
                "measurement_log_sha256": info.measurement_log_sha256,
            }
        )
        return 0
    if args.command == "validate-pf-result":
        info = validate_pf_result(args.result_dir)
        _print({"status": "valid", "schema_version": 1, "result_sha256": info.result_sha256})
        return 0
    if args.command == "validate-mle-result":
        info = validate_mle_result(args.result_dir, expected_mode=args.mode)
        _print(
            {
                "status": "valid",
                "schema_version": 1,
                "mode": info.mode,
                "result_sha256": info.result_sha256,
            }
        )
        return 0
    if args.command == "validate-mle-snapshot":
        payload = validate_mle_snapshot(args.snapshot)
        _print(
            {
                "status": "valid",
                "schema_version": 1,
                "snapshot_id": payload["snapshot_id"],
                "data_cutoff_step": payload["data_cutoff_step"],
            }
        )
        return 0
    if args.command == "validate-spectral-snapshot":
        info = validate_spectral_mle_snapshot_v3(args.snapshot)
        _print(
            {
                "status": "valid",
                "schema_version": 3,
                "snapshot_id": info.payload["snapshot_id"],
                "data_cutoff_step": info.cutoff_step,
            }
        )
        return 0
    if args.command == "validate-pf-checkpoint":
        info = validate_pf_checkpoint_v1(args.checkpoint)
        _print(
            {
                "status": "valid",
                "schema_version": 1,
                "checkpoint_id": info.payload["checkpoint_id"],
                "data_cutoff_step": info.cutoff_step,
            }
        )
        return 0
    if args.command == "validate-rj-directive":
        info = validate_pf_rj_directive_v1(args.directive)
        _print(
            {
                "status": "valid",
                "schema_version": 1,
                "directive_id": info.payload["directive_id"],
            }
        )
        return 0
    if args.command == "validate-rj-receipt":
        info = validate_pf_rj_receipt_v1(args.receipt)
        _print(
            {
                "status": "valid",
                "schema_version": 1,
                "receipt_id": info.payload["receipt_id"],
            }
        )
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
