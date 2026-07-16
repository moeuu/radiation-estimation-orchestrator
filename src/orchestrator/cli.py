"""Public orchestration, validation, and forward-conformance CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .adapters import load_estimator_pins
from .adapters.base import verify_repository_revision
from .benchmark import BenchmarkConfig, BenchmarkRunner
from .conformance import CLIForwardResponseProvider, run_forward_response_conformance
from .contracts import (
    validate_measurement_log,
    validate_mle_result,
    validate_mle_snapshot,
    validate_pf_result,
)
from .errors import ContractError
from .hashing import load_json, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rotating-shield-orchestrator",
        description="Pinned same-log PF/MLE benchmarking without estimator source duplication.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark", help="Run the complete three-estimator benchmark.")
    benchmark.add_argument("--config", type=Path, required=True)

    validate_log = commands.add_parser("validate-log", help="Validate MeasurementLog v1.")
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

    pins = commands.add_parser("verify-pins", help="Verify local estimator checkouts against pins.")
    pins.add_argument("--registry", type=Path, default=Path("PINNED_ESTIMATORS.json"))
    pins.add_argument("--pf-repository", type=Path, default=None)
    pins.add_argument("--mle-repository", type=Path, default=None)
    pins.add_argument(
        "--allowed-dirty-prefix",
        action="append",
        default=["results/", "logs/", "build/", ".cache/", ".venv/"],
    )

    conformance = commands.add_parser(
        "conformance", help="Compare independent PF/MLE unit-strength response CLIs."
    )
    conformance.add_argument("--fixture", type=Path, required=True)
    conformance.add_argument("--pf-provider", type=Path, required=True)
    conformance.add_argument("--mle-provider", type=Path, required=True)
    conformance.add_argument("--output-dir", type=Path, required=True)
    conformance.add_argument("--rtol", type=float, default=1e-9)
    conformance.add_argument("--atol", type=float, default=1e-12)
    return parser


def _provider(path: Path, *, name: str) -> CLIForwardResponseProvider:
    payload = load_json(path)
    configured_name = payload.get("provider")
    repository = payload.get("repository_path")
    revision = payload.get("revision")
    revision_type = payload.get("revision_type")
    require_clean = payload.get("require_clean")
    command = payload.get("command")
    if (
        configured_name != name
        or not isinstance(repository, str)
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or revision_type != "commit"
        or require_clean is not True
        or not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) for value in command)
    ):
        raise ContractError(
            f"{path} must name provider {name!r}, pin an exact 40-character lowercase "
            "commit, require a clean checkout, and supply repository_path plus a string "
            "command array."
        )
    repository_path = Path(repository)
    if not repository_path.is_absolute():
        repository_path = (path.resolve().parent / repository_path).resolve()
    return CLIForwardResponseProvider(
        name,
        repository_path=repository_path,
        revision=revision,
        command_template=tuple(command),
        provider_config_sha256=sha256_file(path),
        timeout_s=float(payload.get("timeout_s", 300.0)),
    )


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    if args.command == "benchmark":
        output = BenchmarkRunner(BenchmarkConfig.load(args.config)).run()
        _print({"status": "complete", "output_directory": output.as_posix()})
        return 0
    if args.command == "validate-log":
        info = validate_measurement_log(args.run_dir)
        _print(
            {
                "status": "valid",
                "schema_version": 1,
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
    if args.command == "verify-pins":
        registry = args.registry.resolve()
        pins = load_estimator_pins(registry)
        roots = {
            "particle_filter": args.pf_repository,
            "surface_mle": args.mle_repository,
        }
        result: dict[str, object] = {}
        for name, pin in pins.items():
            root = roots[name]
            if root is None:
                if pin.local_path_hint is None:
                    raise ValueError(f"No local path available for {name}.")
                root = (registry.parent / pin.local_path_hint).resolve()
            observed, dirty = verify_repository_revision(
                root,
                pin,
                require_clean=True,
                allowed_dirty_prefixes=tuple(args.allowed_dirty_prefix),
            )
            result[name] = {"revision": observed, "dirty_worktree": dirty}
        _print({"status": "valid", "estimators": result})
        return 0
    if args.command == "conformance":
        report = run_forward_response_conformance(
            fixture_path=args.fixture,
            pf_provider=_provider(args.pf_provider, name="particle_filter"),
            mle_provider=_provider(args.mle_provider, name="surface_mle"),
            output_directory=args.output_dir,
            rtol=args.rtol,
            atol=args.atol,
        )
        _print(report)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
