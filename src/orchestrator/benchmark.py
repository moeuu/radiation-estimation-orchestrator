"""End-to-end same-log pure-PF/count-MLE/spectral-MLE benchmark pipeline."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .adapters import MLECLIAdapter, PFCLIAdapter, load_estimator_pins
from .adapters.base import PRODUCTION_ALLOWED_DIRTY_PREFIXES, AdapterExecution, settings_from_dict
from .adapters.mle_cli import DEFAULT_MLE_COMMAND
from .adapters.pf_cli import DEFAULT_PF_COMMAND
from .contracts import validate_measurement_log, validate_mle_result, validate_pf_result
from .errors import ContractError
from .evaluation import evaluate_benchmark
from .hashing import load_json, sha256_file, write_json_atomic
from .manifests import build_benchmark_manifest, write_manifest_bundle

_BENCHMARK_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ESTIMATOR_RUNS = ("pf_strict", "mle_count", "mle_spectral")


def _resolve(base: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"Benchmark config field {field!r} must be a path string.")
    path = Path(value)
    candidate = base / path if not path.is_absolute() else path
    if candidate.is_symlink():
        raise ContractError(f"Benchmark config field {field!r} must not be a symlink.")
    return candidate.resolve()


def _mapping(payload: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(payload, dict):
        raise ContractError(f"Benchmark config field {field!r} must be an object.")
    return payload


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Resolved benchmark paths and execution policy."""

    source_path: Path
    benchmark_id: str
    measurement_log_path: Path
    truth_path: Path
    output_directory: Path
    pin_registry_path: Path
    pf_config_path: Path
    mle_count_config_path: Path
    mle_spectral_config_path: Path
    pf_profile: str
    random_seed: int
    expected_resolved_config_sha256: Mapping[str, str]
    pf_adapter: Mapping[str, object]
    mle_adapter: Mapping[str, object]

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkConfig:
        supplied = Path(path)
        if supplied.is_symlink():
            raise ContractError(f"Benchmark config must not be a symlink: {supplied}")
        source = supplied.resolve()
        payload = load_json(source)
        if payload.get("schema_version") != 1:
            raise ContractError("Benchmark config schema_version must be 1.")
        benchmark_id = payload.get("benchmark_id")
        if not isinstance(benchmark_id, str) or not _BENCHMARK_ID.fullmatch(benchmark_id):
            raise ContractError(
                "benchmark_id must contain only letters, digits, dot, underscore, dash."
            )
        base = source.parent
        estimator_configs = _mapping(payload.get("estimator_configs"), field="estimator_configs")
        adapters = _mapping(payload.get("adapters"), field="adapters")
        pf_adapter = _mapping(adapters.get("particle_filter", {}), field="adapters.particle_filter")
        mle_adapter = _mapping(adapters.get("surface_mle", {}), field="adapters.surface_mle")
        profile = str(payload.get("pf_profile", "pf_strict"))
        if profile != "pf_strict":
            raise ContractError("The common baseline benchmark requires pf_profile='pf_strict'.")
        expected_resolved = _mapping(
            payload.get("expected_resolved_estimator_config_sha256"),
            field="expected_resolved_estimator_config_sha256",
        )
        if set(expected_resolved) != set(_ESTIMATOR_RUNS) or any(
            not isinstance(value, str) or not _SHA256.fullmatch(value)
            for value in expected_resolved.values()
        ):
            raise ContractError(
                "expected_resolved_estimator_config_sha256 must contain exact lowercase "
                "SHA-256 values for pf_strict, mle_count, and mle_spectral."
            )
        instance = cls(
            source_path=source,
            benchmark_id=benchmark_id,
            measurement_log_path=_resolve(
                base, payload.get("measurement_log"), field="measurement_log"
            ),
            truth_path=_resolve(base, payload.get("truth"), field="truth"),
            output_directory=_resolve(
                base, payload.get("output_directory"), field="output_directory"
            ),
            pin_registry_path=_resolve(base, payload.get("pin_registry"), field="pin_registry"),
            pf_config_path=_resolve(
                base, estimator_configs.get("pf"), field="estimator_configs.pf"
            ),
            mle_count_config_path=_resolve(
                base, estimator_configs.get("mle_count"), field="estimator_configs.mle_count"
            ),
            mle_spectral_config_path=_resolve(
                base,
                estimator_configs.get("mle_spectral"),
                field="estimator_configs.mle_spectral",
            ),
            pf_profile=profile,
            random_seed=int(payload.get("random_seed", 0)),
            expected_resolved_config_sha256={
                str(name): str(value) for name, value in expected_resolved.items()
            },
            pf_adapter=dict(pf_adapter),
            mle_adapter=dict(mle_adapter),
        )
        instance._validate_expected_resolved_hashes()
        instance._validate_adapter_policy()
        instance._validate_paths()
        return instance

    def _validate_expected_resolved_hashes(self) -> None:
        if set(self.expected_resolved_config_sha256) != set(_ESTIMATOR_RUNS) or any(
            not isinstance(value, str) or not _SHA256.fullmatch(value)
            for value in self.expected_resolved_config_sha256.values()
        ):
            raise ContractError(
                "Benchmark expected resolved hashes must contain exact lowercase SHA-256 "
                "values for pf_strict, mle_count, and mle_spectral."
            )

    def _validate_adapter_policy(self) -> None:
        approved = set(PRODUCTION_ALLOWED_DIRTY_PREFIXES)
        for label, adapter in (
            ("particle_filter", self.pf_adapter),
            ("surface_mle", self.mle_adapter),
        ):
            if "command" in adapter:
                raise ContractError(
                    f"adapters.{label}.command cannot override the pinned production CLI."
                )
            if adapter.get("verify_revision", True) is not True:
                raise ContractError(f"adapters.{label}.verify_revision must be true.")
            if adapter.get("require_clean", True) is not True:
                raise ContractError(f"adapters.{label}.require_clean must be true.")
            raw_prefixes = adapter.get("allowed_dirty_prefixes", PRODUCTION_ALLOWED_DIRTY_PREFIXES)
            if not isinstance(raw_prefixes, list | tuple) or not all(
                isinstance(value, str) for value in raw_prefixes
            ):
                raise ContractError(
                    f"adapters.{label}.allowed_dirty_prefixes must be an array of strings."
                )
            normalized: set[str] = set()
            for raw_prefix in raw_prefixes:
                prefix = raw_prefix.replace("\\", "/")
                while prefix.startswith("./"):
                    prefix = prefix[2:]
                if not prefix or prefix.startswith(("/", "../")) or "/../" in prefix:
                    raise ContractError(
                        f"adapters.{label} has an invalid dirty prefix: {raw_prefix!r}."
                    )
                normalized.add(prefix if prefix.endswith("/") else prefix + "/")
            broadened = normalized - approved
            if broadened:
                raise ContractError(
                    f"adapters.{label} broadens the artifact-only dirty allowlist: "
                    f"{sorted(broadened)}"
                )

    def _validate_paths(self) -> None:
        for path in (
            self.source_path,
            self.truth_path,
            self.pin_registry_path,
            self.pf_config_path,
            self.mle_count_config_path,
            self.mle_spectral_config_path,
        ):
            if path.is_symlink() or not path.is_file():
                raise ContractError(f"Benchmark input must be a non-symlink file: {path}")
        if not self.measurement_log_path.is_dir() or self.measurement_log_path.is_symlink():
            raise ContractError(f"MeasurementLog path is invalid: {self.measurement_log_path}")
        try:
            self.truth_path.relative_to(self.measurement_log_path)
        except ValueError:
            pass
        else:
            raise ContractError("Truth must be outside the MeasurementLog directory.")
        for path in (
            self.pf_config_path,
            self.mle_count_config_path,
            self.mle_spectral_config_path,
        ):
            load_json(path)


class BenchmarkRunner:
    """Execute the required pipeline in an auditable fixed order."""

    def __init__(self, config: BenchmarkConfig, *, orchestrator_root: str | Path | None = None):
        self.config = config
        config._validate_expected_resolved_hashes()
        config._validate_adapter_policy()
        self.orchestrator_root = (
            Path(orchestrator_root).resolve()
            if orchestrator_root is not None
            else Path(__file__).resolve().parents[2]
        )

    def run(self) -> Path:
        config = self.config
        target = config.output_directory
        if target.exists():
            raise FileExistsError(f"Benchmark output already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{target.name}.staging"
        if staging.exists():
            raise FileExistsError(f"Stale benchmark staging directory exists: {staging}")
        staging.mkdir()
        started = datetime.now(UTC).isoformat()
        try:
            measurement_log = validate_measurement_log(config.measurement_log_path)
            pins = load_estimator_pins(config.pin_registry_path)
            pf_repository = self._repository_path(config.pf_adapter, pins["particle_filter"])
            mle_repository = self._repository_path(config.mle_adapter, pins["surface_mle"])
            pf_settings = settings_from_dict(
                config.pf_adapter,
                default_repository=pf_repository,
                default_command=DEFAULT_PF_COMMAND,
            )
            mle_settings = settings_from_dict(
                config.mle_adapter,
                default_repository=mle_repository,
                default_command=DEFAULT_MLE_COMMAND,
            )
            pf_adapter = PFCLIAdapter(pins["particle_filter"], pf_settings)
            mle_adapter = MLECLIAdapter(pins["surface_mle"], mle_settings)
            execution_root = staging / "executions"
            result_root = staging / "results"

            pf_execution = pf_adapter.run(
                measurement_log,
                config_path=config.pf_config_path,
                output_dir=result_root / "pf_strict",
                execution_dir=execution_root / "pf_strict",
                seed=config.random_seed,
                profile=config.pf_profile,
            )
            count_execution = mle_adapter.run(
                measurement_log,
                mode="count",
                config_path=config.mle_count_config_path,
                output_dir=result_root / "mle_count",
                execution_dir=execution_root / "mle_count",
            )
            spectral_execution = mle_adapter.run(
                measurement_log,
                mode="spectral",
                config_path=config.mle_spectral_config_path,
                output_dir=result_root / "mle_spectral",
                execution_dir=execution_root / "mle_spectral",
            )
            executions = {
                "pf_strict": pf_execution,
                "mle_count": count_execution,
                "mle_spectral": spectral_execution,
            }
            pf_result = validate_pf_result(
                result_root / "pf_strict",
                expected_variant=config.pf_profile,
                expected_isotopes=measurement_log.isotopes,
                expected_log_sha256=measurement_log.measurement_log_sha256,
                expected_commit=pins["particle_filter"].revision,
                expected_config_sha256=sha256_file(config.pf_config_path),
                expected_resolved_config_sha256=config.expected_resolved_config_sha256["pf_strict"],
                expected_record_count=measurement_log.record_count,
                expected_step_ids=measurement_log.step_ids,
            )
            count_result = validate_mle_result(
                result_root / "mle_count",
                expected_mode="count",
                expected_isotopes=measurement_log.isotopes,
                expected_log_sha256=measurement_log.measurement_log_sha256,
                expected_commit=pins["surface_mle"].revision,
                expected_config_sha256=sha256_file(config.mle_count_config_path),
                expected_resolved_config_sha256=config.expected_resolved_config_sha256["mle_count"],
            )
            spectral_result = validate_mle_result(
                result_root / "mle_spectral",
                expected_mode="spectral",
                expected_isotopes=measurement_log.isotopes,
                expected_log_sha256=measurement_log.measurement_log_sha256,
                expected_commit=pins["surface_mle"].revision,
                expected_config_sha256=sha256_file(config.mle_spectral_config_path),
                expected_resolved_config_sha256=config.expected_resolved_config_sha256[
                    "mle_spectral"
                ],
            )
            for name, result in (
                ("pf_strict", pf_result),
                ("mle_count", count_result),
                ("mle_spectral", spectral_result),
            ):
                if result.result_sha256 != executions[name].output_sha256:
                    raise ContractError(
                        f"{name} changed between execution and contract validation."
                    )
            executions = {
                name: self._retarget_execution_paths(execution, staging=staging)
                for name, execution in executions.items()
            }

            # This is deliberately the first read of config.truth_path.
            metrics = evaluate_benchmark(
                measurement_log=measurement_log,
                truth_path=config.truth_path,
                pf_result=pf_result,
                mle_count_result=count_result,
                mle_spectral_result=spectral_result,
                executions=executions,
            )
            metrics_path = write_json_atomic(staging / "metrics.json", metrics)
            completed = datetime.now(UTC).isoformat()
            estimator_config_file_hashes = {
                "pf_strict": sha256_file(config.pf_config_path),
                "mle_count": sha256_file(config.mle_count_config_path),
                "mle_spectral": sha256_file(config.mle_spectral_config_path),
            }
            manifest = build_benchmark_manifest(
                benchmark_id=config.benchmark_id,
                started_at_utc=started,
                completed_at_utc=completed,
                orchestrator_root=self.orchestrator_root,
                pin_registry_path=config.pin_registry_path,
                pins=pins,
                benchmark_config_path=config.source_path,
                estimator_config_file_hashes=estimator_config_file_hashes,
                expected_resolved_config_hashes=config.expected_resolved_config_sha256,
                measurement_log=measurement_log,
                truth_path=config.truth_path,
                metrics_path=metrics_path,
                executions=executions,
                pf_result=pf_result,
                mle_count_result=count_result,
                mle_spectral_result=spectral_result,
            )
            write_manifest_bundle(staging, manifest)
            os.replace(staging, target)
            return target
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _retarget_execution_paths(
        self, execution: AdapterExecution, *, staging: Path
    ) -> AdapterExecution:
        def relative_path(value: str) -> str:
            try:
                return Path(value).resolve().relative_to(staging.resolve()).as_posix()
            except ValueError as exc:
                raise ContractError(
                    f"Execution log path escaped the benchmark staging root: {value}"
                ) from exc

        return replace(
            execution,
            stdout_path=relative_path(execution.stdout_path),
            stderr_path=relative_path(execution.stderr_path),
        )

    def _repository_path(self, adapter: Mapping[str, object], pin: object) -> Path:
        configured = adapter.get("repository_path")
        if configured is not None:
            return Path(str(configured)).resolve()
        hint = pin.local_path_hint
        if hint is None:
            raise ContractError(f"No local repository_path for estimator {pin.name}.")
        path = Path(str(hint))
        return (
            (self.config.pin_registry_path.parent / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )
