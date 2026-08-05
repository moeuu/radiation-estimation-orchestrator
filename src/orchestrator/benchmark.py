"""End-to-end same-log pure-estimator benchmark pipelines."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import validate_measurement_log, validate_mle_result, validate_pf_result
from .errors import ContractError
from .estimators.artifacts import (
    mle_resolved_config_sha256,
    pf_resolved_config_sha256,
    repository_commit,
)
from .estimators.local_services import (
    LocalPFCheckpointService,
    LocalSpectralMLEService,
)
from .evaluation import evaluate_benchmark
from .hashing import load_json, sha256_file, write_json_atomic
from .manifests import (
    build_benchmark_manifest,
    environment_provenance,
    orchestrator_provenance,
    write_manifest_bundle,
)

if TYPE_CHECKING:
    from .adapters.base import AdapterExecution

_BENCHMARK_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_V1_ESTIMATOR_RUNS = ("pf_strict", "mle_count", "mle_spectral")
_V2_ESTIMATOR_RUNS = ("pf_strict", "mle_spectral")


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
    schema_version: int
    measurement_log_schema_version: int
    estimator_runs: tuple[str, ...]
    benchmark_id: str
    measurement_log_path: Path
    truth_path: Path
    output_directory: Path
    pin_registry_path: Path | None
    pf_config_path: Path
    mle_count_config_path: Path | None
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
        schema_version = payload.get("schema_version")
        if schema_version not in {1, 2}:
            raise ContractError("Benchmark config schema_version must be 1 or 2.")
        if schema_version == 1:
            estimator_runs = _V1_ESTIMATOR_RUNS
            measurement_log_schema_version = 1
        else:
            if payload.get("estimator_runs") != list(_V2_ESTIMATOR_RUNS):
                raise ContractError(
                    "Benchmark config v2 requires estimator_runs="
                    "['pf_strict', 'mle_spectral']."
                )
            estimator_runs = _V2_ESTIMATOR_RUNS
            measurement_log_schema_version = payload.get("measurement_log_schema_version")
            if measurement_log_schema_version != 2:
                raise ContractError(
                    "Benchmark config v2 requires measurement_log_schema_version=2."
                )
        benchmark_id = payload.get("benchmark_id")
        if not isinstance(benchmark_id, str) or not _BENCHMARK_ID.fullmatch(benchmark_id):
            raise ContractError(
                "benchmark_id must contain only letters, digits, dot, underscore, dash."
            )
        base = source.parent
        estimator_configs = _mapping(payload.get("estimator_configs"), field="estimator_configs")
        adapters = _mapping(payload.get("adapters", {}), field="adapters")
        pf_adapter = _mapping(adapters.get("particle_filter", {}), field="adapters.particle_filter")
        mle_adapter = _mapping(adapters.get("surface_mle", {}), field="adapters.surface_mle")
        profile = str(payload.get("pf_profile", "pf_strict"))
        if profile != "pf_strict":
            raise ContractError("The common baseline benchmark requires pf_profile='pf_strict'.")
        count_config_path = (
            _resolve(base, estimator_configs.get("mle_count"), field="estimator_configs.mle_count")
            if "mle_count" in estimator_runs
            else None
        )
        pf_config_path = _resolve(
            base, estimator_configs.get("pf"), field="estimator_configs.pf"
        )
        mle_spectral_config_path = _resolve(
            base,
            estimator_configs.get("mle_spectral"),
            field="estimator_configs.mle_spectral",
        )
        if schema_version == 2:
            if "pin_registry" in payload or adapters:
                raise ContractError(
                    "Benchmark v2 uses local estimators; pin_registry/adapters are forbidden."
                )
            expected_resolved: Mapping[str, str] = {
                "pf_strict": pf_resolved_config_sha256(pf_config_path),
                "mle_spectral": mle_resolved_config_sha256(mle_spectral_config_path),
            }
        else:
            expected_resolved = _mapping(
                payload.get("expected_resolved_estimator_config_sha256"),
                field="expected_resolved_estimator_config_sha256",
            )  # type: ignore[assignment]
        instance = cls(
            source_path=source,
            schema_version=int(schema_version),
            measurement_log_schema_version=int(measurement_log_schema_version),
            estimator_runs=estimator_runs,
            benchmark_id=benchmark_id,
            measurement_log_path=_resolve(
                base, payload.get("measurement_log"), field="measurement_log"
            ),
            truth_path=_resolve(base, payload.get("truth"), field="truth"),
            output_directory=_resolve(
                base, payload.get("output_directory"), field="output_directory"
            ),
            pin_registry_path=(
                None
                if schema_version == 2
                else _resolve(base, payload.get("pin_registry"), field="pin_registry")
            ),
            pf_config_path=pf_config_path,
            mle_count_config_path=count_config_path,
            mle_spectral_config_path=mle_spectral_config_path,
            pf_profile=profile,
            random_seed=int(payload.get("random_seed", 0)),
            expected_resolved_config_sha256={
                str(name): str(value) for name, value in expected_resolved.items()
            },
            pf_adapter=dict(pf_adapter),
            mle_adapter=dict(mle_adapter),
        )
        instance._validate_expected_resolved_hashes()
        if schema_version == 1:
            instance._validate_adapter_policy()
        instance._validate_paths()
        return instance

    def _validate_expected_resolved_hashes(self) -> None:
        if set(self.expected_resolved_config_sha256) != set(self.estimator_runs) or any(
            not isinstance(value, str) or not _SHA256.fullmatch(value)
            for value in self.expected_resolved_config_sha256.values()
        ):
            raise ContractError(
                "Benchmark expected resolved hashes must contain exact lowercase SHA-256 "
                f"values for {', '.join(self.estimator_runs)}."
            )

    def _validate_adapter_policy(self) -> None:
        from .adapters.base import PRODUCTION_ALLOWED_DIRTY_PREFIXES

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
        required_paths = [
            self.source_path,
            self.truth_path,
            self.pf_config_path,
            self.mle_spectral_config_path,
        ]
        if self.pin_registry_path is not None:
            required_paths.append(self.pin_registry_path)
        if self.mle_count_config_path is not None:
            required_paths.append(self.mle_count_config_path)
        for path in required_paths:
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
        estimator_paths = [self.pf_config_path, self.mle_spectral_config_path]
        if self.mle_count_config_path is not None:
            estimator_paths.append(self.mle_count_config_path)
        for path in estimator_paths:
            load_json(path)


class BenchmarkRunner:
    """Execute the required pipeline in an auditable fixed order."""

    def __init__(self, config: BenchmarkConfig, *, orchestrator_root: str | Path | None = None):
        self.config = config
        config._validate_expected_resolved_hashes()
        if config.schema_version == 1:
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
            if measurement_log.schema_version != config.measurement_log_schema_version:
                raise ContractError(
                    "Benchmark config expects MeasurementLog schema version "
                    f"{config.measurement_log_schema_version}, got "
                    f"{measurement_log.schema_version}."
                )
            if config.schema_version == 2:
                return self._run_local_v2(
                    measurement_log=measurement_log,
                    staging=staging,
                    target=target,
                    started=started,
                )
            # Historical schema-v1 execution is isolated here and is not exposed by
            # the production CLI. Importing the active benchmark does not load an
            # external-estimator adapter.
            from .adapters import MLECLIAdapter, PFCLIAdapter, load_estimator_pins
            from .adapters.base import settings_from_dict
            from .adapters.mle_cli import DEFAULT_MLE_COMMAND
            from .adapters.pf_cli import DEFAULT_PF_COMMAND

            assert config.pin_registry_path is not None
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
            count_execution = None
            if config.mle_count_config_path is not None:
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
            executions: dict[str, AdapterExecution] = {
                "pf_strict": pf_execution,
                "mle_spectral": spectral_execution,
            }
            if count_execution is not None:
                executions["mle_count"] = count_execution
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
            count_result = None
            if config.mle_count_config_path is not None:
                count_result = validate_mle_result(
                    result_root / "mle_count",
                    expected_mode="count",
                    expected_isotopes=measurement_log.isotopes,
                    expected_log_sha256=measurement_log.measurement_log_sha256,
                    expected_commit=pins["surface_mle"].revision,
                    expected_config_sha256=sha256_file(config.mle_count_config_path),
                    expected_resolved_config_sha256=config.expected_resolved_config_sha256[
                        "mle_count"
                    ],
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
            validated_results = {
                "pf_strict": pf_result,
                "mle_spectral": spectral_result,
            }
            if count_result is not None:
                validated_results["mle_count"] = count_result
            for name, result in validated_results.items():
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
            estimator_config_file_hashes: dict[str, str] = {
                "pf_strict": sha256_file(config.pf_config_path),
                "mle_spectral": sha256_file(config.mle_spectral_config_path),
            }
            if config.mle_count_config_path is not None:
                estimator_config_file_hashes["mle_count"] = sha256_file(
                    config.mle_count_config_path
                )
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

    def _run_local_v2(
        self,
        *,
        measurement_log,
        staging: Path,
        target: Path,
        started: str,
    ) -> Path:
        config = self.config
        execution_root = staging / "executions"
        result_root = staging / "results"
        pf_execution = LocalPFCheckpointService().run(
            measurement_log,
            config_path=config.pf_config_path,
            output_dir=result_root / "pf_strict",
            checkpoint_out=result_root / "pf_strict" / "pf_checkpoint.json",
            execution_dir=execution_root / "pf_strict",
            seed=config.random_seed,
            stop_after=measurement_log.record_count,
            checkpoint_in=None,
        )
        spectral_execution = LocalSpectralMLEService().run(
            measurement_log,
            mode="spectral",
            config_path=config.mle_spectral_config_path,
            output_dir=result_root / "mle_spectral",
            execution_dir=execution_root / "mle_spectral",
        )
        executions = {
            "pf_strict": pf_execution,
            "mle_spectral": spectral_execution,
        }
        pf_result = validate_pf_result(
            result_root / "pf_strict",
            expected_variant="pf_strict",
            expected_isotopes=measurement_log.isotopes,
            expected_log_sha256=measurement_log.measurement_log_sha256,
            expected_commit=repository_commit(),
            expected_config_sha256=sha256_file(config.pf_config_path),
            expected_resolved_config_sha256=config.expected_resolved_config_sha256[
                "pf_strict"
            ],
            expected_record_count=measurement_log.record_count,
            expected_step_ids=measurement_log.step_ids,
        )
        spectral_result = validate_mle_result(
            result_root / "mle_spectral",
            expected_mode="spectral",
            expected_isotopes=measurement_log.isotopes,
            expected_log_sha256=measurement_log.measurement_log_sha256,
            expected_commit=repository_commit(),
            expected_config_sha256=sha256_file(config.mle_spectral_config_path),
            expected_resolved_config_sha256=config.expected_resolved_config_sha256[
                "mle_spectral"
            ],
        )
        for name, result in {
            "pf_strict": pf_result,
            "mle_spectral": spectral_result,
        }.items():
            if result.result_sha256 != executions[name].output_sha256:
                raise ContractError(f"{name} changed after its local execution.")
        retargeted = {
            name: self._retarget_execution_paths(execution, staging=staging)
            for name, execution in executions.items()
        }
        metrics = evaluate_benchmark(
            measurement_log=measurement_log,
            truth_path=config.truth_path,
            pf_result=pf_result,
            mle_count_result=None,
            mle_spectral_result=spectral_result,
            executions=retargeted,
        )
        metrics_path = write_json_atomic(staging / "metrics.json", metrics)
        completed = datetime.now(UTC).isoformat()
        manifest = {
            "schema_version": 2,
            "benchmark_id": config.benchmark_id,
            "status": "complete",
            "started_at_utc": started,
            "completed_at_utc": completed,
            "pipeline_order": [
                "validate_measurement_log",
                "pure_pf_replay",
                "spectral_mle_replay",
                "validate_result_contracts",
                "open_evaluation_truth",
                "compute_metrics",
                "write_manifest",
            ],
            "truth_isolation": {
                "truth_path": config.truth_path.as_posix(),
                "truth_sha256": sha256_file(config.truth_path),
                "opened_only_after_all_result_validation": True,
                "passed_to_estimator_commands": False,
            },
            "contracts": {"measurement_log": 2, "pf_result": 1, "mle_result": 1},
            "orchestrator": orchestrator_provenance(self.orchestrator_root),
            "runtime_environment": environment_provenance(),
            "estimator_ownership": {
                "repository": "radiation-estimation-orchestrator",
                "commit": repository_commit(),
                "external_estimator_repositories": [],
            },
            "benchmark_config": {
                "path": config.source_path.as_posix(),
                "sha256": sha256_file(config.source_path),
                "resolved_estimator_config_sha256": dict(
                    config.expected_resolved_config_sha256
                ),
            },
            "measurement_log": {
                "path": measurement_log.root.as_posix(),
                "schema_version": 2,
                "run_id": measurement_log.manifest["run_id"],
                "sha256": measurement_log.measurement_log_sha256,
                "artifact_inventory": dict(measurement_log.artifact_inventory),
            },
            "executions": {
                name: execution.to_dict() for name, execution in sorted(retargeted.items())
            },
            "validated_outputs": {
                "pf_strict": {
                    "sha256": pf_result.result_sha256,
                    "artifact_inventory": dict(pf_result.artifact_inventory),
                },
                "mle_spectral": {
                    "sha256": spectral_result.result_sha256,
                    "artifact_inventory": dict(spectral_result.artifact_inventory),
                },
            },
            "metrics": {"path": metrics_path.name, "sha256": sha256_file(metrics_path)},
        }
        write_manifest_bundle(staging, manifest)
        os.replace(staging, target)
        return target

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
