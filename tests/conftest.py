from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from orchestrator.adapters import MLECLIAdapter, PFCLIAdapter, load_estimator_pins
from orchestrator.adapters.base import AdapterSettings
from orchestrator.adapters.mle_cli import DEFAULT_MLE_COMMAND
from orchestrator.adapters.pf_cli import DEFAULT_PF_COMMAND
from orchestrator.benchmark import BenchmarkConfig, BenchmarkRunner
from orchestrator.contracts import validate_measurement_log, validate_mle_result, validate_pf_result
from orchestrator.evaluation import evaluate_benchmark
from orchestrator.hashing import canonical_json_bytes, load_json, sha256_bytes, sha256_file

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_LOG = ROOT / "fixtures" / "shared_small_run" / "measurement_log"
TRUTH = ROOT / "fixtures" / "shared_small_run" / "evaluation" / "truth.json"
FAKE_ESTIMATOR = ROOT / "tests" / "fakes" / "fake_estimator.py"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _materialize_v2_log(target: Path) -> Path:
    """Convert the v1 contract fixture to raw-spectrum-only MeasurementLog v2."""
    shutil.copytree(MEASUREMENT_LOG, target)
    manifest_path = target / "run_manifest.json"
    forward_path = target / "forward_model_manifest.json"
    observation_path = target / "observations.npz"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_forward = json.loads(forward_path.read_text(encoding="utf-8"))
    with np.load(observation_path, allow_pickle=False) as loaded:
        names = (
            "step_id",
            "action_id",
            "station_id",
            "detector_pose_xyz",
            "detector_quat_wxyz",
            "fe_orientation_index",
            "pb_orientation_index",
            "live_time_s",
            "travel_time_s",
            "shield_actuation_time_s",
            "energy_bin_edges_keV",
            "spectrum_counts",
        )
        arrays = {name: np.array(loaded[name], copy=True) for name in names}
    arrays["spectrum_counts"] = arrays["spectrum_counts"].astype(np.int64)
    with observation_path.open("wb") as handle:
        np.savez(handle, **arrays)

    source_semantics = {
        "quantity": "expected_pre_dead_time_detector_pulse_rate",
        "unit": "cps",
        "normalization_distance_m": 1.0,
    }
    forward = {
        "schema_version": 2,
        "repository_commit": old_manifest["repository_commit"],
        "resolved_config_sha256": old_manifest["resolved_config_sha256"],
        "source_rate_model": "detector_cps_1m",
        "source_rate_semantics": source_semantics,
        "units": old_forward["units"],
        "response_semantics": {
            **old_forward["response_semantics"],
            "observation_distribution": "joint_renewal_total_and_conditional_energy_marks",
        },
        "line_mu_by_isotope": old_forward["line_mu_by_isotope"],
        "model_identifiers": old_manifest["model_identifiers"],
    }
    _write_json(forward_path, forward)
    edges = arrays["energy_bin_edges_keV"]
    manifest = {
        "schema_version": 2,
        "run_id": old_manifest["run_id"],
        "record_count": int(arrays["step_id"].size),
        "repository_commit": old_manifest["repository_commit"],
        "resolved_config_sha256": old_manifest["resolved_config_sha256"],
        "forward_model_manifest_sha256": sha256_file(forward_path),
        "source_rate_model": "detector_cps_1m",
        "source_rate_semantics": source_semantics,
        "isotopes": old_manifest["isotopes"],
        "environment": old_manifest["environment"],
        "obstacle_layout_path": None,
        "source_layout_path": None,
        "sim_backend": old_manifest["sim_backend"],
        "observation_model": "joint_full_spectrum_generative",
        "energy_bin_count": int(arrays["spectrum_counts"].shape[1]),
        "energy_min_keV": float(edges[0]),
        "energy_max_keV": float(edges[-2]),
        "bin_width_keV": float(edges[1] - edges[0]),
        "full_spectrum_contract_hash_sha256": "a" * 64,
        "full_spectrum_contract_schema_version": 3,
        "model_identifiers": old_manifest["model_identifiers"],
        "index_conventions": old_manifest["index_conventions"],
        "artifact_hashes": {},
        "metadata": {"closed_loop": False, "estimator_independent": True},
    }
    manifest["artifact_hashes"] = {
        path.relative_to(target).as_posix(): sha256_file(path)
        for path in target.iterdir()
        if path.is_file() and path.name != "run_manifest.json"
    }
    _write_json(manifest_path, manifest)
    return target


def _run(root: Path, *arguments: str) -> str:
    return subprocess.run(
        arguments,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _create_fake_repository(root: Path, *, kind: str) -> str:
    root.mkdir()
    source = root / "src"
    source.mkdir()
    shutil.copyfile(FAKE_ESTIMATOR, source / "fake_estimator.py")
    if kind == "pf":
        package = source / "pf"
        package.mkdir()
        (package / "__init__.py").write_text("\n", encoding="utf-8")
        (package / "replay.py").write_text(
            """from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import fake_estimator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-log", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    sys.argv = [
        "fake-estimator", "--kind", "pf", "--log", str(args.measurement_log),
        "--config", str(args.config), "--output", str(args.output_dir),
        "--commit", commit, "--profile", args.profile, "--seed", str(args.seed),
    ]
    fake_estimator.main()


if __name__ == "__main__":
    main()
""",
            encoding="utf-8",
        )
        project_name = "fake-pf-estimator"
        scripts = ""
        modules = '["fake_estimator"]'
    else:
        (source / "mle_cli.py").write_text(
            """from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import fake_estimator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("replay", "fit-spectrum"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mle-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
    ).stdout.strip()
    sys.argv = [
        "fake-estimator", "--kind", "mle", "--mode", args.mode,
        "--log", str(args.run_dir), "--config", str(args.mle_config),
        "--output", str(args.output_dir), "--commit", commit,
    ]
    fake_estimator.main()


if __name__ == "__main__":
    main()
""",
            encoding="utf-8",
        )
        project_name = "fake-mle-estimator"
        scripts = '[project.scripts]\nestimate-radiation-mle = "mle_cli:main"\n'
        modules = '["fake_estimator", "mle_cli"]'
    (root / "pyproject.toml").write_text(
        f"""[project]
name = "{project_name}"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = ["numpy>=2.1,<3"]

{scripts}[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {{"" = "src"}}
py-modules = {modules}

[tool.setuptools.packages.find]
where = ["src"]
""",
        encoding="utf-8",
    )
    (root / ".gitignore").write_text(".venv/\n__pycache__/\n*.egg-info/\n", encoding="utf-8")
    _run(root, "git", "init", "-b", "main")
    _run(root, "git", "config", "user.email", "test@example.invalid")
    _run(root, "git", "config", "user.name", "Test")
    _run(root, "uv", "lock")
    _run(root, "git", "add", ".")
    _run(root, "git", "commit", "-m", "contract test estimator")
    return _run(root, "git", "rev-parse", "HEAD")


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def measurement_log_path() -> Path:
    return MEASUREMENT_LOG


@pytest.fixture(scope="session")
def truth_path() -> Path:
    return TRUTH


@pytest.fixture(scope="session")
def benchmark_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    temporary = tmp_path_factory.mktemp("same-log-benchmark")
    pf_repository = temporary / "fake-pf-repository"
    mle_repository = temporary / "fake-mle-repository"
    pf_revision = _create_fake_repository(pf_repository, kind="pf")
    mle_revision = _create_fake_repository(mle_repository, kind="mle")
    output = temporary / "benchmark-output"
    pin_registry = temporary / "PINNED_ESTIMATORS.json"
    pin_registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "particle_filter": {
                    "repository": "fake/particle-filter",
                    "revision": pf_revision,
                    "revision_type": "commit",
                    "release_tag": None,
                    "local_path_hint": str(pf_repository),
                    "expected_measurement_log_schema_version": 1,
                    "expected_result_schema_version": 1,
                },
                "surface_mle": {
                    "repository": "fake/surface-mle",
                    "revision": mle_revision,
                    "revision_type": "commit",
                    "release_tag": None,
                    "local_path_hint": str(mle_repository),
                    "expected_measurement_log_schema_version": 1,
                    "expected_result_schema_version": 1,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pf_config = ROOT / "configs" / "estimators" / "pf_strict_shared_small.json"
    count_config = ROOT / "configs" / "estimators" / "mle_count_shared_small.json"
    spectral_config = ROOT / "configs" / "estimators" / "mle_spectral_shared_small.json"

    def resolved_hash(path: Path) -> str:
        return sha256_bytes(canonical_json_bytes(load_json(path)))

    config_path = temporary / "benchmark.json"
    payload = {
        "schema_version": 1,
        "benchmark_id": "pytest-same-log",
        "measurement_log": str(MEASUREMENT_LOG),
        "truth": str(TRUTH),
        "output_directory": str(output),
        "pin_registry": str(pin_registry),
        "pf_profile": "pf_strict",
        "random_seed": 314159,
        "expected_resolved_estimator_config_sha256": {
            "pf_strict": resolved_hash(pf_config),
            "mle_count": resolved_hash(count_config),
            "mle_spectral": resolved_hash(spectral_config),
        },
        "estimator_configs": {
            "pf": str(pf_config),
            "mle_count": str(count_config),
            "mle_spectral": str(spectral_config),
        },
        "adapters": {
            "particle_filter": {
                "repository_path": str(pf_repository),
                "timeout_s": 60,
                "verify_revision": True,
                "require_clean": True,
                "allowed_dirty_prefixes": [],
            },
            "surface_mle": {
                "repository_path": str(mle_repository),
                "timeout_s": 60,
                "verify_revision": True,
                "require_clean": True,
                "allowed_dirty_prefixes": [],
            },
        },
    }
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BenchmarkRunner(BenchmarkConfig.load(config_path), orchestrator_root=ROOT).run()


@pytest.fixture(scope="session")
def benchmark_v2_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build legacy-shaped v2 result fixtures without using the active benchmark path."""
    temporary = tmp_path_factory.mktemp("same-log-benchmark-v2")
    measurement_log = _materialize_v2_log(temporary / "measurement-log-v2")
    pf_repository = temporary / "fake-pf-repository"
    mle_repository = temporary / "fake-mle-repository"
    pf_revision = _create_fake_repository(pf_repository, kind="pf")
    mle_revision = _create_fake_repository(mle_repository, kind="mle")
    output = temporary / "benchmark-output"
    pin_registry = temporary / "PINNED_ESTIMATORS.json"
    _write_json(
        pin_registry,
        {
            "schema_version": 1,
            "particle_filter": {
                "repository": "fake/particle-filter",
                "revision": pf_revision,
                "revision_type": "commit",
                "release_tag": None,
                "local_path_hint": str(pf_repository),
                "expected_measurement_log_schema_version": 2,
                "expected_result_schema_version": 1,
            },
            "surface_mle": {
                "repository": "fake/surface-mle",
                "revision": mle_revision,
                "revision_type": "commit",
                "release_tag": None,
                "local_path_hint": str(mle_repository),
                "expected_measurement_log_schema_version": 2,
                "expected_result_schema_version": 1,
            },
        },
    )
    pf_config = ROOT / "configs" / "estimators" / "pf_strict_shared_small.json"
    spectral_config = ROOT / "configs" / "estimators" / "mle_spectral_shared_small.json"

    pins = load_estimator_pins(pin_registry)
    log = validate_measurement_log(measurement_log)
    result_root = output / "results"
    execution_root = output / "executions"
    pf_execution = PFCLIAdapter(
        pins["particle_filter"],
        AdapterSettings(
            repository_path=pf_repository,
            command_template=DEFAULT_PF_COMMAND,
            timeout_s=60,
            allowed_dirty_prefixes=(),
        ),
    ).run(
        log,
        config_path=pf_config,
        output_dir=result_root / "pf_strict",
        execution_dir=execution_root / "pf_strict",
        seed=314159,
        profile="pf_strict",
    )
    mle_execution = MLECLIAdapter(
        pins["surface_mle"],
        AdapterSettings(
            repository_path=mle_repository,
            command_template=DEFAULT_MLE_COMMAND,
            timeout_s=60,
            allowed_dirty_prefixes=(),
        ),
    ).run(
        log,
        mode="spectral",
        config_path=spectral_config,
        output_dir=result_root / "mle_spectral",
        execution_dir=execution_root / "mle_spectral",
    )
    pf_result = validate_pf_result(result_root / "pf_strict")
    mle_result = validate_mle_result(result_root / "mle_spectral", expected_mode="spectral")
    metrics = evaluate_benchmark(
        measurement_log=log,
        truth_path=TRUTH,
        pf_result=pf_result,
        mle_count_result=None,
        mle_spectral_result=mle_result,
        executions={"pf_strict": pf_execution, "mle_spectral": mle_execution},
    )
    _write_json(output / "metrics.json", metrics)
    _write_json(
        output / "benchmark_manifest.json",
        {
            "schema_version": 2,
            "measurement_log": {"path": str(measurement_log)},
            "executions": {
                "pf_strict": pf_execution.to_dict(),
                "mle_spectral": mle_execution.to_dict(),
            },
            "validated_outputs": {
                "pf_strict": pf_result.result_sha256,
                "mle_spectral": mle_result.result_sha256,
            },
            "contracts": {"measurement_log": 2},
            "pipeline_order": [
                "validate_measurement_log",
                "pure_pf_replay",
                "spectral_mle_replay",
                "validate_all_results",
                "open_truth_for_evaluation",
            ],
            "truth_isolation": {"opened_only_after_all_result_validation": True},
        },
    )
    return output
