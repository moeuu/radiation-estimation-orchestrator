from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from orchestrator.benchmark import BenchmarkConfig, BenchmarkRunner
from orchestrator.hashing import canonical_json_bytes, load_json, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
MEASUREMENT_LOG = ROOT / "fixtures" / "shared_small_run" / "measurement_log"
TRUTH = ROOT / "fixtures" / "shared_small_run" / "evaluation" / "truth.json"
FAKE_ESTIMATOR = ROOT / "tests" / "fakes" / "fake_estimator.py"


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
