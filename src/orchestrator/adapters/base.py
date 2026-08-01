"""Safe command expansion, revision checks, and subprocess resource accounting."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import psutil

from orchestrator.errors import AdapterExecutionError, ContractError, RevisionError
from orchestrator.hashing import directory_inventory, inventory_digest, load_json, sha256_file

_SAFE_ENV_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "UV_CACHE_DIR",
    "XDG_CACHE_HOME",
    "CUDA_VISIBLE_DEVICES",
)
_PLACEHOLDERS = frozenset(
    {
        "repository",
        "log_dir",
        "config",
        "output_dir",
        "seed",
        "relocation_seed",
        "profile",
        "mode",
        "directive_schedule",
        "initial_estimate",
        "snapshot",
        "snapshot_estimate",
        "planning_request",
        "stop_after",
    }
)
PRODUCTION_ALLOWED_DIRTY_PREFIXES = (
    "results/",
    "logs/",
    "build/",
    ".cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".venv/",
)
_DEFAULT_ALLOWED_DIRTY_PREFIXES = PRODUCTION_ALLOWED_DIRTY_PREFIXES


@dataclass(frozen=True, slots=True)
class EstimatorPin:
    """One immutable external estimator repository revision."""

    name: str
    repository: str
    revision: str
    revision_type: str
    release_tag: str | None
    local_path_hint: str | None
    expected_measurement_log_schema_version: int
    expected_result_schema_version: int


@dataclass(frozen=True, slots=True)
class AdapterSettings:
    """Execution policy for one estimator adapter."""

    repository_path: Path
    command_template: tuple[str, ...]
    timeout_s: float = 3600.0
    verify_revision: bool = True
    require_clean: bool = True
    poll_interval_s: float = 0.02
    allowed_dirty_prefixes: tuple[str, ...] = _DEFAULT_ALLOWED_DIRTY_PREFIXES

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository_path", Path(self.repository_path).resolve())
        object.__setattr__(
            self, "command_template", tuple(str(value) for value in self.command_template)
        )
        normalized_prefixes: list[str] = []
        for value in self.allowed_dirty_prefixes:
            prefix = str(value).replace("\\", "/")
            while prefix.startswith("./"):
                prefix = prefix[2:]
            if not prefix or prefix.startswith(("/", "../")) or "/../" in prefix:
                raise ValueError(f"Invalid allowed dirty prefix: {value!r}")
            normalized_prefixes.append(prefix if prefix.endswith("/") else prefix + "/")
        object.__setattr__(self, "allowed_dirty_prefixes", tuple(sorted(set(normalized_prefixes))))
        if not self.command_template:
            raise ValueError("command_template must not be empty.")
        if self.timeout_s <= 0 or self.poll_interval_s <= 0:
            raise ValueError("timeout and poll interval must be positive.")


@dataclass(frozen=True, slots=True)
class AdapterExecution:
    """Auditable result of exactly one estimator subprocess."""

    estimator: str
    requested_revision: str
    observed_revision: str | None
    repository_path: str
    command: tuple[str, ...]
    started_at_utc: str
    completed_at_utc: str
    exit_code: int
    runtime_s: float
    peak_memory_bytes: int
    timed_out: bool
    measurement_log_sha256: str
    config_sha256: str
    stdout_path: str
    stdout_sha256: str
    stderr_path: str
    stderr_sha256: str
    output_inventory: Mapping[str, str]
    output_sha256: str
    dirty_worktree: Mapping[str, Mapping[str, str | None]]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        payload["output_inventory"] = dict(sorted(self.output_inventory.items()))
        payload["dirty_worktree"] = {
            path: dict(details) for path, details in sorted(self.dirty_worktree.items())
        }
        return payload


def load_estimator_pins(path: str | Path) -> dict[str, EstimatorPin]:
    """Load and strictly validate the pinned estimator registry."""
    payload = load_json(path)
    if payload.get("schema_version") != 1:
        raise ContractError("PINNED_ESTIMATORS.json schema_version must be 1.")
    result: dict[str, EstimatorPin] = {}
    for name in ("particle_filter", "surface_mle"):
        item = payload.get(name)
        if not isinstance(item, dict):
            raise ContractError(f"PINNED_ESTIMATORS.json lacks object {name!r}.")
        try:
            pin = EstimatorPin(
                name=name,
                repository=str(item["repository"]),
                revision=str(item["revision"]),
                revision_type=str(item["revision_type"]),
                release_tag=None if item.get("release_tag") is None else str(item["release_tag"]),
                local_path_hint=(
                    None if item.get("local_path_hint") is None else str(item["local_path_hint"])
                ),
                expected_measurement_log_schema_version=int(
                    item["expected_measurement_log_schema_version"]
                ),
                expected_result_schema_version=int(item["expected_result_schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"Invalid estimator pin {name!r}.") from exc
        if (
            len(pin.revision) not in (40, 64)
            or any(character not in "0123456789abcdef" for character in pin.revision)
            or pin.revision_type != "commit"
        ):
            raise ContractError(f"Estimator {name!r} must pin a full lowercase commit SHA.")
        if pin.release_tag is not None:
            raise ContractError(
                f"Estimator {name!r} declares a release tag but revision_type is commit."
            )
        if pin.expected_measurement_log_schema_version not in {1, 2}:
            raise ContractError(
                f"Estimator {name!r} must target MeasurementLog v1 or v2."
            )
        if pin.expected_result_schema_version != 1:
            raise ContractError(f"Estimator {name!r} must target result schema v1.")
        result[name] = pin
    return result


def _git(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RevisionError(
            f"Could not inspect estimator Git repository {repository}: {exc}"
        ) from exc
    return completed.stdout.strip()


def _dirty_inventory(repository: Path) -> dict[str, dict[str, str | None]]:
    try:
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ),
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RevisionError(f"Could not inspect dirty paths in {repository}: {exc}") from exc
    entries = completed.stdout.decode("utf-8", errors="surrogateescape").split("\x00")
    result: dict[str, dict[str, str | None]] = {}
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4:
            raise RevisionError(f"Could not parse Git status entry: {entry!r}")
        status = entry[:2]
        current_path = entry[3:].replace("\\", "/")
        display_path = current_path
        if any(marker in {"R", "C"} for marker in status) and index < len(entries):
            original_path = entries[index].replace("\\", "/")
            index += 1
            display_path = f"{original_path} -> {current_path}"
        target = repository / current_path
        digest = sha256_file(target) if target.is_file() and not target.is_symlink() else None
        result[display_path] = {"status": status, "sha256": digest}
    return result


def verify_repository_revision(
    repository: str | Path,
    pin: EstimatorPin,
    *,
    require_clean: bool,
    allowed_dirty_prefixes: Sequence[str] = (),
) -> tuple[str, dict[str, dict[str, str | None]]]:
    """Require exact HEAD and reject dirty code while inventorying allowed artifacts."""
    root = Path(repository).resolve()
    if not root.is_dir():
        raise RevisionError(f"Estimator repository does not exist: {root}")
    observed = _git(root, "rev-parse", "HEAD")
    if observed != pin.revision:
        raise RevisionError(
            f"{pin.name} HEAD {observed} does not match pinned revision {pin.revision}."
        )
    committed = _git(root, "rev-parse", f"{pin.revision}^{{commit}}")
    if committed != pin.revision:
        raise RevisionError(f"Pinned revision for {pin.name} is not an exact commit object.")
    dirty = _dirty_inventory(root)
    if require_clean:
        forbidden = [
            path
            for path in dirty
            if not any(
                path.split(" -> ")[-1].startswith(prefix) for prefix in allowed_dirty_prefixes
            )
        ]
        if forbidden:
            raise RevisionError(
                f"{pin.name} has dirty code/config paths outside the explicit allowlist: "
                f"{sorted(forbidden)[:20]}"
            )
    return observed, dirty


def expand_command(template: Sequence[str], values: Mapping[str, object]) -> tuple[str, ...]:
    """Expand whole-token placeholders without invoking a shell."""
    command: list[str] = []
    for token in template:
        rendered = str(token)
        for placeholder in _PLACEHOLDERS:
            marker = "{" + placeholder + "}"
            if marker in rendered:
                if placeholder not in values:
                    raise ContractError(f"Command requires missing placeholder {marker}.")
                rendered = rendered.replace(marker, str(values[placeholder]))
        if "{" in rendered or "}" in rendered:
            raise ContractError(f"Unknown or unbalanced command placeholder in {token!r}.")
        if "\x00" in rendered:
            raise ContractError("Command arguments may not contain NUL bytes.")
        command.append(rendered)
    if not command or not command[0]:
        raise ContractError("Expanded command is empty.")
    return tuple(command)


def _peak_tree_rss(process: psutil.Process) -> int:
    total = 0
    try:
        processes = [process, *process.children(recursive=True)]
    except (psutil.Error, OSError):
        processes = [process]
    for child in processes:
        try:
            total += int(child.memory_info().rss)
        except (psutil.Error, OSError):
            continue
    return total


def _terminate_process_group(process: psutil.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    try:
        process.wait(timeout=3)
    except (psutil.TimeoutExpired, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def run_adapter_process(
    *,
    estimator: str,
    pin: EstimatorPin,
    settings: AdapterSettings,
    command_values: Mapping[str, object],
    measurement_log_sha256: str,
    config_path: Path,
    output_dir: Path,
    execution_dir: Path,
) -> AdapterExecution:
    """Execute one pinned estimator with resource and artifact accounting."""
    if settings.verify_revision:
        observed, dirty_worktree = verify_repository_revision(
            settings.repository_path,
            pin,
            require_clean=settings.require_clean,
            allowed_dirty_prefixes=settings.allowed_dirty_prefixes,
        )
    else:
        observed = None
        dirty_worktree = (
            _dirty_inventory(settings.repository_path)
            if (settings.repository_path / ".git").is_dir()
            else {}
        )
    config_path = config_path.resolve()
    if config_path.is_symlink() or not config_path.is_file():
        raise ContractError(f"Estimator config must be a non-symlink file: {config_path}")
    load_json(config_path)
    config_sha256 = sha256_file(config_path)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Estimator output directory is not empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    execution_dir = execution_dir.resolve()
    execution_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = execution_dir / "stdout.log"
    stderr_path = execution_dir / "stderr.log"
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError(f"Execution logs already exist in {execution_dir}")
    command = expand_command(settings.command_template, command_values)
    environment = {key: os.environ[key] for key in _SAFE_ENV_KEYS if key in os.environ}
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": (settings.repository_path / "src").as_posix(),
            "RSE_TRUTH_ACCESS": "forbidden",
        }
    )
    started = datetime.now(UTC)
    monotonic_start = time.monotonic()
    peak_memory = 0
    timed_out = False
    with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
        try:
            process: psutil.Popen[bytes] = psutil.Popen(
                command,
                cwd=settings.repository_path,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
        except OSError as exc:
            raise AdapterExecutionError(f"Could not start {estimator}: {exc}") from exc
        while process.poll() is None:
            peak_memory = max(peak_memory, _peak_tree_rss(process))
            if time.monotonic() - monotonic_start > settings.timeout_s:
                timed_out = True
                _terminate_process_group(process)
                break
            time.sleep(settings.poll_interval_s)
        exit_code = int(process.wait())
        peak_memory = max(peak_memory, _peak_tree_rss(process))
    completed = datetime.now(UTC)
    runtime = time.monotonic() - monotonic_start
    stdout_hash = sha256_file(stdout_path)
    stderr_hash = sha256_file(stderr_path)
    if timed_out or exit_code != 0:
        try:
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            stderr_tail = "<unreadable>"
        reason = "timed out" if timed_out else f"exited with code {exit_code}"
        raise AdapterExecutionError(f"{estimator} {reason}. stderr tail:\n{stderr_tail}")
    if not output_dir.is_dir():
        raise AdapterExecutionError(f"{estimator} completed without creating {output_dir}")
    inventory = directory_inventory(output_dir)
    if not inventory:
        raise AdapterExecutionError(f"{estimator} produced an empty output directory.")
    return AdapterExecution(
        estimator=estimator,
        requested_revision=pin.revision,
        observed_revision=observed,
        repository_path=settings.repository_path.as_posix(),
        command=command,
        started_at_utc=started.isoformat(),
        completed_at_utc=completed.isoformat(),
        exit_code=exit_code,
        runtime_s=float(runtime),
        peak_memory_bytes=int(peak_memory),
        timed_out=timed_out,
        measurement_log_sha256=measurement_log_sha256,
        config_sha256=config_sha256,
        stdout_path=stdout_path.as_posix(),
        stdout_sha256=stdout_hash,
        stderr_path=stderr_path.as_posix(),
        stderr_sha256=stderr_hash,
        output_inventory=inventory,
        output_sha256=inventory_digest(inventory),
        dirty_worktree=dirty_worktree,
    )


def settings_from_dict(
    payload: Mapping[str, object], *, default_repository: Path, default_command: tuple[str, ...]
) -> AdapterSettings:
    """Resolve a JSON adapter configuration with safe defaults."""
    raw_command = payload.get("command", default_command)
    if not isinstance(raw_command, list | tuple) or not all(
        isinstance(value, str) for value in raw_command
    ):
        raise ContractError("Adapter command must be an array of strings.")
    raw_prefixes = payload.get("allowed_dirty_prefixes", _DEFAULT_ALLOWED_DIRTY_PREFIXES)
    if not isinstance(raw_prefixes, list | tuple) or not all(
        isinstance(value, str) for value in raw_prefixes
    ):
        raise ContractError("allowed_dirty_prefixes must be an array of path prefixes.")
    return AdapterSettings(
        repository_path=Path(str(payload.get("repository_path", default_repository))),
        command_template=tuple(raw_command),
        timeout_s=float(payload.get("timeout_s", 3600.0)),
        verify_revision=bool(payload.get("verify_revision", True)),
        require_clean=bool(payload.get("require_clean", True)),
        poll_interval_s=float(payload.get("poll_interval_s", 0.02)),
        allowed_dirty_prefixes=tuple(raw_prefixes),
    )
