"""Filesystem-aware validation for MeasurementLog, PFResult, and MLEResult v1."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from importlib.resources import files
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator
from numpy.typing import NDArray

from orchestrator.errors import ContractError, TruthIsolationError
from orchestrator.hashing import (
    canonical_json_bytes,
    directory_inventory,
    hash_json_file,
    inventory_digest,
    load_json,
    sha256_bytes,
    sha256_file,
)

SCHEMA_VERSION = 1
MEASUREMENT_REQUIRED_FILES = (
    "run_manifest.json",
    "runtime_config.resolved.json",
    "environment.json",
    "forward_model_manifest.json",
    "observations.npz",
    "observation_metadata.jsonl",
    "repository_commit.txt",
)
PF_REQUIRED_FILES = ("pf_posterior.json", "pf_trace.jsonl", "pf_diagnostics.json")
MLE_REQUIRED_FILES = (
    "mle_estimate.npz",
    "mle_diagnostics.json",
    "hotspot_clusters.json",
)
_REALIZED_SOURCE_NAMES = frozenset(
    {
        "sourcelayout",
        "sourcelayoutpath",
        "sourcepositions",
        "pointsources",
        "sources",
        "sourcelist",
    }
)
_REALIZED_SOURCE_FRAGMENTS = ("sourcelayout", "sourcepositions", "pointsources")

OBSERVATION_ARRAYS = frozenset(
    {
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
        "spectrum_variance",
        "spectrum_variance_present",
        "isotope_counts",
        "isotope_counts_present",
        "isotope_counts_record_present",
        "isotope_count_covariance",
        "isotope_count_covariance_present",
        "isotope_count_covariance_record_present",
    }
)


@dataclass(frozen=True, slots=True)
class MeasurementLogInfo:
    """Validated estimator-safe MeasurementLog metadata and read-only arrays."""

    root: Path
    manifest: MappingProxyType[str, object]
    forward_model_manifest: MappingProxyType[str, object]
    arrays: MappingProxyType[str, NDArray[Any]]
    artifact_inventory: MappingProxyType[str, str]
    measurement_log_sha256: str

    @property
    def record_count(self) -> int:
        return int(self.manifest["record_count"])

    @property
    def isotopes(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.manifest["isotopes"])  # type: ignore[arg-type]

    @property
    def step_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.arrays["step_id"])

    @property
    def station_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.arrays["station_id"])


@dataclass(frozen=True, slots=True)
class PFResultInfo:
    """Validated pure-PF result bundle."""

    root: Path
    posterior: MappingProxyType[str, object]
    diagnostics: MappingProxyType[str, object]
    trace: tuple[MappingProxyType[str, object], ...]
    artifact_inventory: MappingProxyType[str, str]
    result_sha256: str


@dataclass(frozen=True, slots=True)
class MLEResultInfo:
    """Validated standalone surface-MLE result bundle."""

    root: Path
    diagnostics: MappingProxyType[str, object]
    hotspot_clusters: tuple[MappingProxyType[str, object], ...]
    arrays: MappingProxyType[str, NDArray[Any]]
    artifact_inventory: MappingProxyType[str, str]
    result_sha256: str

    @property
    def mode(self) -> str:
        nested = self.diagnostics["diagnostics"]
        assert isinstance(nested, dict)
        return str(nested["mode"])


def _schema(name: str) -> dict[str, object]:
    resource = files("orchestrator.contracts").joinpath(name)
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Packaged schema {name} is unreadable.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Packaged schema {name} is not an object.")
    Draft202012Validator.check_schema(payload)
    return payload


def _validate_schema(payload: object, schema_name: str, *, label: str) -> None:
    validator = Draft202012Validator(_schema(schema_name))
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.absolute_path))
    if not errors:
        return
    details: list[str] = []
    for error in errors[:8]:
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path
        )
        details.append(f"{path}: {error.message}")
    remainder = f" (+{len(errors) - 8} more)" if len(errors) > 8 else ""
    raise ContractError(f"{label} violates {schema_name}: {'; '.join(details)}{remainder}")


def _require_directory(root: str | Path, names: tuple[str, ...], *, label: str) -> Path:
    supplied = Path(root)
    if supplied.is_symlink():
        raise ContractError(f"{label} must not be a symlink: {supplied}")
    directory = supplied.resolve()
    if not directory.is_dir():
        raise ContractError(f"{label} must be a non-symlink directory: {directory}")
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise ContractError(f"{label} is missing required files: {missing}")
    for name in names:
        if (directory / name).is_symlink():
            raise ContractError(f"{label} file may not be a symlink: {name}")
    return directory


def _normalized_name(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _is_model_source_semantics(name: str) -> bool:
    return name.startswith(("sourcerate", "sourceextent"))


def _indicates_realized_truth(name: str, *, key: bool) -> bool:
    if "truth" in name or any(fragment in name for fragment in _REALIZED_SOURCE_FRAGMENTS):
        return True
    if _is_model_source_semantics(name):
        return False
    return key and name in _REALIZED_SOURCE_NAMES


def _reject_realized_truth(payload: object, *, label: str, location: str = "$") -> None:
    """Recursively reject realized source/truth pointers while allowing model semantics."""
    if isinstance(payload, dict):
        for raw_key, value in payload.items():
            key = _normalized_name(raw_key)
            child_location = f"{location}.{raw_key}"
            if _indicates_realized_truth(key, key=True):
                raise TruthIsolationError(
                    f"{label} may not expose realized truth at {child_location}."
                )
            _reject_realized_truth(value, label=label, location=child_location)
        return
    if isinstance(payload, list | tuple):
        for index, value in enumerate(payload):
            _reject_realized_truth(value, label=label, location=f"{location}[{index}]")
        return
    if isinstance(payload, str):
        value = _normalized_name(payload)
        if _indicates_realized_truth(value, key=False):
            raise TruthIsolationError(
                f"{label} may not expose a realized-truth value at {location}."
            )


def _reject_truth(root: Path) -> None:
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        for component in relative.parts:
            names = {_normalized_name(component), _normalized_name(Path(component).stem)}
            if any(_indicates_realized_truth(name, key=True) for name in names):
                raise TruthIsolationError(
                    "Estimator input/result bundle artifact path indicates evaluation truth: "
                    f"{relative.as_posix()}"
                )


def _load_npz(path: Path) -> dict[str, NDArray[Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            result = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ContractError(f"Could not read safe NPZ archive {path}: {exc}") from exc
    for name, array in result.items():
        if array.dtype.hasobject:
            raise ContractError(f"NPZ array {name!r} may not use object dtype.")
        array.setflags(write=False)
    return result


def _array(
    arrays: dict[str, NDArray[Any]],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[np.generic],
) -> NDArray[Any]:
    if name not in arrays:
        raise ContractError(f"observations.npz is missing array {name!r}.")
    value = arrays[name]
    if value.shape != shape:
        raise ContractError(f"Array {name!r} has shape {value.shape}; expected {shape}.")
    expected = np.dtype(dtype)
    if value.dtype != expected:
        raise ContractError(f"Array {name!r} has dtype {value.dtype}; expected {expected}.")
    return value


def _finite(name: str, array: NDArray[Any]) -> None:
    if not np.all(np.isfinite(array)):
        raise ContractError(f"Array {name!r} contains non-finite values.")


def _check_masked_values(
    values: NDArray[Any], mask: NDArray[np.bool_], *, name: str, nonnegative: bool = False
) -> None:
    if not np.all(np.isfinite(values[mask])):
        raise ContractError(f"Present entries in {name!r} must be finite.")
    if nonnegative and np.any(values[mask] < 0):
        raise ContractError(f"Present entries in {name!r} must be nonnegative.")
    if np.any(np.isfinite(values[~mask])):
        raise ContractError(f"Absent entries in {name!r} must use NaN sentinels.")


def _validate_forward_manifest(payload: dict[str, object], run_manifest: dict[str, object]) -> None:
    _validate_schema(
        payload,
        "forward_model_manifest_schema.json",
        label="forward_model_manifest.json",
    )
    for key in (
        "repository_commit",
        "resolved_config_sha256",
        "source_rate_model",
        "source_rate_semantics",
    ):
        if payload.get(key) != run_manifest.get(key):
            raise ContractError(f"Forward-model and run manifest field {key!r} differ.")
    identifiers = payload.get("model_identifiers")
    if identifiers != run_manifest.get("model_identifiers"):
        raise ContractError("Forward-model and run-manifest model identifiers differ.")
    units = payload.get("units")
    required_units = {
        "distance": "m",
        "time": "s",
        "energy": "keV",
        "source_strength": "detector_cps_1m",
        "linear_attenuation": "cm^-1",
    }
    if not isinstance(units, dict) or any(
        units.get(key) != value for key, value in required_units.items()
    ):
        raise ContractError(
            f"Forward-model units must include exact canonical units: {required_units}"
        )
    semantics = payload.get("response_semantics")
    if not isinstance(semantics, dict):
        raise ContractError("forward_model_manifest.json requires response_semantics.")
    for key in (
        "distance_attenuation",
        "detector_geometry",
        "shield_attenuation",
        "obstacle_attenuation",
        "live_time_scaling",
        "line_resolved_response",
    ):
        if not isinstance(semantics.get(key), str) or not semantics[key]:
            raise ContractError(f"Forward-model response_semantics requires {key!r}.")

    line_table = payload["line_mu_by_isotope"]
    assert isinstance(line_table, dict)
    isotope_names = run_manifest.get("isotopes")
    assert isinstance(isotope_names, list)
    if set(line_table) != set(isotope_names):
        raise ContractError(
            "Forward-model line_mu_by_isotope must contain exactly the manifest isotopes."
        )
    spectrum_table: dict[str, list[dict[str, float]]] = {}
    for isotope in isotope_names:
        rows = line_table[isotope]
        assert isinstance(rows, list)
        energies: list[float] = []
        weights: list[float] = []
        spectrum_rows: list[dict[str, float]] = []
        for row in rows:
            assert isinstance(row, dict)
            energy = float(row["energy_keV"])
            weight = float(row["weight"])
            energies.append(energy)
            weights.append(weight)
            spectrum_rows.append({"energy_keV": energy, "weight": weight})
        if any(right <= left for left, right in pairwise(energies)):
            raise ContractError(f"Forward-model lines for {isotope} must be energy-sorted.")
        if not np.isclose(sum(weights), 1.0, rtol=1e-12, atol=1e-15):
            raise ContractError(f"Forward-model line weights for {isotope} must sum to one.")
        spectrum_table[str(isotope)] = spectrum_rows

    assert isinstance(identifiers, dict)
    shield_identifier = identifiers["shield"]
    spectrum_identifier = identifiers["spectrum"]
    assert isinstance(shield_identifier, dict)
    assert isinstance(spectrum_identifier, dict)
    shield_hash = sha256_bytes(canonical_json_bytes(line_table))
    spectrum_hash = sha256_bytes(canonical_json_bytes(spectrum_table))
    if shield_identifier.get("sha256") != shield_hash:
        raise ContractError(
            "Forward-model shield hash must bind the full line-resolved Fe/Pb table."
        )
    if spectrum_identifier.get("sha256") != spectrum_hash:
        raise ContractError(
            "Forward-model spectrum hash must bind line energies and normalized weights."
        )


def _validate_measurement_arrays(
    arrays: dict[str, NDArray[Any]], *, records: int, bins: int, isotopes: int
) -> None:
    missing = sorted(OBSERVATION_ARRAYS - arrays.keys())
    if missing:
        raise ContractError(f"observations.npz is missing v1 arrays: {missing}")
    if any(_indicates_realized_truth(_normalized_name(name), key=True) for name in arrays):
        raise TruthIsolationError("observations.npz may not embed realized-truth arrays.")
    steps = _array(arrays, "step_id", shape=(records,), dtype=np.int64)
    actions = _array(arrays, "action_id", shape=(records,), dtype=np.int64)
    stations = _array(arrays, "station_id", shape=(records,), dtype=np.int64)
    if np.any(steps < 0) or np.any(np.diff(steps) <= 0):
        raise ContractError("step_id must be nonnegative and strictly increasing in causal order.")
    if np.any(actions < 0) or np.unique(actions).size != records:
        raise ContractError("action_id must be nonnegative and unique per record.")
    if np.any(stations < 0) or np.any(np.diff(stations) < 0):
        raise ContractError("station_id must be nonnegative and nondecreasing.")

    for name, shape in (
        ("detector_pose_xyz", (records, 3)),
        ("detector_quat_wxyz", (records, 4)),
        ("live_time_s", (records,)),
        ("travel_time_s", (records,)),
        ("shield_actuation_time_s", (records,)),
        ("spectrum_counts", (records, bins)),
    ):
        value = _array(arrays, name, shape=shape, dtype=np.float64)
        _finite(name, value)
    quaternion = arrays["detector_quat_wxyz"]
    if not np.allclose(np.linalg.norm(quaternion, axis=1), 1.0, rtol=1e-9, atol=1e-12):
        raise ContractError("detector_quat_wxyz rows must be unit quaternions.")
    for name in ("live_time_s", "travel_time_s", "shield_actuation_time_s", "spectrum_counts"):
        if np.any(arrays[name] < 0):
            raise ContractError(f"Array {name!r} must be nonnegative.")
    if np.any(arrays["live_time_s"] <= 0):
        raise ContractError("live_time_s must be strictly positive.")

    for name in ("fe_orientation_index", "pb_orientation_index"):
        value = _array(arrays, name, shape=(records,), dtype=np.int64)
        if np.any(value < 0) or np.any(value > 7):
            raise ContractError(f"{name} must lie in [0, 7] for the 64-pair program.")
    edges = _array(arrays, "energy_bin_edges_keV", shape=(bins + 1,), dtype=np.float64)
    _finite("energy_bin_edges_keV", edges)
    if np.any(np.diff(edges) <= 0):
        raise ContractError("energy_bin_edges_keV must be strictly increasing.")

    variance = _array(arrays, "spectrum_variance", shape=(records, bins), dtype=np.float64)
    variance_record = _array(arrays, "spectrum_variance_present", shape=(records,), dtype=np.bool_)
    variance_mask = np.broadcast_to(variance_record[:, None], variance.shape)
    _check_masked_values(variance, variance_mask, name="spectrum_variance", nonnegative=True)

    counts = _array(arrays, "isotope_counts", shape=(records, isotopes), dtype=np.float64)
    counts_mask = _array(
        arrays, "isotope_counts_present", shape=(records, isotopes), dtype=np.bool_
    )
    counts_record = _array(
        arrays, "isotope_counts_record_present", shape=(records,), dtype=np.bool_
    )
    if np.any(counts_mask & ~counts_record[:, None]):
        raise ContractError("isotope_counts presence masks are inconsistent.")
    _check_masked_values(counts, counts_mask, name="isotope_counts", nonnegative=True)

    covariance = _array(
        arrays,
        "isotope_count_covariance",
        shape=(records, isotopes, isotopes),
        dtype=np.float64,
    )
    covariance_mask = _array(
        arrays,
        "isotope_count_covariance_present",
        shape=(records, isotopes, isotopes),
        dtype=np.bool_,
    )
    covariance_record = _array(
        arrays,
        "isotope_count_covariance_record_present",
        shape=(records,),
        dtype=np.bool_,
    )
    if np.any(covariance_mask & ~covariance_record[:, None, None]):
        raise ContractError("isotope covariance presence masks are inconsistent.")
    _check_masked_values(covariance, covariance_mask, name="isotope_count_covariance")
    for row in range(records):
        if bool(covariance_record[row]):
            if not np.all(covariance_mask[row]):
                raise ContractError("A present covariance record must contain a full matrix.")
            matrix = covariance[row]
            if not np.allclose(matrix, matrix.T, rtol=1e-9, atol=1e-12):
                raise ContractError("Isotope count covariance must be symmetric.")
            if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-9:
                raise ContractError("Isotope count covariance must be positive semidefinite.")


def validate_measurement_log(root: str | Path) -> MeasurementLogInfo:
    """Validate a canonical truth-free MeasurementLog v1 bundle."""
    directory = _require_directory(root, MEASUREMENT_REQUIRED_FILES, label="MeasurementLog")
    _reject_truth(directory)
    manifest = load_json(directory / "run_manifest.json")
    _validate_schema(manifest, "measurement_log_schema.json", label="run_manifest.json")
    config = load_json(directory / "runtime_config.resolved.json")
    environment = load_json(directory / "environment.json")
    forward = load_json(directory / "forward_model_manifest.json")
    if manifest["source_layout_path"] is not None:
        raise TruthIsolationError("run_manifest.source_layout_path must be null.")
    _reject_realized_truth(manifest.get("metadata", {}), label="run manifest metadata")
    _reject_realized_truth(manifest["environment"], label="run manifest environment")
    _reject_realized_truth(environment, label="environment.json")
    _reject_realized_truth(config, label="runtime_config.resolved.json")
    if manifest["environment"] != environment:
        raise ContractError("environment.json must exactly match run_manifest.environment.")
    config_hash = hash_json_file(directory / "runtime_config.resolved.json")
    if config_hash != manifest["resolved_config_sha256"]:
        raise ContractError("resolved_config_sha256 does not match runtime_config.resolved.json.")
    if "runtime_config_sha256" in manifest and manifest["runtime_config_sha256"] != config_hash:
        raise ContractError("Legacy runtime_config_sha256 alias disagrees with canonical hash.")
    forward_hash = sha256_file(directory / "forward_model_manifest.json")
    if forward_hash != manifest["forward_model_manifest_sha256"]:
        raise ContractError("forward_model_manifest_sha256 does not match its artifact.")
    _validate_forward_manifest(forward, manifest)

    declared_hashes = manifest["artifact_hashes"]
    assert isinstance(declared_hashes, dict)
    required_hashed = set(MEASUREMENT_REQUIRED_FILES) - {"run_manifest.json"}
    for name in required_hashed:
        expected = declared_hashes.get(name)
        if expected != sha256_file(directory / name):
            raise ContractError(f"artifact_hashes[{name!r}] does not match file content.")
    unknown_declared = set(declared_hashes) - {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }
    if unknown_declared:
        raise ContractError(
            f"artifact_hashes declares missing artifacts: {sorted(unknown_declared)}"
        )

    records = int(manifest["record_count"])
    bins = int(manifest["energy_bin_count"])
    isotope_names = manifest["isotopes"]
    assert isinstance(isotope_names, list)
    arrays = _load_npz(directory / "observations.npz")
    _validate_measurement_arrays(arrays, records=records, bins=bins, isotopes=len(isotope_names))

    try:
        lines = (directory / "observation_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractError("Could not read observation_metadata.jsonl.") from exc
    if len(lines) != records:
        raise ContractError("observation_metadata.jsonl line count must equal record_count.")
    for index, line in enumerate(lines):
        try:
            payload = json.loads(line, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise ContractError(f"Invalid metadata JSON on line {index + 1}.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
            raise ContractError(f"Metadata line {index + 1} must contain a metadata object.")
        for field in ("step_id", "action_id", "station_id"):
            if payload.get(field) != int(arrays[field][index]):
                raise ContractError(f"Metadata line {index + 1} {field} disagrees with NPZ.")
        if payload.get("run_id") != manifest["run_id"]:
            raise ContractError(f"Metadata line {index + 1} run_id disagrees with manifest.")
        if payload.get("array_index") != index:
            raise ContractError(f"Metadata line {index + 1} array_index must equal row index.")
        _reject_realized_truth(payload, label=f"observation metadata line {index + 1}")

    canonical_commit = (directory / "repository_commit.txt").read_text(encoding="utf-8").strip()
    if canonical_commit != manifest["repository_commit"]:
        raise ContractError("repository_commit.txt disagrees with repository_commit.")
    optional_commit = directory / "upstream_pf_commit.txt"
    if optional_commit.exists():
        if optional_commit.is_symlink():
            raise ContractError("upstream_pf_commit.txt may not be a symlink.")
        value = optional_commit.read_text(encoding="utf-8").strip()
        if value != manifest["repository_commit"]:
            raise ContractError("upstream_pf_commit.txt disagrees with repository_commit.")
    del config
    inventory = directory_inventory(directory)
    return MeasurementLogInfo(
        root=directory,
        manifest=MappingProxyType(manifest),
        forward_model_manifest=MappingProxyType(forward),
        arrays=MappingProxyType(arrays),
        artifact_inventory=MappingProxyType(inventory),
        measurement_log_sha256=inventory_digest(inventory),
    )


def _reject_json_constant(value: str) -> None:
    raise ContractError(f"Non-finite JSON constant is forbidden: {value}")


def _jsonl(path: Path, *, label: str) -> tuple[dict[str, object], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"Could not read {label}.") from exc
    payloads: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            raise ContractError(f"{label} contains a blank line at {index + 1}.")
        try:
            payload = json.loads(line, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{label} line {index + 1} is invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise ContractError(f"{label} line {index + 1} must contain an object.")
        payloads.append(payload)
    return tuple(payloads)


def _provenance(payload: dict[str, object]) -> dict[str, object]:
    direct = payload.get("provenance")
    if isinstance(direct, dict):
        return direct
    nested = payload.get("diagnostics")
    if isinstance(nested, dict) and isinstance(nested.get("provenance"), dict):
        return nested["provenance"]  # type: ignore[return-value]
    raise ContractError("Result lacks required estimator provenance.")


def validate_pf_result(
    root: str | Path,
    *,
    expected_variant: str | None = None,
    expected_isotopes: tuple[str, ...] | None = None,
    expected_log_sha256: str | None = None,
    expected_commit: str | None = None,
    expected_config_sha256: str | None = None,
    expected_resolved_config_sha256: str | None = None,
    expected_record_count: int | None = None,
    expected_step_ids: tuple[int, ...] | None = None,
) -> PFResultInfo:
    """Validate a pure-PF posterior bundle and its causal trace."""
    directory = _require_directory(root, PF_REQUIRED_FILES, label="PFResult")
    _reject_truth(directory)
    posterior = load_json(directory / "pf_posterior.json")
    diagnostics = load_json(directory / "pf_diagnostics.json")
    _validate_schema(posterior, "pf_result_schema.json", label="pf_posterior.json")
    provenance = _provenance(posterior)
    observed_variant = str(posterior["estimator_variant"])
    if expected_variant is not None and observed_variant != expected_variant:
        raise ContractError(
            f"PF estimator variant {observed_variant!r} does not match requested "
            f"variant {expected_variant!r}."
        )
    provenance_variant = provenance.get("estimator_variant")
    if provenance_variant is not None and provenance_variant != observed_variant:
        raise ContractError("PF provenance estimator_variant differs from the posterior.")
    if (
        expected_log_sha256 is not None
        and provenance["measurement_log_sha256"] != expected_log_sha256
    ):
        raise ContractError("PF provenance measurement-log hash does not match its input.")
    if expected_commit is not None and provenance["estimator_commit"] != expected_commit:
        raise ContractError("PF provenance commit does not match the pinned revision.")
    if expected_config_sha256 is not None and provenance["config_sha256"] != expected_config_sha256:
        raise ContractError("PF provenance config hash does not match its input file.")
    if (
        expected_resolved_config_sha256 is not None
        and provenance["resolved_config_sha256"] != expected_resolved_config_sha256
    ):
        raise ContractError("PF provenance resolved-config hash does not match its artifact.")
    isotopes = posterior["isotopes"]
    assert isinstance(isotopes, dict)
    if expected_isotopes is not None and set(isotopes) != set(expected_isotopes):
        raise ContractError("PF posterior isotope keys do not match MeasurementLog isotopes.")
    for isotope, estimate in isotopes.items():
        assert isinstance(estimate, dict)
        distribution = estimate["cardinality_distribution"]
        assert isinstance(distribution, dict)
        total = sum(float(value) for value in distribution.values())
        if not np.isclose(total, 1.0, rtol=1e-9, atol=1e-12):
            raise ContractError(f"PF cardinality distribution for {isotope} sums to {total}.")
        maximum_probability = max(float(value) for value in distribution.values())
        deterministic_map = min(
            int(cardinality)
            for cardinality, probability in distribution.items()
            if float(probability) == maximum_probability
        )
        if int(estimate["map_cardinality"]) != deterministic_map:
            raise ContractError(
                f"PF map_cardinality for {isotope} must be the probability argmax "
                "with smallest-cardinality tie breaking."
            )
        modes = estimate.get("modes", estimate.get("sources"))
        assert isinstance(modes, list)
        if len(modes) != int(estimate["map_cardinality"]):
            raise ContractError(f"PF mode count for {isotope} must equal map_cardinality.")
        for mode in modes:
            if not isinstance(mode, dict):
                continue
            covariance = mode.get("position_covariance_xyz")
            if covariance is not None:
                matrix = np.asarray(covariance, dtype=float)
                if not np.allclose(matrix, matrix.T, rtol=1e-9, atol=1e-12):
                    raise ContractError(f"PF position covariance for {isotope} is not symmetric.")
                if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-9:
                    raise ContractError(f"PF position covariance for {isotope} is not PSD.")

    trace_payloads = _jsonl(directory / "pf_trace.jsonl", label="pf_trace.jsonl")
    if not trace_payloads:
        raise ContractError("pf_trace.jsonl must contain at least one causal snapshot.")
    trace_steps: list[int] = []
    for index, payload in enumerate(trace_payloads):
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ContractError(f"PF trace line {index + 1} has unsupported schema_version.")
        if payload.get("estimator_family") != "particle_filter":
            raise ContractError(f"PF trace line {index + 1} has invalid estimator_family.")
        step = payload.get("step_id")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ContractError(f"PF trace line {index + 1} requires nonnegative step_id.")
        trace_steps.append(step)
    if any(right <= left for left, right in pairwise(trace_steps)):
        raise ContractError("PF trace step IDs must be strictly increasing.")
    if expected_step_ids is not None and tuple(trace_steps) != expected_step_ids:
        raise ContractError("PF trace step IDs do not exactly match MeasurementLog causal steps.")
    if expected_record_count is not None and len(trace_payloads) != expected_record_count:
        raise ContractError("PF trace length does not match MeasurementLog record_count.")
    if diagnostics.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("pf_diagnostics.json schema_version must be 1.")
    if (
        expected_record_count is not None
        and diagnostics.get("record_count") != expected_record_count
    ):
        raise ContractError("PF diagnostic record_count disagrees with MeasurementLog.")
    for forbidden in ("truth", "truth_sources", "ground_truth"):
        if forbidden in diagnostics:
            raise TruthIsolationError(f"PF diagnostics may not include {forbidden!r}.")
    inventory = directory_inventory(directory)
    return PFResultInfo(
        root=directory,
        posterior=MappingProxyType(posterior),
        diagnostics=MappingProxyType(diagnostics),
        trace=tuple(MappingProxyType(value) for value in trace_payloads),
        artifact_inventory=MappingProxyType(inventory),
        result_sha256=inventory_digest(inventory),
    )


def validate_mle_result(
    root: str | Path,
    *,
    expected_mode: str | None = None,
    expected_isotopes: tuple[str, ...] | None = None,
    expected_log_sha256: str | None = None,
    expected_commit: str | None = None,
    expected_config_sha256: str | None = None,
    expected_resolved_config_sha256: str | None = None,
) -> MLEResultInfo:
    """Validate one count- or spectral-domain standalone MLE bundle."""
    directory = _require_directory(root, MLE_REQUIRED_FILES, label="MLEResult")
    _reject_truth(directory)
    diagnostics = load_json(directory / "mle_diagnostics.json")
    hotspots_payload = load_json(directory / "hotspot_clusters.json")
    _validate_schema(diagnostics, "mle_result_schema.json", label="mle_diagnostics.json")
    nested = diagnostics["diagnostics"]
    assert isinstance(nested, dict)
    mode = str(nested["mode"])
    if expected_mode is not None and mode != expected_mode:
        raise ContractError(f"MLE mode {mode!r} does not match requested mode {expected_mode!r}.")
    clusters = hotspots_payload.get("hotspot_clusters")
    if hotspots_payload.get("schema_version") != SCHEMA_VERSION or not isinstance(clusters, list):
        raise ContractError("hotspot_clusters.json must be a v1 cluster list.")
    if clusters != nested["hotspot_clusters"]:
        raise ContractError("hotspot_clusters.json does not mirror MLE diagnostics.")
    arrays = _load_npz(directory / "mle_estimate.npz")
    for name in (
        "schema_version",
        "diagnostics_sha256",
        "isotope_names",
        "patch_ids",
        "patch_centroids_xyz",
        "patch_surface_kinds",
        "patch_strength_by_isotope",
        "objective_value",
        "poisson_deviance",
        "iterations",
        "converged",
        "patch_count",
    ):
        if name not in arrays:
            raise ContractError(f"mle_estimate.npz is missing {name!r}.")
    if int(np.asarray(arrays["schema_version"]).item()) != SCHEMA_VERSION:
        raise ContractError("mle_estimate.npz schema_version must be 1.")
    if str(np.asarray(arrays["diagnostics_sha256"]).item()) != sha256_file(
        directory / "mle_diagnostics.json"
    ):
        raise ContractError("MLE diagnostics hash embedded in NPZ does not match.")
    isotope_names = tuple(str(value) for value in arrays["isotope_names"].tolist())
    if isotope_names != tuple(str(value) for value in diagnostics["isotope_names"]):  # type: ignore[arg-type]
        raise ContractError("MLE NPZ isotope order differs from diagnostics.")
    if expected_isotopes is not None and isotope_names != expected_isotopes:
        raise ContractError("MLE isotope order does not exactly match MeasurementLog isotopes.")
    patch_count = int(diagnostics["patch_count"])
    patch_count_array = np.asarray(arrays["patch_count"])
    if (
        patch_count_array.shape != ()
        or patch_count_array.dtype != np.dtype(np.int64)
        or int(patch_count_array.item()) != patch_count
    ):
        raise ContractError("MLE NPZ patch_count differs from diagnostics.")
    for field in ("objective_value", "poisson_deviance"):
        scalar = np.asarray(arrays[field])
        if scalar.shape != () or scalar.dtype != np.dtype(np.float64):
            raise ContractError(f"MLE NPZ {field} must be a scalar float64.")
        value = float(scalar.item())
        if not np.isfinite(value) or value != float(diagnostics[field]):
            raise ContractError(f"MLE NPZ {field} differs from diagnostics.")
    iterations = np.asarray(arrays["iterations"])
    if (
        iterations.shape != ()
        or iterations.dtype != np.dtype(np.int64)
        or int(iterations.item()) != int(diagnostics["iterations"])
    ):
        raise ContractError("MLE NPZ iterations differs from diagnostics.")
    converged = np.asarray(arrays["converged"])
    if converged.shape != () or converged.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
        raise ContractError("MLE NPZ converged must be a scalar bool or uint8.")
    converged_value = int(converged.item())
    if converged_value not in (0, 1) or bool(converged_value) is not diagnostics["converged"]:
        raise ContractError("MLE NPZ converged differs from diagnostics.")

    patch_ids = np.asarray(arrays["patch_ids"])
    if patch_ids.shape != (patch_count,) or patch_ids.dtype != np.dtype(np.int64):
        raise ContractError("MLE patch_ids must be an int64 vector of patch_count entries.")
    if np.any(patch_ids < 0) or np.unique(patch_ids).size != patch_count:
        raise ContractError("MLE patch_ids must be nonnegative and unique.")
    patch_id_to_index = {int(patch_id): index for index, patch_id in enumerate(patch_ids)}
    centroids = arrays["patch_centroids_xyz"]
    strengths = arrays["patch_strength_by_isotope"]
    surface_kinds = arrays["patch_surface_kinds"]
    if centroids.shape != (patch_count, 3) or centroids.dtype != np.dtype(np.float64):
        raise ContractError("MLE patch_centroids_xyz shape is invalid.")
    if not np.all(np.isfinite(centroids)):
        raise ContractError("MLE patch centroids must be finite.")
    if strengths.shape != (len(isotope_names), patch_count) or strengths.dtype != np.dtype(
        np.float64
    ):
        raise ContractError("MLE patch_strength_by_isotope shape is invalid.")
    if not np.all(np.isfinite(strengths)) or np.any(strengths < 0):
        raise ContractError("MLE patch strengths must be finite and nonnegative.")
    if surface_kinds.shape != (patch_count,) or surface_kinds.dtype.kind not in {"U", "S"}:
        raise ContractError(
            "MLE patch_surface_kinds must be a string vector of patch_count entries."
        )
    if any(not str(value) for value in surface_kinds.tolist()):
        raise ContractError("MLE patch_surface_kinds entries must be nonempty.")

    seen_cluster_ids: set[int] = set()
    claimed_patches: set[tuple[str, int]] = set()
    for cluster in clusters:
        assert isinstance(cluster, dict)
        isotope = str(cluster["isotope"])
        if isotope not in isotope_names:
            raise ContractError("MLE cluster isotope is absent from the result isotope order.")
        cluster_id = int(cluster["cluster_id"])
        if cluster_id in seen_cluster_ids:
            raise ContractError("MLE cluster IDs must be unique.")
        seen_cluster_ids.add(cluster_id)
        centroid = np.asarray(cluster["centroid_xyz"], dtype=float)
        cluster_strength = float(cluster["integrated_strength_cps_1m"])
        if not np.all(np.isfinite(centroid)) or not np.isfinite(cluster_strength):
            raise ContractError("MLE cluster centroids and strengths must be finite.")
        peak_density = cluster.get("peak_density_cps_1m_m2")
        if peak_density is not None and not np.isfinite(float(peak_density)):
            raise ContractError("MLE cluster peak density must be finite.")
        raw_cluster_patch_ids = cluster.get("patch_ids")
        if not isinstance(raw_cluster_patch_ids, list) or not raw_cluster_patch_ids:
            raise ContractError("Every MLE cluster must identify at least one patch.")
        cluster_patch_ids = [int(value) for value in raw_cluster_patch_ids]
        if len(set(cluster_patch_ids)) != len(cluster_patch_ids):
            raise ContractError("MLE cluster patch_ids must be unique within a cluster.")
        if any(patch_id not in patch_id_to_index for patch_id in cluster_patch_ids):
            raise ContractError("MLE cluster references an unknown patch ID.")
        claims = {(isotope, patch_id) for patch_id in cluster_patch_ids}
        if claimed_patches & claims:
            raise ContractError("MLE clusters for one isotope may not overlap patch IDs.")
        claimed_patches.update(claims)
        patch_indices = [patch_id_to_index[patch_id] for patch_id in cluster_patch_ids]
        isotope_index = isotope_names.index(isotope)
        patch_strengths = strengths[isotope_index, patch_indices]
        expected_strength = float(np.sum(patch_strengths))
        if not np.isclose(cluster_strength, expected_strength, rtol=1e-9, atol=1e-9):
            raise ContractError("MLE cluster strength is inconsistent with its patches.")
        patch_centroids = centroids[patch_indices]
        expected_centroid = (
            np.sum(patch_centroids * patch_strengths[:, None], axis=0) / expected_strength
            if expected_strength > 0
            else np.mean(patch_centroids, axis=0)
        )
        if not np.allclose(centroid, expected_centroid, rtol=1e-9, atol=1e-9):
            raise ContractError("MLE cluster centroid is inconsistent with its patches.")
        expected_surface_kinds = {str(surface_kinds[index]) for index in patch_indices}
        if set(str(value) for value in cluster["surface_kinds"]) != expected_surface_kinds:  # type: ignore[union-attr]
            raise ContractError("MLE cluster surface kinds are inconsistent with its patches.")
    provenance = _provenance(diagnostics)
    if provenance["estimator_variant"] != mode:
        raise ContractError("MLE provenance estimator_variant differs from diagnostics mode.")
    if (
        expected_log_sha256 is not None
        and provenance["measurement_log_sha256"] != expected_log_sha256
    ):
        raise ContractError("MLE provenance measurement-log hash does not match its input.")
    if expected_commit is not None and provenance["estimator_commit"] != expected_commit:
        raise ContractError("MLE provenance commit does not match the pinned revision.")
    if expected_config_sha256 is not None and provenance["config_sha256"] != expected_config_sha256:
        raise ContractError("MLE provenance config hash does not match its input.")
    if (
        expected_resolved_config_sha256 is not None
        and provenance["resolved_estimator_config_sha256"] != expected_resolved_config_sha256
    ):
        raise ContractError("MLE provenance resolved estimator config hash does not match.")
    inventory = directory_inventory(directory)
    return MLEResultInfo(
        root=directory,
        diagnostics=MappingProxyType(diagnostics),
        hotspot_clusters=tuple(MappingProxyType(value) for value in clusters),
        arrays=MappingProxyType(arrays),
        artifact_inventory=MappingProxyType(inventory),
        result_sha256=inventory_digest(inventory),
    )


def validate_mle_snapshot(path: str | Path) -> MappingProxyType[str, object]:
    """Validate a cutoff-bound hybrid MLE snapshot and its coverage invariant."""
    payload = load_json(path)
    _validate_schema(payload, "mle_snapshot_schema.json", label="MLESnapshot")
    steps = payload["covered_step_ids"]
    assert isinstance(steps, list)
    cutoff = int(payload["data_cutoff_step"])
    if steps != sorted(steps) or max(int(value) for value in steps) != cutoff:
        raise ContractError(
            "MLESnapshot covered_step_ids must be sorted and end exactly at data_cutoff_step."
        )
    return MappingProxyType(payload)


def validate_truth(path: str | Path, *, expected_run_id: str | None = None) -> dict[str, object]:
    """Validate evaluation-only truth; callers must remain in the evaluation phase."""
    payload = load_json(path)
    required = {"schema_version", "run_id", "sources", "ceiling_z_m", "match_radius_m"}
    missing = required - payload.keys()
    if missing:
        raise ContractError(f"Truth file is missing fields: {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ContractError("Truth schema_version must be 1.")
    if expected_run_id is not None and payload["run_id"] != expected_run_id:
        raise ContractError("Truth run_id does not match MeasurementLog run_id.")
    sources = payload["sources"]
    if not isinstance(sources, list):
        raise ContractError("Truth sources must be an array.")
    seen: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ContractError(f"Truth source {index} must be an object.")
        for field in ("source_id", "isotope", "position_xyz", "strength_cps_1m", "surface_kind"):
            if field not in source:
                raise ContractError(f"Truth source {index} is missing {field!r}.")
        source_id = str(source["source_id"])
        if source_id in seen:
            raise ContractError(f"Duplicate truth source_id: {source_id}")
        seen.add(source_id)
        position = np.asarray(source["position_xyz"], dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ContractError(f"Truth source {index} position_xyz is invalid.")
        strength = source["strength_cps_1m"]
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or strength < 0:
            raise ContractError(f"Truth source {index} strength must be nonnegative.")
    if not isinstance(payload["ceiling_z_m"], (int, float)):
        raise ContractError("ceiling_z_m must be numeric.")
    if not isinstance(payload["match_radius_m"], (int, float)) or payload["match_radius_m"] <= 0:
        raise ContractError("match_radius_m must be positive.")
    return payload
