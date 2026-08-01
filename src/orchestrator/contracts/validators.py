"""Filesystem-aware validation for pure-estimator and causal-hybrid contracts."""

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
SUPPORTED_MEASUREMENT_LOG_SCHEMA_VERSIONS = frozenset({1, 2})
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

OBSERVATION_BASE_ARRAYS = frozenset(
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
    }
)
OBSERVATION_V1_OPTIONAL_ARRAYS = frozenset(
    {
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
OBSERVATION_ARRAYS = OBSERVATION_BASE_ARRAYS | OBSERVATION_V1_OPTIONAL_ARRAYS


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
    def schema_version(self) -> int:
        """Return the validated MeasurementLog schema version."""
        return int(self.manifest["schema_version"])

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


@dataclass(frozen=True, slots=True)
class MLESnapshotInfo:
    """Validated v2 exact-prefix MLE snapshot."""

    path: Path
    payload: MappingProxyType[str, object]
    snapshot_sha256: str

    @property
    def cutoff_step(self) -> int:
        return int(self.payload["data_cutoff_step"])

    @property
    def covered_step_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["covered_step_ids"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FutureCandidateScoreInfo:
    """Validated future-only predictive scores for one frozen MLE snapshot."""

    path: Path
    payload: MappingProxyType[str, object]
    score_sha256: str

    @property
    def future_step_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["future_step_ids"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class HybridPlanningRequestInfo:
    """Validated collision-attested request for the PF DSS-PP boundary."""

    path: Path
    payload: MappingProxyType[str, object]
    request_sha256: str


@dataclass(frozen=True, slots=True)
class HybridPlanningRecommendationInfo:
    """Validated algorithmic recommendation that cannot authorize actuation."""

    path: Path
    payload: MappingProxyType[str, object]
    recommendation_sha256: str


@dataclass(frozen=True, slots=True)
class PFDirectiveInfo:
    """Validated once-only MLE-to-PF directive."""

    path: Path
    payload: MappingProxyType[str, object]
    directive_sha256: str

    @property
    def cutoff_step(self) -> int:
        return int(self.payload["data_cutoff_step"])


@dataclass(frozen=True, slots=True)
class PFDirectiveReceiptInfo:
    """Validated proof of one safe PF directive application."""

    path: Path
    payload: MappingProxyType[str, object]
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class HybridLedgerSummaryInfo:
    """Validated hash-chained observation-use ledger summary."""

    path: Path
    payload: MappingProxyType[str, object]
    summary_sha256: str


@dataclass(frozen=True, slots=True)
class HybridResultInfo:
    """Validated causal PF+MLE hybrid result manifest."""

    path: Path
    payload: MappingProxyType[str, object]
    result_sha256: str

    @property
    def authoritative_clusters(self) -> tuple[MappingProxyType[str, object], ...]:
        """Return the final cold spectral-MLE hotspot report."""
        report = self.payload["authoritative_report"]
        assert isinstance(report, dict)
        clusters = report["hotspot_clusters"]
        assert isinstance(clusters, list)
        return tuple(MappingProxyType(cluster) for cluster in clusters)


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
        aggregate_validation_metrics = location.endswith(
            ".full_spectrum_generative_model.validation.metrics"
        )
        for raw_key, value in payload.items():
            key = _normalized_name(raw_key)
            child_location = f"{location}.{raw_key}"
            if (
                not aggregate_validation_metrics
                and _indicates_realized_truth(key, key=True)
            ):
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


def _validate_forward_manifest(
    payload: dict[str, object],
    run_manifest: dict[str, object],
    *,
    schema_version: int,
) -> None:
    """Validate the matching producer-owned forward-model manifest version."""
    _validate_schema(
        payload,
        (
            "forward_model_manifest_v2_schema.json"
            if schema_version == 2
            else "forward_model_manifest_schema.json"
        ),
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

    if schema_version == 2:
        # V2 component hashes bind the complete producer-side configuration
        # and file-backed model assets, not only this line-table projection.
        # The orchestrator verifies their immutable identity but does not
        # recreate simulation physics in a second repository.
        return

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
    arrays: dict[str, NDArray[Any]],
    *,
    records: int,
    bins: int,
    isotopes: int,
    schema_version: int,
) -> None:
    expected_arrays = (
        OBSERVATION_BASE_ARRAYS if schema_version == 2 else OBSERVATION_ARRAYS
    )
    missing = sorted(expected_arrays - arrays.keys())
    extra = sorted(arrays.keys() - expected_arrays)
    if missing or extra:
        raise ContractError(
            f"observations.npz v{schema_version} schema mismatch; "
            f"missing={missing}, extra={extra}"
        )
    if any(_indicates_realized_truth(_normalized_name(name), key=True) for name in arrays):
        raise TruthIsolationError("observations.npz may not embed realized-truth arrays.")
    steps = _array(arrays, "step_id", shape=(records,), dtype=np.int64)
    actions = _array(arrays, "action_id", shape=(records,), dtype=np.int64)
    stations = _array(arrays, "station_id", shape=(records,), dtype=np.int64)
    if schema_version == 2:
        row_order = np.arange(records, dtype=np.int64)
        if not np.array_equal(steps, row_order):
            raise ContractError("MeasurementLog v2 step_id must equal causal row order.")
        if not np.array_equal(actions, row_order):
            raise ContractError("MeasurementLog v2 action_id must equal causal row order.")
        station_delta = np.diff(stations)
        if stations[0] != 0 or np.any((station_delta < 0) | (station_delta > 1)):
            raise ContractError(
                "MeasurementLog v2 station_id must form contiguous zero-based groups."
            )
    else:
        if np.any(steps < 0) or np.any(np.diff(steps) <= 0):
            raise ContractError(
                "step_id must be nonnegative and strictly increasing in causal order."
            )
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
    ):
        value = _array(arrays, name, shape=shape, dtype=np.float64)
        _finite(name, value)
    _array(
        arrays,
        "spectrum_counts",
        shape=(records, bins),
        dtype=np.int64 if schema_version == 2 else np.float64,
    )
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

    if schema_version == 2:
        return

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
    """Validate canonical truth-free MeasurementLog v1 or raw-spectrum v2."""
    directory = _require_directory(root, MEASUREMENT_REQUIRED_FILES, label="MeasurementLog")
    _reject_truth(directory)
    manifest = load_json(directory / "run_manifest.json")
    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_MEASUREMENT_LOG_SCHEMA_VERSIONS:
        raise ContractError(
            "Unsupported MeasurementLog schema_version "
            f"{schema_version!r}; expected one of "
            f"{sorted(SUPPORTED_MEASUREMENT_LOG_SCHEMA_VERSIONS)}."
        )
    _validate_schema(
        manifest,
        (
            "measurement_log_v2_schema.json"
            if schema_version == 2
            else "measurement_log_schema.json"
        ),
        label="run_manifest.json",
    )
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
    _validate_forward_manifest(
        forward,
        manifest,
        schema_version=int(schema_version),
    )

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
    _validate_measurement_arrays(
        arrays,
        records=records,
        bins=bins,
        isotopes=len(isotope_names),
        schema_version=int(schema_version),
    )

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
    measurement_log_schema_version = provenance.get(
        "measurement_log_schema_version"
    )
    if (
        isinstance(measurement_log_schema_version, bool)
        or not isinstance(measurement_log_schema_version, int)
        or measurement_log_schema_version not in SUPPORTED_MEASUREMENT_LOG_SCHEMA_VERSIONS
    ):
        raise ContractError(
            "PF provenance must declare MeasurementLog schema version 1 or 2."
        )
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
    expected_trace_schema = 2 if measurement_log_schema_version == 2 else SCHEMA_VERSION
    expected_trace_family = (
        "pure_particle_filter"
        if measurement_log_schema_version == 2
        else "particle_filter"
    )
    for index, payload in enumerate(trace_payloads):
        if payload.get("schema_version") != expected_trace_schema:
            raise ContractError(f"PF trace line {index + 1} has unsupported schema_version.")
        if payload.get("estimator_family") != expected_trace_family:
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
    expected_diagnostics_schema = (
        2 if measurement_log_schema_version == 2 else SCHEMA_VERSION
    )
    if diagnostics.get("schema_version") != expected_diagnostics_schema:
        raise ContractError(
            "pf_diagnostics.json schema_version is incompatible with its "
            "MeasurementLog input."
        )
    if diagnostics.get("measurement_log_schema_version") != (
        measurement_log_schema_version
    ):
        raise ContractError(
            "PF diagnostics MeasurementLog schema version differs from provenance."
        )
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


def _snapshot_coverage(payload: dict[str, object]) -> tuple[int, ...]:
    steps = payload["covered_step_ids"]
    assert isinstance(steps, list)
    normalized = tuple(int(value) for value in steps)
    cutoff = int(payload["data_cutoff_step"])
    if tuple(sorted(normalized)) != normalized or normalized[-1] != cutoff:
        raise ContractError(
            "MLESnapshot covered_step_ids must be sorted and end exactly at data_cutoff_step."
        )
    return normalized


def validate_mle_snapshot(path: str | Path) -> MappingProxyType[str, object]:
    """Validate a v1 or v2 cutoff-bound MLE snapshot without weakening v1 callers."""
    payload = load_json(path)
    version = payload.get("schema_version")
    if version == 1:
        _validate_schema(payload, "mle_snapshot_schema.json", label="MLESnapshot")
    elif version == 2:
        _validate_schema(payload, "mle_snapshot_v2_schema.json", label="MLESnapshot")
    else:
        raise ContractError(f"Unsupported MLESnapshot schema_version: {version!r}.")
    _snapshot_coverage(payload)
    if version == 2:
        _validate_mle_snapshot_v2_semantics(payload)
    return MappingProxyType(payload)


def _validate_mle_snapshot_v2_semantics(payload: dict[str, object]) -> None:
    steps = _snapshot_coverage(payload)
    predictions = payload["predicted_observations"]
    assert isinstance(predictions, list)
    prediction_steps = tuple(int(row["step_id"]) for row in predictions)  # type: ignore[index]
    if prediction_steps != steps:
        raise ContractError(
            "MLESnapshot predicted_observations must exactly cover the declared prefix."
        )
    warm = payload["warm_start"]
    assert isinstance(warm, dict)
    has_identifiers = warm["snapshot_id"] is not None and warm["mle_result_sha256"] is not None
    if bool(warm["used"]) != has_identifiers:
        raise ContractError("MLESnapshot warm-start flag and identifiers are inconsistent.")
    clusters = payload["clusters"]
    assert isinstance(clusters, list)
    candidate_ids = [str(cluster["snapshot_candidate_id"]) for cluster in clusters]  # type: ignore[index]
    cluster_ids = [int(cluster["cluster_id"]) for cluster in clusters]  # type: ignore[index]
    if len(candidate_ids) != len(set(candidate_ids)) or len(cluster_ids) != len(set(cluster_ids)):
        raise ContractError("MLESnapshot candidate and cluster IDs must be unique.")


def validate_mle_snapshot_v2(
    path: str | Path,
    *,
    expected_covered_step_ids: tuple[int, ...] | None = None,
    expected_source_run_id: str | None = None,
    expected_prefix_log_sha256: str | None = None,
    expected_covered_records_sha256: str | None = None,
    expected_covered_station_boundaries_sha256: str | None = None,
    expected_previous_snapshot: MLESnapshotInfo | None = None,
    expected_previous_mle_result: MLEResultInfo | None = None,
) -> MLESnapshotInfo:
    """Validate v2 and optionally bind it to the controller's exact prefix."""
    source = Path(path).resolve()
    payload = load_json(source)
    _validate_schema(payload, "mle_snapshot_v2_schema.json", label="MLESnapshot v2")
    _validate_mle_snapshot_v2_semantics(payload)
    steps = _snapshot_coverage(payload)
    if expected_covered_step_ids is not None and steps != expected_covered_step_ids:
        raise ContractError("MLESnapshot coverage differs from the exact controller prefix.")
    if expected_source_run_id is not None and payload["source_run_id"] != expected_source_run_id:
        raise ContractError("MLESnapshot source run ID differs from its controller run.")
    if (
        expected_prefix_log_sha256 is not None
        and payload["prefix_measurement_log_sha256"] != expected_prefix_log_sha256
    ):
        raise ContractError("MLESnapshot prefix MeasurementLog hash differs from its fit input.")
    if (
        expected_covered_records_sha256 is not None
        and payload["covered_records_sha256"] != expected_covered_records_sha256
    ):
        raise ContractError("MLESnapshot covered-record digest differs from its exact prefix.")
    if (
        expected_covered_station_boundaries_sha256 is not None
        and payload["covered_station_boundaries_sha256"]
        != expected_covered_station_boundaries_sha256
    ):
        raise ContractError("MLESnapshot station-boundary digest differs from its prefix.")
    if (expected_previous_snapshot is None) != (expected_previous_mle_result is None):
        raise ContractError(
            "Previous snapshot and MLE result must be supplied together for warm validation."
        )
    if expected_previous_snapshot is not None and expected_previous_mle_result is not None:
        warm = payload["warm_start"]
        assert isinstance(warm, dict)
        expected_warm = {
            "used": True,
            "snapshot_id": expected_previous_snapshot.payload["snapshot_id"],
            "mle_result_sha256": expected_previous_mle_result.result_sha256,
        }
        if warm != expected_warm:
            raise ContractError("MLESnapshot warm-start ancestry differs from prior artifacts.")
    return MLESnapshotInfo(
        path=source,
        payload=MappingProxyType(payload),
        snapshot_sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


def validate_future_candidate_score(
    path: str | Path,
    *,
    expected_snapshot: MLESnapshotInfo | None = None,
    expected_snapshot_mle_result: MLEResultInfo | None = None,
    expected_current_log: MeasurementLogInfo | None = None,
    expected_current_covered_records_sha256: str | None = None,
) -> FutureCandidateScoreInfo:
    """Validate frozen-snapshot scores and bind them to an exact later prefix."""
    source = Path(path).resolve()
    payload = load_json(source)
    _validate_schema(
        payload,
        "future_candidate_score_schema.json",
        label="FutureCandidateScore",
    )
    cutoff = int(payload["snapshot_data_cutoff_step"])
    future_steps = tuple(int(value) for value in payload["future_step_ids"])  # type: ignore[arg-type]
    future_stations = tuple(int(value) for value in payload["future_station_ids"])  # type: ignore[arg-type]
    if len(future_steps) != len(future_stations):
        raise ContractError("Future score step and station arrays must have equal length.")
    if any(right <= left for left, right in pairwise(future_steps)):
        raise ContractError("Future score steps must be strictly increasing.")
    if future_steps[0] <= cutoff:
        raise ContractError("Future score evidence must begin strictly after the snapshot cutoff.")
    if any(right < left for left, right in pairwise(future_stations)):
        raise ContractError("Future score station IDs must be nondecreasing.")

    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    candidate_ids: set[str] = set()
    cluster_ids: set[int] = set()
    for candidate in candidates:
        assert isinstance(candidate, dict)
        candidate_id = str(candidate["snapshot_candidate_id"])
        cluster_id = int(candidate["cluster_id"])
        if candidate_id in candidate_ids or cluster_id in cluster_ids:
            raise ContractError("Future score candidate and cluster IDs must be unique.")
        candidate_ids.add(candidate_id)
        cluster_ids.add(cluster_id)
        rows = candidate["future_step_scores"]
        assert isinstance(rows, list)
        row_steps = tuple(int(row["step_id"]) for row in rows)  # type: ignore[index]
        row_stations = tuple(int(row["station_id"]) for row in rows)  # type: ignore[index]
        if row_steps != future_steps or row_stations != future_stations:
            raise ContractError(
                "Every future-score candidate must cover the exact declared future rows."
            )
        values = tuple(float(row["log_predictive_likelihood_ratio"]) for row in rows)  # type: ignore[index]
        cumulative = float(candidate["cumulative_log_predictive_likelihood_ratio"])
        if not np.isclose(cumulative, sum(values), rtol=0.0, atol=1e-12):
            raise ContractError("Future score cumulative likelihood ratio is inconsistent.")

    hashes = payload["hashes"]
    assert isinstance(hashes, dict)
    if expected_snapshot is not None:
        snapshot = expected_snapshot.payload
        expected_fields = {
            "source_run_id": snapshot["source_run_id"],
            "snapshot_id": snapshot["snapshot_id"],
            "snapshot_data_cutoff_step": snapshot["data_cutoff_step"],
            "snapshot_data_cutoff_station": snapshot["data_cutoff_station"],
        }
        if any(payload[name] != value for name, value in expected_fields.items()):
            raise ContractError("Future score identity differs from its frozen MLE snapshot.")
        expected_hashes = {
            "snapshot_file_sha256": sha256_file(expected_snapshot.path),
            "snapshot_canonical_sha256": expected_snapshot.snapshot_sha256,
            "snapshot_mle_report_sha256": snapshot["mle_result_sha256"],
            "snapshot_prefix_measurement_log_sha256": snapshot["prefix_measurement_log_sha256"],
            "snapshot_covered_station_boundaries_sha256": snapshot[
                "covered_station_boundaries_sha256"
            ],
        }
        if any(hashes[name] != value for name, value in expected_hashes.items()):
            raise ContractError("Future score hashes differ from its frozen MLE snapshot.")
        snapshot_clusters = snapshot["clusters"]
        assert isinstance(snapshot_clusters, list)
        expected_candidates = {
            str(cluster["snapshot_candidate_id"]): (  # type: ignore[index]
                int(cluster["cluster_id"]),  # type: ignore[index]
                str(cluster["isotope"]),  # type: ignore[index]
                tuple(int(value) for value in cluster["patch_ids"]),  # type: ignore[index]
            )
            for cluster in snapshot_clusters
        }
        observed_candidates = {
            str(candidate["snapshot_candidate_id"]): (
                int(candidate["cluster_id"]),
                str(candidate["isotope"]),
                tuple(int(value) for value in candidate["patch_ids"]),  # type: ignore[arg-type]
            )
            for candidate in candidates
        }
        if observed_candidates != expected_candidates:
            raise ContractError("Future score candidates differ from the snapshot clusters.")
    if expected_snapshot_mle_result is not None:
        if hashes["snapshot_mle_report_sha256"] != expected_snapshot_mle_result.result_sha256:
            raise ContractError("Future score MLE report hash differs from its validated result.")
    if expected_current_log is not None:
        expected_future = tuple(step for step in expected_current_log.step_ids if step > cutoff)
        expected_stations = tuple(
            station
            for step, station in zip(
                expected_current_log.step_ids,
                expected_current_log.station_ids,
                strict=True,
            )
            if step > cutoff
        )
        if future_steps != expected_future or future_stations != expected_stations:
            raise ContractError("Future score rows differ from the exact current prefix suffix.")
        if payload["source_run_id"] != expected_current_log.manifest["run_id"]:
            raise ContractError("Future score source run differs from the current prefix.")
        if hashes["current_measurement_log_sha256"] != (
            expected_current_log.measurement_log_sha256
        ):
            raise ContractError("Future score current MeasurementLog hash is invalid.")
    if (
        expected_current_covered_records_sha256 is not None
        and hashes["current_covered_records_sha256"] != expected_current_covered_records_sha256
    ):
        raise ContractError("Future score current covered-record digest is invalid.")
    return FutureCandidateScoreInfo(
        path=source,
        payload=MappingProxyType(payload),
        score_sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


def validate_hybrid_planning_request(path: str | Path) -> HybridPlanningRequestInfo:
    """Validate the estimator-neutral, collision-attested planning input."""
    source = Path(path).resolve()
    payload = load_json(source)
    _validate_schema(
        payload,
        "hybrid_planning_request_schema.json",
        label="HybridPlanningRequest",
    )
    candidates = np.asarray(payload["candidate_poses_xyz"], dtype=np.float64)
    if not np.all(np.isfinite(candidates)) or len({tuple(row) for row in candidates}) != len(
        candidates
    ):
        raise ContractError("Hybrid planning candidates must be finite and unique.")
    attestation = payload["candidate_attestation"]
    assert isinstance(attestation, dict)
    expected_candidates_hash = sha256_bytes(canonical_json_bytes(candidates.tolist()))
    if attestation["candidate_poses_sha256"] != expected_candidates_hash:
        raise ContractError("Hybrid planning candidate attestation hash is invalid.")
    dsspp = payload["dsspp_config"]
    assert isinstance(dsspp, dict)
    if dsspp.get("augment_candidates") is not False:
        raise ContractError("Hybrid planning must disable unattested candidate augmentation.")
    if (
        dsspp.get("include_runtime_rescue_modes", False) is not False
        or dsspp.get("include_global_surface_rescue_modes", False) is not False
    ):
        raise ContractError("Hybrid planning may not consume legacy rescue modes.")
    modes = payload["external_modes"]
    assert isinstance(modes, list)
    mode_ids = [str(mode["mode_id"]) for mode in modes]  # type: ignore[index]
    if len(mode_ids) != len(set(mode_ids)):
        raise ContractError("Hybrid planning external mode IDs must be unique.")
    bounds = payload.get("bounds_xyz")
    if isinstance(bounds, dict):
        lower = np.asarray(bounds["min"], dtype=float)
        upper = np.asarray(bounds["max"], dtype=float)
        if np.any(lower > upper):
            raise ContractError("Hybrid planning bounds minimum exceeds maximum.")
    heights = payload.get("continuous_height_bounds_m")
    if isinstance(heights, list) and float(heights[0]) > float(heights[1]):
        raise ContractError("Hybrid planning height bounds are reversed.")
    return HybridPlanningRequestInfo(
        path=source,
        payload=MappingProxyType(payload),
        request_sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


def _contains_key(value: object, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(
            _contains_key(child, forbidden) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def validate_hybrid_planning_recommendation(
    path: str | Path,
    *,
    expected_request: HybridPlanningRequestInfo | None = None,
) -> HybridPlanningRecommendationInfo:
    """Validate recommendation-only semantics and optional request binding."""
    source = Path(path).resolve()
    payload = load_json(source)
    _validate_schema(
        payload,
        "hybrid_planning_recommendation_schema.json",
        label="HybridPlanningRecommendation",
    )
    if _contains_key(payload, "measurement_log_sha256"):
        raise ContractError("Hybrid planning causal identity may not include a full-log hash.")
    integrity = payload["pf_state_integrity"]
    assert isinstance(integrity, dict)
    if (
        integrity.get("state_sha256_before_planning")
        != integrity.get("state_sha256_after_planning")
        or integrity.get("pf_particles_or_weights_mutated_by_planning") is not False
        or integrity.get("external_modes_mutated_pf") is not False
    ):
        raise ContractError("Hybrid planning recommendation mutated PF state.")
    if expected_request is not None:
        request = expected_request.payload
        boundary = payload["causal_boundary"]
        attestation = payload["candidate_attestation"]
        belief = payload["belief"]
        provenance = payload["provenance"]
        selected = payload["selected_action"]
        assert isinstance(boundary, dict)
        assert isinstance(attestation, dict)
        assert isinstance(belief, dict)
        assert isinstance(provenance, dict)
        assert isinstance(selected, dict)
        expected_boundary = {
            "source_run_id": request["source_run_id"],
            "data_cutoff_step": request["data_cutoff_step"],
            "data_cutoff_station": request["data_cutoff_station"],
            "covered_records_sha256": request["covered_records_sha256"],
            "pf_resolved_config_sha256": request["pf_resolved_config_sha256"],
            "causal_identity_uses_record_prefix_only": True,
        }
        if boundary != expected_boundary:
            raise ContractError("Hybrid planning recommendation causal boundary is invalid.")
        if attestation != request["candidate_attestation"]:
            raise ContractError("Hybrid planning recommendation candidate attestation differs.")
        if provenance.get("pf_resolved_config_sha256") != request["pf_resolved_config_sha256"]:
            raise ContractError("Hybrid planning recommendation PF config hash differs.")
        if provenance.get("causal_planning_request_sha256") != expected_request.request_sha256:
            raise ContractError(
                "Hybrid planning recommendation is not bound to the exact request artifact."
            )
        index = int(selected["candidate_index"])
        candidates = request["candidate_poses_xyz"]
        assert isinstance(candidates, list)
        if index >= len(candidates) or selected["pose_xyz"] != candidates[index]:
            raise ContractError("Hybrid planning selected action is outside attested candidates.")
        modes = request["external_modes"]
        assert isinstance(modes, list)
        included = [
            str(mode["mode_id"])  # type: ignore[index]
            for mode in modes
            if mode["verification_state"] in {"pending", "verified"}  # type: ignore[index]
        ]
        excluded = [
            str(mode["mode_id"])  # type: ignore[index]
            for mode in modes
            if mode["verification_state"] == "quarantined"  # type: ignore[index]
        ]
        if (
            belief.get("included_external_mode_ids") != included
            or belief.get("excluded_quarantined_mode_ids") != excluded
        ):
            raise ContractError("Hybrid planning recommendation mode filtering differs.")
    return HybridPlanningRecommendationInfo(
        path=source,
        payload=MappingProxyType(payload),
        recommendation_sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


def _directive_semantics(
    payload: dict[str, object], *, expected_snapshot: MLESnapshotInfo | None = None
) -> None:
    steps = tuple(int(value) for value in payload["covered_step_ids"])  # type: ignore[arg-type]
    cutoff = int(payload["data_cutoff_step"])
    if tuple(sorted(steps)) != steps or steps[-1] != cutoff:
        raise ContractError("PFDirective covered steps must be sorted and end at its cutoff.")
    if int(payload["apply_after_step"]) != cutoff:
        raise ContractError("PFDirective must be applied immediately after its cutoff state.")
    if int(payload["corroboration_min_step"]) != cutoff + 1:
        raise ContractError("PFDirective corroboration must begin strictly after the cutoff.")
    safety = payload["safety_policy"]
    assert isinstance(safety, dict)
    kind = str(payload["directive_kind"])
    requires_mh = kind == "proposal_only_mh"
    if bool(safety["requires_target_preserving_mh"]) != requires_mh:
        raise ContractError("PFDirective MH requirement is inconsistent with directive_kind.")
    proposals = payload["proposals"]
    assert isinstance(proposals, list)
    if not proposals:
        raise ContractError("PFDirective must contain at least one proposal.")
    identifiers = [str(proposal["proposal_id"]) for proposal in proposals]  # type: ignore[index]
    if len(identifiers) != len(set(identifiers)):
        raise ContractError("PFDirective proposal IDs must be unique.")
    for proposal in proposals:
        assert isinstance(proposal, dict)
        kernel = proposal["proposal_kernel"]
        if requires_mh and kernel is None:
            raise ContractError("proposal_only_mh requires a density-defined proposal kernel.")
        if not requires_mh and kernel is not None:
            raise ContractError("verification_only may not carry a PF proposal kernel.")
    if expected_snapshot is not None:
        snapshot = expected_snapshot.payload
        if payload["snapshot_id"] != snapshot["snapshot_id"]:
            raise ContractError("PFDirective snapshot ID differs from its validated snapshot.")
        if payload["snapshot_sha256"] != expected_snapshot.snapshot_sha256:
            raise ContractError("PFDirective snapshot hash differs from its validated snapshot.")
        for field in (
            "source_run_id",
            "prefix_measurement_log_sha256",
            "covered_records_sha256",
            "covered_station_boundaries_sha256",
            "data_cutoff_step",
            "data_cutoff_station",
            "covered_step_ids",
            "cutoff_station_complete",
        ):
            if payload[field] != snapshot[field]:
                raise ContractError(f"PFDirective field {field!r} differs from MLESnapshot.")
        available = {
            str(cluster["snapshot_candidate_id"])
            for cluster in snapshot["clusters"]  # type: ignore[index,union-attr]
        }
        if any(str(proposal["snapshot_candidate_id"]) not in available for proposal in proposals):
            raise ContractError("PFDirective proposal is absent from its MLESnapshot.")


def validate_pf_directive(
    path: str | Path, *, expected_snapshot: MLESnapshotInfo | None = None
) -> PFDirectiveInfo:
    """Validate a safe once-only PF directive and its source-snapshot binding."""
    source = Path(path).resolve()
    payload = load_json(source)
    _validate_schema(payload, "pf_directive_schema.json", label="PFDirective")
    _directive_semantics(payload, expected_snapshot=expected_snapshot)
    return PFDirectiveInfo(
        path=source,
        payload=MappingProxyType(payload),
        directive_sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


def validate_pf_directive_receipt(
    path: str | Path, *, expected_directive: PFDirectiveInfo | None = None
) -> PFDirectiveReceiptInfo:
    """Validate one PF application proof and its no-reweight/MH evidence."""
    source = Path(path).resolve()
    payload = load_json(source)
    _validate_schema(payload, "pf_directive_receipt_schema.json", label="PFDirectiveReceipt")
    cutoff = int(payload["data_cutoff_step"])
    if int(payload["applied_after_step"]) != cutoff:
        raise ContractError("PFDirectiveReceipt applied_after_step must equal its cutoff.")
    safety = payload["safety_evidence"]
    assert isinstance(safety, dict)
    if int(safety["next_observation_min_step"]) != cutoff + 1:
        raise ContractError("PF must resume observation processing strictly after the cutoff.")
    kind = str(payload["directive_kind"])
    status = str(payload["status"])
    target_mh = kind == "proposal_only_mh" and status == "applied"
    if bool(safety["target_preserving_mh_performed"]) != target_mh:
        raise ContractError("PFDirectiveReceipt target-preserving MH evidence is inconsistent.")
    outcomes = payload["candidate_outcomes"]
    assert isinstance(outcomes, list)
    outcome_ids = [str(outcome["proposal_id"]) for outcome in outcomes]  # type: ignore[index]
    if len(outcome_ids) != len(set(outcome_ids)):
        raise ContractError("PFDirectiveReceipt proposal outcomes must be unique.")
    for outcome in outcomes:
        assert isinstance(outcome, dict)
        value = str(outcome["outcome"])
        ratio = outcome["mh_log_acceptance_ratio"]
        draw = outcome["mh_log_uniform_draw"]
        attempts = int(outcome["mh_attempt_count"])
        accepted = int(outcome["mh_accepted_count"])
        rejected = int(outcome["mh_rejected_count"])
        not_sampled = int(outcome["not_sampled_count"])
        eligible = int(outcome["eligible_particle_count"])
        if attempts != accepted + rejected or eligible != attempts + not_sampled:
            raise ContractError("PFDirectiveReceipt candidate aggregate counts are inconsistent.")
        expected_value = (
            "not_applied"
            if attempts == 0
            else "mh_accepted"
            if accepted == attempts
            else "mh_rejected"
            if rejected == attempts
            else "mh_mixed"
        )
        paired_scalar_evidence = ratio is not None and draw is not None
        if (ratio is None) != (draw is None) or paired_scalar_evidence != (attempts == 1):
            raise ContractError(
                "PFDirectiveReceipt scalar MH evidence is valid only for one attempt."
            )
        if kind == "proposal_only_mh" and status == "applied":
            if value != expected_value:
                raise ContractError(
                    "Applied MH outcome label differs from its aggregate particle counts."
                )
        elif kind == "verification_only" and status == "applied":
            if (
                value != "registered"
                or attempts != 0
                or eligible != 0
                or ratio is not None
                or draw is not None
            ):
                raise ContractError("Verification receipts may only register candidates.")
        elif status == "rejected" and (value != "not_applied" or eligible != 0):
            raise ContractError(
                "Rejected directives must mark every proposal not_applied with zero counts."
            )
    if expected_directive is not None:
        directive = expected_directive.payload
        for receipt_field, directive_field in (
            ("directive_id", "directive_id"),
            ("directive_kind", "directive_kind"),
            ("data_cutoff_step", "data_cutoff_step"),
        ):
            if payload[receipt_field] != directive[directive_field]:
                raise ContractError("PFDirectiveReceipt differs from its validated directive.")
        if payload["directive_sha256"] != expected_directive.directive_sha256:
            raise ContractError("PFDirectiveReceipt directive hash is invalid.")
        provenance = payload["provenance"]
        assert isinstance(provenance, dict)
        for field in (
            "source_run_id",
            "covered_records_sha256",
            "pf_resolved_config_sha256",
        ):
            if provenance[field] != directive[field]:
                raise ContractError(
                    f"PFDirectiveReceipt provenance field {field!r} differs from directive."
                )
        expected_ids = {
            str(proposal["proposal_id"])
            for proposal in directive["proposals"]  # type: ignore[index,union-attr]
        }
        if set(outcome_ids) != expected_ids:
            raise ContractError("PFDirectiveReceipt must account for every directive proposal.")
    return PFDirectiveReceiptInfo(
        path=source,
        payload=MappingProxyType(payload),
        receipt_sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


def validate_hybrid_ledger_summary(path: str | Path) -> HybridLedgerSummaryInfo:
    """Recompute a ledger hash chain and its once-only/future-only invariants."""
    source = Path(path).resolve()
    payload = load_json(source)
    _validate_schema(payload, "hybrid_ledger_summary_schema.json", label="HybridLedgerSummary")
    events = payload["events"]
    assert isinstance(events, list)
    if int(payload["event_count"]) != len(events):
        raise ContractError("Hybrid ledger event_count differs from its event array.")
    previous = str(payload["genesis_event_sha256"])
    seen_event_ids: set[str] = set()
    snapshots: set[str] = set()
    directives: dict[str, tuple[int, set[str], str]] = {}
    receipts: set[str] = set()
    applied_receipts: set[str] = set()
    corroboration: set[tuple[str, str, int]] = set()
    for index, event in enumerate(events):
        assert isinstance(event, dict)
        if int(event["event_index"]) != index:
            raise ContractError("Hybrid ledger event indices must be contiguous from zero.")
        if event["previous_event_sha256"] != previous:
            raise ContractError("Hybrid ledger previous-event hash chain is broken.")
        event_id = str(event["event_id"])
        if event_id in seen_event_ids:
            raise ContractError("Hybrid ledger event IDs must be unique.")
        seen_event_ids.add(event_id)
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        expected_hash = sha256_bytes(canonical_json_bytes(body))
        if event["event_sha256"] != expected_hash:
            raise ContractError("Hybrid ledger event content hash is invalid.")
        previous = expected_hash
        event_type = str(event["event_type"])
        child = event["payload"]
        assert isinstance(child, dict)
        if event_type == "snapshot_registered":
            snapshots.add(str(child["snapshot_id"]))
        elif event_type == "directive_issued":
            directive_id = str(child["directive_id"])
            if str(child["snapshot_id"]) not in snapshots:
                raise ContractError("Hybrid ledger directive precedes its snapshot.")
            if directive_id in directives:
                raise ContractError("Hybrid ledger issued a directive more than once.")
            proposals = {str(value) for value in child["proposal_ids"]}  # type: ignore[union-attr]
            if child.get("direct_mle_objective_reweight") is not False:
                raise ContractError("Hybrid ledger directive attempted MLE-objective reweighting.")
            if child.get("hard_prune_authorized") is not False:
                raise ContractError("Hybrid ledger directive attempted to authorize hard pruning.")
            directives[directive_id] = (
                int(child["data_cutoff_step"]),
                proposals,
                str(child["directive_sha256"]),
            )
        elif event_type == "directive_receipt":
            directive_id = str(child["directive_id"])
            if directive_id not in directives:
                raise ContractError("Hybrid ledger receipt precedes directive issuance.")
            if directive_id in receipts:
                raise ContractError("Hybrid ledger applied one directive more than once.")
            _validate_schema(
                child,
                "pf_directive_receipt_schema.json",
                label="Hybrid ledger directive receipt",
            )
            cutoff, _, directive_hash = directives[directive_id]
            if child["directive_sha256"] != directive_hash:
                raise ContractError("Hybrid ledger receipt has a different directive hash.")
            if int(child["data_cutoff_step"]) != cutoff:
                raise ContractError("Hybrid ledger receipt has a different directive cutoff.")
            receipts.add(directive_id)
            if child["status"] == "applied":
                applied_receipts.add(directive_id)
        elif event_type == "corroboration":
            directive_id = str(child["directive_id"])
            proposal_id = str(child["proposal_id"])
            step_id = int(child["step_id"])
            if directive_id not in directives:
                raise ContractError("Hybrid ledger corroboration precedes directive issuance.")
            if directive_id not in applied_receipts:
                raise ContractError("Hybrid ledger corroboration lacks an applied receipt.")
            cutoff, proposal_ids, _ = directives[directive_id]
            if proposal_id not in proposal_ids or step_id <= cutoff:
                raise ContractError(
                    "Hybrid ledger corroboration is not independent future evidence."
                )
            use_key = (directive_id, proposal_id, step_id)
            if use_key in corroboration:
                raise ContractError("Hybrid ledger reused corroboration evidence.")
            if child.get("future_only") is not True:
                raise ContractError("Hybrid ledger corroboration must be future-only.")
            if child.get("evidence_family") != (
                "frozen_count_snapshot_cluster_log_predictive_ratio"
            ):
                raise ContractError("Hybrid ledger corroboration evidence family is invalid.")
            ratio = child.get("log_predictive_likelihood_ratio")
            if not isinstance(ratio, int | float) or not np.isfinite(float(ratio)):
                raise ContractError("Hybrid ledger predictive likelihood ratio is invalid.")
            if child.get("candidate_state") not in {
                "pending",
                "verified",
                "quarantined",
            }:
                raise ContractError("Hybrid ledger candidate state is invalid.")
            for name in ("future_score_sha256", "current_covered_records_sha256"):
                digest = child.get(name)
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ContractError(f"Hybrid ledger corroboration {name} is invalid.")
            if not isinstance(child.get("snapshot_id"), str) or not isinstance(
                child.get("snapshot_candidate_id"), str
            ):
                raise ContractError("Hybrid ledger corroboration snapshot identity is invalid.")
            corroboration.add(use_key)
    if payload["last_event_sha256"] != previous:
        raise ContractError("Hybrid ledger last_event_sha256 differs from the chain tail.")
    return HybridLedgerSummaryInfo(
        path=source,
        payload=MappingProxyType(payload),
        summary_sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


def _hybrid_cluster_projection(result: MLEResultInfo) -> list[dict[str, object]]:
    """Project the standalone MLE cluster contract into the hybrid report."""
    return [
        {
            "cluster_id": int(cluster["cluster_id"]),
            "isotope": str(cluster["isotope"]),
            "centroid_xyz": [float(value) for value in cluster["centroid_xyz"]],  # type: ignore[arg-type]
            "integrated_strength_cps_1m": float(cluster["integrated_strength_cps_1m"]),
            "surface_kinds": [str(value) for value in cluster["surface_kinds"]],  # type: ignore[arg-type]
            "patch_ids": [int(value) for value in cluster["patch_ids"]],  # type: ignore[arg-type]
        }
        for cluster in result.hotspot_clusters
    ]


def _ledger_candidate_state_counts(
    ledger: HybridLedgerSummaryInfo,
) -> dict[str, int]:
    """Derive proposal states solely from the validated append-only ledger."""
    states: dict[tuple[str, str], str] = {}
    events = ledger.payload["events"]
    assert isinstance(events, list)
    for event in events:
        assert isinstance(event, dict)
        child = event["payload"]
        assert isinstance(child, dict)
        if event["event_type"] == "directive_issued":
            directive_id = str(child["directive_id"])
            for proposal_id in child["proposal_ids"]:  # type: ignore[union-attr]
                states[(directive_id, str(proposal_id))] = "pending"
        elif event["event_type"] == "corroboration":
            key = (str(child["directive_id"]), str(child["proposal_id"]))
            if key not in states:
                raise ContractError("Ledger candidate state references an unknown proposal.")
            states[key] = str(child["candidate_state"])
    return {
        name: sum(value == name for value in states.values())
        for name in ("pending", "verified", "quarantined")
    }


def _require_cold_full_mle(
    result: MLEResultInfo,
    *,
    mode: str,
    expected_step_ids: tuple[int, ...] | None,
) -> None:
    """Require explicit MLE lineage proving an independent cold full-history fit."""
    if result.mode != mode:
        raise ContractError(f"Final {mode} MLE role is bound to an {result.mode!r} result.")
    nested = result.diagnostics["diagnostics"]
    assert isinstance(nested, dict)
    lineage = nested.get("causal_lineage")
    if not isinstance(lineage, dict):
        raise ContractError("Final MLE result lacks causal_lineage evidence.")
    if lineage.get("fit_kind") != "cold_start_all_history" or lineage.get("warm_start") is not None:
        raise ContractError("Final MLE result must be an independent cold all-history fit.")
    if expected_step_ids is not None:
        covered = tuple(int(value) for value in lineage.get("covered_step_ids", ()))
        if covered != expected_step_ids or int(lineage.get("record_count", -1)) != len(
            expected_step_ids
        ):
            raise ContractError("Final MLE result does not cover the complete MeasurementLog.")


def validate_hybrid_result(
    path: str | Path,
    *,
    expected_measurement_log: MeasurementLogInfo | None = None,
    expected_source_measurement_log: MeasurementLogInfo | None = None,
    expected_pf_result: PFResultInfo | None = None,
    expected_final_count_mle_result: MLEResultInfo | None = None,
    expected_final_spectral_mle_result: MLEResultInfo | None = None,
    expected_ledger: HybridLedgerSummaryInfo | None = None,
    expected_snapshots: tuple[MLESnapshotInfo, ...] | None = None,
    expected_directives: tuple[PFDirectiveInfo, ...] | None = None,
    expected_receipts: tuple[PFDirectiveReceiptInfo, ...] | None = None,
    expected_future_candidate_scores: tuple[FutureCandidateScoreInfo, ...] | None = None,
    expected_planning_recommendations: tuple[HybridPlanningRecommendationInfo, ...] | None = None,
    expected_verification_queue_sha256: str | None = None,
) -> HybridResultInfo:
    """Validate a complete causal hybrid result and optional referenced artifacts."""
    source = Path(path).resolve()
    payload = load_json(source)
    _validate_schema(payload, "hybrid_result_schema.json", label="HybridResult")

    measurement = payload["measurement_log"]
    artifacts = payload["artifacts"]
    report = payload["authoritative_report"]
    verification = payload["verification_summary"]
    safety = payload["safety"]
    assert isinstance(measurement, dict)
    assert isinstance(artifacts, dict)
    assert isinstance(report, dict)
    assert isinstance(verification, dict)
    assert isinstance(safety, dict)

    if int(measurement["source_record_count"]) != int(measurement["inference_record_count"]):
        raise ContractError(
            "Station-marker derivation may not change the MeasurementLog record count."
        )

    roles = payload["estimator_roles"]
    assert isinstance(roles, dict)
    count_role = roles["final_count_diagnostic"]
    report_role = roles["final_report"]
    assert isinstance(count_role, dict)
    assert isinstance(report_role, dict)
    if count_role["estimator_variant"] != "count" or count_role["role"] != "diagnostic":
        raise ContractError("Hybrid final count role must remain a non-authoritative diagnostic.")
    if (
        report_role["estimator_variant"] != "spectral"
        or report_role["role"] != "authoritative_final_report"
    ):
        raise ContractError("Hybrid final report role must be the cold spectral MLE.")

    snapshots = artifacts["snapshots"]
    directives = artifacts["directives"]
    receipts = artifacts["receipts"]
    future_scores = artifacts["future_candidate_scores"]
    planning_recommendations = artifacts["planning_recommendations"]
    assert isinstance(snapshots, list)
    assert isinstance(directives, list)
    assert isinstance(receipts, list)
    assert isinstance(future_scores, list)
    assert isinstance(planning_recommendations, list)

    snapshot_ids: dict[str, tuple[str, int]] = {}
    previous_cutoff = -1
    for snapshot in snapshots:
        assert isinstance(snapshot, dict)
        snapshot_id = str(snapshot["snapshot_id"])
        cutoff = int(snapshot["data_cutoff_step"])
        if snapshot_id in snapshot_ids or cutoff <= previous_cutoff:
            raise ContractError("Hybrid snapshot IDs must be unique with increasing cutoffs.")
        if cutoff > int(measurement["final_step_id"]):
            raise ContractError("Hybrid snapshot cutoff exceeds the final observation step.")
        snapshot_ids[snapshot_id] = (str(snapshot["sha256"]), cutoff)
        previous_cutoff = cutoff

    seen_score_refs: set[tuple[str, tuple[int, ...]]] = set()
    for score in future_scores:
        assert isinstance(score, dict)
        snapshot_id = str(score["snapshot_id"])
        if snapshot_id not in snapshot_ids:
            raise ContractError("Future score references an unknown MLE snapshot.")
        steps = tuple(int(value) for value in score["future_step_ids"])  # type: ignore[arg-type]
        key = (snapshot_id, steps)
        if key in seen_score_refs:
            raise ContractError("Hybrid result repeats a future-score artifact reference.")
        seen_score_refs.add(key)
        if steps[0] <= snapshot_ids[snapshot_id][1] or any(
            right <= left for left, right in pairwise(steps)
        ):
            raise ContractError("Hybrid future-score rows are not strictly post-cutoff.")

    recommendation_ids: set[str] = set()
    previous_planning_cutoff = -1
    for recommendation in planning_recommendations:
        assert isinstance(recommendation, dict)
        recommendation_id = str(recommendation["recommendation_id"])
        cutoff = int(recommendation["data_cutoff_step"])
        if recommendation_id in recommendation_ids or cutoff <= previous_planning_cutoff:
            raise ContractError(
                "Hybrid planning recommendations must have unique IDs and increasing cutoffs."
            )
        if cutoff > int(measurement["final_step_id"]):
            raise ContractError("Hybrid planning recommendation exceeds the final step.")
        if recommendation["robot_actuation_authorized"] is not False:
            raise ContractError("Hybrid planning recommendations may not authorize actuation.")
        recommendation_ids.add(recommendation_id)
        previous_planning_cutoff = cutoff

    directive_ids: dict[str, tuple[str, str]] = {}
    total_proposals = 0
    hybrid_mode = str(payload["hybrid_mode"])
    for directive in directives:
        assert isinstance(directive, dict)
        directive_id = str(directive["directive_id"])
        snapshot_id = str(directive["snapshot_id"])
        if directive_id in directive_ids:
            raise ContractError("Hybrid directive IDs must be unique.")
        if snapshot_id not in snapshot_ids:
            raise ContractError("Hybrid directive references an unknown snapshot.")
        if int(directive["data_cutoff_step"]) != snapshot_ids[snapshot_id][1]:
            raise ContractError("Hybrid directive cutoff differs from its snapshot.")
        if directive["directive_kind"] != hybrid_mode:
            raise ContractError("Hybrid directive kind differs from the declared hybrid mode.")
        directive_ids[directive_id] = (str(directive["sha256"]), str(directive["directive_kind"]))
        total_proposals += int(directive["proposal_count"])

    receipt_ids: set[str] = set()
    received_directives: set[str] = set()
    applied_mh = False
    for receipt in receipts:
        assert isinstance(receipt, dict)
        receipt_id = str(receipt["receipt_id"])
        directive_id = str(receipt["directive_id"])
        if receipt_id in receipt_ids or directive_id in received_directives:
            raise ContractError("Hybrid receipts must be unique and once-only per directive.")
        if directive_id not in directive_ids:
            raise ContractError("Hybrid receipt references an unknown directive.")
        receipt_ids.add(receipt_id)
        received_directives.add(directive_id)
        applied_mh |= hybrid_mode == "proposal_only_mh" and str(receipt["status"]) == "applied"
    if received_directives != set(directive_ids):
        raise ContractError("Every issued hybrid directive must have exactly one receipt.")

    counts = {name: int(verification[name]) for name in ("pending", "verified", "quarantined")}
    if int(verification["total"]) != sum(counts.values()):
        raise ContractError("Hybrid verification state counts do not sum to total.")
    if int(verification["total"]) != total_proposals:
        raise ContractError("Hybrid verification total differs from issued proposal count.")
    if bool(safety["target_preserving_fixed_cardinality_mh_performed"]) != applied_mh:
        raise ContractError("Hybrid MH safety claim differs from its applied receipts.")

    cluster_ids: set[int] = set()
    clusters = report["hotspot_clusters"]
    assert isinstance(clusters, list)
    for cluster in clusters:
        assert isinstance(cluster, dict)
        cluster_id = int(cluster["cluster_id"])
        if cluster_id in cluster_ids:
            raise ContractError("Authoritative hybrid-report cluster IDs must be unique.")
        cluster_ids.add(cluster_id)
    if report["result_sha256"] != artifacts["final_spectral_mle_result_sha256"]:
        raise ContractError("Authoritative report hash differs from the final spectral MLE hash.")

    if expected_measurement_log is not None:
        if payload["source_run_id"] != expected_measurement_log.manifest["run_id"]:
            raise ContractError("Hybrid result source run differs from MeasurementLog.")
        if (
            measurement["inference_measurement_log_sha256"]
            != expected_measurement_log.measurement_log_sha256
        ):
            raise ContractError("Hybrid inference MeasurementLog hash differs from its input.")
        if int(measurement["inference_record_count"]) != expected_measurement_log.record_count:
            raise ContractError("Hybrid inference record count differs from MeasurementLog.")
        if int(measurement["final_step_id"]) != expected_measurement_log.step_ids[-1]:
            raise ContractError("Hybrid result final step differs from MeasurementLog.")
    if expected_source_measurement_log is not None:
        if payload["source_run_id"] != expected_source_measurement_log.manifest["run_id"]:
            raise ContractError("Hybrid result source run differs from source MeasurementLog.")
        if (
            measurement["source_measurement_log_sha256"]
            != expected_source_measurement_log.measurement_log_sha256
        ):
            raise ContractError("Hybrid source MeasurementLog hash differs from its input.")
        if int(measurement["source_record_count"]) != expected_source_measurement_log.record_count:
            raise ContractError("Hybrid source record count differs from MeasurementLog.")
        if expected_measurement_log is not None and (
            expected_source_measurement_log.step_ids != expected_measurement_log.step_ids
        ):
            raise ContractError("Station-marker derivation may not change observation step IDs.")

    if expected_pf_result is not None:
        if artifacts["final_pf_result_sha256"] != expected_pf_result.result_sha256:
            raise ContractError("Hybrid result final PF hash differs from its validated bundle.")
    expected_steps = None if expected_measurement_log is None else expected_measurement_log.step_ids
    if expected_final_count_mle_result is not None:
        if (
            artifacts["final_count_mle_result_sha256"]
            != expected_final_count_mle_result.result_sha256
        ):
            raise ContractError("Hybrid result final count MLE hash differs from its bundle.")
        _require_cold_full_mle(
            expected_final_count_mle_result,
            mode="count",
            expected_step_ids=expected_steps,
        )
    if expected_final_spectral_mle_result is not None:
        if (
            artifacts["final_spectral_mle_result_sha256"]
            != expected_final_spectral_mle_result.result_sha256
        ):
            raise ContractError("Hybrid result final spectral MLE hash differs from its bundle.")
        _require_cold_full_mle(
            expected_final_spectral_mle_result,
            mode="spectral",
            expected_step_ids=expected_steps,
        )
        if clusters != _hybrid_cluster_projection(expected_final_spectral_mle_result):
            raise ContractError("Authoritative report does not mirror final spectral MLE clusters.")
        if expected_final_spectral_mle_result.diagnostics.get("converged") is not True:
            raise ContractError("Authoritative final spectral MLE did not converge.")
        if (
            report["objective_value"]
            != expected_final_spectral_mle_result.diagnostics["objective_value"]
            or report["poisson_deviance"]
            != expected_final_spectral_mle_result.diagnostics["poisson_deviance"]
        ):
            raise ContractError(
                "Authoritative report diagnostics differ from the final spectral MLE."
            )
    if expected_ledger is not None:
        if artifacts["hybrid_ledger_summary_sha256"] != expected_ledger.summary_sha256:
            raise ContractError("Hybrid result ledger hash differs from its validated ledger.")
        if int(artifacts["ledger_event_count"]) != int(expected_ledger.payload["event_count"]):
            raise ContractError("Hybrid result ledger event count differs from its ledger.")
        if artifacts["ledger_last_event_sha256"] != expected_ledger.payload["last_event_sha256"]:
            raise ContractError("Hybrid result ledger tail differs from its ledger.")
        if counts != _ledger_candidate_state_counts(expected_ledger):
            raise ContractError("Hybrid verification counts differ from ledger-derived states.")
    if (
        expected_verification_queue_sha256 is not None
        and artifacts["verification_queue_sha256"] != expected_verification_queue_sha256
    ):
        raise ContractError("Hybrid result verification queue hash differs from its artifact.")

    if expected_snapshots is not None:
        expected = [
            {
                "snapshot_id": str(snapshot.payload["snapshot_id"]),
                "sha256": snapshot.snapshot_sha256,
                "data_cutoff_step": snapshot.cutoff_step,
                "data_cutoff_station": int(snapshot.payload["data_cutoff_station"]),
                "warm_start_used": bool(snapshot.payload["warm_start"]["used"]),  # type: ignore[index]
            }
            for snapshot in sorted(expected_snapshots, key=lambda item: item.cutoff_step)
        ]
        if snapshots != expected:
            raise ContractError("Hybrid result snapshot references differ from validated files.")
    if expected_directives is not None:
        expected = [
            {
                "directive_id": str(directive.payload["directive_id"]),
                "sha256": directive.directive_sha256,
                "snapshot_id": str(directive.payload["snapshot_id"]),
                "data_cutoff_step": directive.cutoff_step,
                "directive_kind": str(directive.payload["directive_kind"]),
                "proposal_count": len(directive.payload["proposals"]),  # type: ignore[arg-type]
            }
            for directive in sorted(
                expected_directives,
                key=lambda item: (item.cutoff_step, str(item.payload["directive_id"])),
            )
        ]
        if directives != expected:
            raise ContractError("Hybrid result directive references differ from validated files.")
    if expected_receipts is not None:
        expected = [
            {
                "receipt_id": str(receipt.payload["receipt_id"]),
                "sha256": receipt.receipt_sha256,
                "directive_id": str(receipt.payload["directive_id"]),
                "status": str(receipt.payload["status"]),
            }
            for receipt in sorted(
                expected_receipts,
                key=lambda item: str(item.payload["directive_id"]),
            )
        ]
        if receipts != expected:
            raise ContractError("Hybrid result receipt references differ from validated files.")
    if expected_future_candidate_scores is not None:
        expected = [
            {
                "snapshot_id": str(score.payload["snapshot_id"]),
                "sha256": score.score_sha256,
                "future_step_ids": list(score.future_step_ids),
            }
            for score in sorted(
                expected_future_candidate_scores,
                key=lambda item: (
                    int(item.payload["snapshot_data_cutoff_step"]),
                    item.future_step_ids[-1],
                ),
            )
        ]
        if future_scores != expected:
            raise ContractError(
                "Hybrid result future-score references differ from validated files."
            )
        if expected_ledger is not None:
            score_rows: dict[tuple[str, str, int], set[tuple[int, float, str, str]]] = {}
            for score in expected_future_candidate_scores:
                snapshot_id = str(score.payload["snapshot_id"])
                hashes = score.payload["hashes"]
                assert isinstance(hashes, dict)
                candidates = score.payload["candidates"]
                assert isinstance(candidates, list)
                for candidate in candidates:
                    assert isinstance(candidate, dict)
                    candidate_id = str(candidate["snapshot_candidate_id"])
                    rows = candidate["future_step_scores"]
                    assert isinstance(rows, list)
                    for row in rows:
                        assert isinstance(row, dict)
                        key = (snapshot_id, candidate_id, int(row["step_id"]))
                        value = (
                            int(row["station_id"]),
                            float(row["log_predictive_likelihood_ratio"]),
                            score.score_sha256,
                            str(hashes["current_covered_records_sha256"]),
                        )
                        score_rows.setdefault(key, set()).add(value)
            ledger_events = expected_ledger.payload["events"]
            assert isinstance(ledger_events, list)
            for event in ledger_events:
                assert isinstance(event, dict)
                if event["event_type"] != "corroboration":
                    continue
                child = event["payload"]
                assert isinstance(child, dict)
                key = (
                    str(child["snapshot_id"]),
                    str(child["snapshot_candidate_id"]),
                    int(child["step_id"]),
                )
                expected_rows = score_rows.get(key, set())
                observed_row = (
                    int(child["station_id"]),
                    float(child["log_predictive_likelihood_ratio"]),
                    str(child["future_score_sha256"]),
                    str(child["current_covered_records_sha256"]),
                )
                if observed_row not in expected_rows:
                    raise ContractError(
                        "Ledger corroboration is not hash-bound to its future-score row."
                    )
    if expected_planning_recommendations is not None:
        expected = []
        for recommendation in sorted(
            expected_planning_recommendations,
            key=lambda item: int(
                item.payload["causal_boundary"]["data_cutoff_step"]  # type: ignore[index]
            ),
        ):
            boundary = recommendation.payload["causal_boundary"]
            provenance = recommendation.payload["provenance"]
            selected = recommendation.payload["selected_action"]
            assert isinstance(boundary, dict)
            assert isinstance(provenance, dict)
            assert isinstance(selected, dict)
            expected.append(
                {
                    "recommendation_id": str(recommendation.payload["recommendation_id"]),
                    "sha256": recommendation.recommendation_sha256,
                    "data_cutoff_step": int(boundary["data_cutoff_step"]),
                    "data_cutoff_station": int(boundary["data_cutoff_station"]),
                    "causal_planning_request_sha256": str(
                        provenance["causal_planning_request_sha256"]
                    ),
                    "selected_action": dict(selected),
                    "robot_actuation_authorized": False,
                }
            )
        if planning_recommendations != expected:
            raise ContractError("Hybrid result planning references differ from validated files.")

    return HybridResultInfo(
        path=source,
        payload=MappingProxyType(payload),
        result_sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


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
