from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from orchestrator.contracts import (
    MeasurementLogInfo,
    validate_hybrid_ledger_summary,
    validate_measurement_log,
    validate_mle_snapshot_v2,
    validate_pf_directive,
    validate_pf_directive_receipt,
)
from orchestrator.errors import ContractError, DataReuseError
from orchestrator.hashing import load_json, write_json_atomic
from orchestrator.hybrid import (
    CandidateOutcome,
    CandidateState,
    DirectiveProposal,
    HybridConfig,
    HybridMode,
    HybridScheduler,
    HybridTrigger,
    MeasurementPrefix,
    MLESnapshot,
    ObservationUseLedger,
    PFDirectiveReceipt,
    PredictiveMonitor,
    SnapshotCluster,
    SnapshotPrediction,
    StationBoundarySchedule,
    TriggerReason,
    VerificationQueue,
    build_pf_directive,
)
from orchestrator.hybrid.controller import HybridController


@pytest.fixture()
def measurement_log(measurement_log_path: Path) -> MeasurementLogInfo:
    return validate_measurement_log(measurement_log_path)


def _schedule(log: MeasurementLogInfo) -> StationBoundarySchedule:
    return StationBoundarySchedule.create(
        source_run_id=str(log.manifest["run_id"]),
        station_end_steps=((0, 2), (1, 5), (2, 8), (3, 11)),
    )


def _prefix(
    log: MeasurementLogInfo, *, schedule: StationBoundarySchedule | None = None
) -> MeasurementPrefix:
    return MeasurementPrefix.from_measurement_log(
        log,
        cutoff_step=5,
        prefix_measurement_log_sha256="a" * 64,
        covered_records_sha256="b" * 64,
        station_boundaries=schedule or _schedule(log),
        station_complete_marker=True,
    )


def _trigger() -> HybridTrigger:
    return HybridTrigger(
        trigger_id="trigger-test",
        data_cutoff_step=5,
        data_cutoff_station=1,
        station_complete=True,
        reasons=(TriggerReason.STATION_INTERVAL,),
        normalized_predictive_deviance=0.5,
    )


def _cluster() -> SnapshotCluster:
    return SnapshotCluster(
        snapshot_candidate_id="candidate-1",
        cluster_id=1,
        isotope="Cs-137",
        centroid_xyz=(1.0, 2.0, 3.0),
        integrated_strength_cps_1m=25.0,
        surface_kinds=("wall",),
        patch_ids=(4, 5),
    )


def _snapshot(
    log: MeasurementLogInfo, *, schedule: StationBoundarySchedule | None = None
) -> MLESnapshot:
    prefix = _prefix(log, schedule=schedule)
    predictions = tuple(
        SnapshotPrediction(step_id=step, isotope_counts={"Cs-137": 2.0 + step})
        for step in prefix.covered_step_ids
    )
    return MLESnapshot.create(
        trigger=_trigger(),
        prefix=prefix,
        estimator_variant="count",
        mle_result_sha256="c" * 64,
        clusters=(_cluster(),),
        predictions=predictions,
        fit_diagnostics={"converged": True},
        provenance={"estimator_commit": "d" * 40},
    )


def _verification_directive(log: MeasurementLogInfo):
    snapshot = _snapshot(log)
    proposal = DirectiveProposal.from_cluster(_cluster())
    directive = build_pf_directive(
        snapshot,
        config=HybridConfig(mode=HybridMode.VERIFICATION_ONLY),
        pf_resolved_config_sha256="e" * 64,
        proposals=(proposal,),
        provenance={"controller_commit": "f" * 40},
    )
    return snapshot, directive


def test_hybrid_config_is_fail_closed() -> None:
    verification = HybridConfig(mode=HybridMode.VERIFICATION_ONLY)
    assert verification.capabilities.register_verification_candidates
    assert not verification.capabilities.submit_external_proposals
    proposal = HybridConfig(mode=HybridMode.PROPOSAL_ONLY_MH)
    assert proposal.capabilities.require_target_preserving_mh
    assert not proposal.capabilities.direct_mle_objective_reweight
    assert not proposal.capabilities.hard_prune
    with pytest.raises(ValueError, match="reweighting"):
        HybridConfig(allow_direct_mle_objective_reweight=True)
    with pytest.raises(ValueError, match="hard pruning"):
        HybridConfig(allow_hard_prune=True)


def test_prefix_requires_predeclared_station_complete_boundary(
    measurement_log: MeasurementLogInfo,
) -> None:
    schedule = _schedule(measurement_log)
    with pytest.raises(DataReuseError, match="station-complete marker"):
        MeasurementPrefix.from_measurement_log(
            measurement_log,
            cutoff_step=5,
            prefix_measurement_log_sha256="a" * 64,
            covered_records_sha256="b" * 64,
            station_boundaries=schedule,
            station_complete_marker=False,
        )
    with pytest.raises(DataReuseError, match="terminal step"):
        MeasurementPrefix.from_measurement_log(
            measurement_log,
            cutoff_step=4,
            prefix_measurement_log_sha256="a" * 64,
            covered_records_sha256="b" * 64,
            station_boundaries=schedule,
            station_complete_marker=True,
        )


def test_snapshot_and_directive_identity_ignore_unseen_schedule_suffix(
    measurement_log: MeasurementLogInfo,
) -> None:
    run_id = str(measurement_log.manifest["run_id"])
    schedule_a = StationBoundarySchedule.create(
        source_run_id=run_id,
        station_end_steps=((0, 2), (1, 5), (2, 8)),
    )
    schedule_b = StationBoundarySchedule.create(
        source_run_id=run_id,
        station_end_steps=((0, 2), (1, 5), (2, 9)),
    )
    snapshot_a = _snapshot(measurement_log, schedule=schedule_a)
    snapshot_b = _snapshot(measurement_log, schedule=schedule_b)
    assert snapshot_a.prefix.station_boundary_schedule_sha256 != (
        snapshot_b.prefix.station_boundary_schedule_sha256
    )
    assert snapshot_a.prefix.covered_station_boundaries_sha256 == (
        snapshot_b.prefix.covered_station_boundaries_sha256
    )
    assert snapshot_a.to_dict() == snapshot_b.to_dict()
    proposal = DirectiveProposal.from_cluster(_cluster())
    kwargs = {
        "config": HybridConfig(),
        "pf_resolved_config_sha256": "e" * 64,
        "proposals": (proposal,),
        "provenance": {},
    }
    directive_a = build_pf_directive(snapshot_a, **kwargs)
    directive_b = build_pf_directive(snapshot_b, **kwargs)
    assert directive_a.to_dict() == directive_b.to_dict()
    assert "source_measurement_log_sha256" not in directive_a.to_dict()


def test_snapshot_v2_enforces_exact_prediction_prefix(
    measurement_log: MeasurementLogInfo, tmp_path: Path
) -> None:
    snapshot = _snapshot(measurement_log)
    path = write_json_atomic(tmp_path / "snapshot.json", snapshot.to_dict())
    info = validate_mle_snapshot_v2(
        path,
        expected_covered_step_ids=tuple(range(6)),
        expected_source_run_id="shared-small-run-v1",
        expected_prefix_log_sha256="a" * 64,
        expected_covered_records_sha256="b" * 64,
    )
    assert info.snapshot_sha256 == snapshot.sha256

    payload = load_json(path)
    predictions = payload["predicted_observations"]
    assert isinstance(predictions, list)
    predictions.pop()
    invalid = write_json_atomic(tmp_path / "invalid-snapshot.json", payload)
    with pytest.raises(ContractError, match="exactly cover"):
        validate_mle_snapshot_v2(invalid)


def test_scheduler_uses_only_explicit_station_terminal_signals() -> None:
    config = HybridConfig(station_interval=2, predictive_deviance_threshold=None)
    scheduler = HybridScheduler(config)
    monitor = PredictiveMonitor()
    incomplete = monitor.record(
        step_id=0,
        station_id=0,
        prediction_data_cutoff_step=-1,
        station_complete_marker=False,
        observed_counts={"Cs-137": 2.0},
        predicted_counts={"Cs-137": 1.0},
    )
    assert scheduler.consider(incomplete) is None
    first = monitor.record(
        step_id=2,
        station_id=0,
        prediction_data_cutoff_step=1,
        station_complete_marker=True,
        observed_counts={"Cs-137": 2.0},
        predicted_counts={"Cs-137": 1.0},
    )
    assert scheduler.consider(first) is None
    second = monitor.record(
        step_id=5,
        station_id=1,
        prediction_data_cutoff_step=4,
        station_complete_marker=True,
        observed_counts={"Cs-137": 2.0},
        predicted_counts={"Cs-137": 1.0},
    )
    trigger = scheduler.consider(second)
    assert trigger is not None
    assert trigger.data_cutoff_step == 5
    assert trigger.reasons == (TriggerReason.STATION_INTERVAL,)


def test_scheduler_accumulates_nonterminal_mismatch_until_station_completion() -> None:
    scheduler = HybridScheduler(
        HybridConfig(
            station_interval=99,
            predictive_deviance_threshold=20.0,
            predictive_mismatch_streak=1,
        )
    )
    monitor = PredictiveMonitor()
    anomalous_view = monitor.record(
        step_id=0,
        station_id=0,
        prediction_data_cutoff_step=-1,
        station_complete_marker=False,
        observed_counts={"Cs-137": 20.0},
        predicted_counts={"Cs-137": 1.0},
    )
    assert scheduler.consider(anomalous_view) is None
    terminal_view = monitor.record(
        step_id=1,
        station_id=0,
        prediction_data_cutoff_step=0,
        station_complete_marker=True,
        observed_counts={"Cs-137": 1.0},
        predicted_counts={"Cs-137": 1.0},
    )

    trigger = scheduler.consider(terminal_view)

    assert trigger is not None
    assert trigger.data_cutoff_step == 1
    assert trigger.reasons == (TriggerReason.PREDICTIVE_MISMATCH,)
    assert trigger.normalized_predictive_deviance > 20.0


def test_scheduler_station_mismatch_is_invariant_to_view_order() -> None:
    def run_station(views: tuple[tuple[float, float], ...]) -> HybridTrigger:
        scheduler = HybridScheduler(
            HybridConfig(
                station_interval=99,
                predictive_deviance_threshold=2.0,
                predictive_mismatch_streak=1,
            )
        )
        monitor = PredictiveMonitor()
        trigger = None
        for step_id, (observed, predicted) in enumerate(views):
            signal = monitor.record(
                step_id=step_id,
                station_id=0,
                prediction_data_cutoff_step=step_id - 1,
                station_complete_marker=step_id == len(views) - 1,
                observed_counts={"Cs-137": observed},
                predicted_counts={"Cs-137": predicted},
            )
            trigger = scheduler.consider(signal)
        assert trigger is not None
        return trigger

    views = ((12.0, 2.0), (3.0, 3.0), (8.0, 1.0))
    assert run_station(views) == run_station((views[2], views[0], views[1]))


def test_predictive_monitor_requires_pre_observation_prediction() -> None:
    monitor = PredictiveMonitor()
    with pytest.raises(DataReuseError, match="frozen before"):
        monitor.record(
            step_id=3,
            station_id=1,
            prediction_data_cutoff_step=3,
            station_complete_marker=True,
            observed_counts={"Cs-137": 2.0},
            predicted_counts={"Cs-137": 1.0},
        )


def test_proposal_only_mh_requires_position_kernel_and_pf_config(
    measurement_log: MeasurementLogInfo, tmp_path: Path
) -> None:
    snapshot = _snapshot(measurement_log)
    config = HybridConfig(mode=HybridMode.PROPOSAL_ONLY_MH)
    missing_kernel = DirectiveProposal.from_cluster(_cluster())
    with pytest.raises(ContractError, match="density-defined"):
        build_pf_directive(
            snapshot,
            config=config,
            pf_resolved_config_sha256="e" * 64,
            proposals=(missing_kernel,),
            provenance={},
        )
    proposal = DirectiveProposal.from_cluster(
        _cluster(),
        proposal_kernel={
            "family": "defensive_truncated_gaussian_position",
            "position_sigma_xyz_m": [0.2, 0.2, 0.4],
            "defensive_weight": 0.1,
            "candidate_weight": 1.0,
        },
    )
    directive = build_pf_directive(
        snapshot,
        config=config,
        pf_resolved_config_sha256="e" * 64,
        proposals=(proposal,),
        provenance={},
    )
    snapshot_path = write_json_atomic(tmp_path / "snapshot.json", snapshot.to_dict())
    snapshot_info = validate_mle_snapshot_v2(snapshot_path)
    directive_path = write_json_atomic(tmp_path / "directive.json", directive.to_dict())
    directive_info = validate_pf_directive(directive_path, expected_snapshot=snapshot_info)
    assert directive_info.payload["pf_resolved_config_sha256"] == "e" * 64
    kernel = directive_info.payload["proposals"][0]["proposal_kernel"]  # type: ignore[index]
    assert kernel["family"] == "defensive_truncated_gaussian_position"  # type: ignore[index]


def test_receipt_ledger_and_queue_enforce_once_only_future_evidence(
    measurement_log: MeasurementLogInfo, tmp_path: Path
) -> None:
    snapshot, directive = _verification_directive(measurement_log)
    proposal = directive.proposals[0]
    receipt = PFDirectiveReceipt.create(
        directive=directive,
        consumer_variant="pf_hybrid_verification",
        status="applied",
        pf_state_sha256_before="1" * 64,
        pf_state_sha256_after="1" * 64,
        outcomes=(CandidateOutcome(proposal.proposal_id, "registered"),),
        provenance={"consumer_commit": "2" * 40},
    )
    snapshot_info = validate_mle_snapshot_v2(
        write_json_atomic(tmp_path / "snapshot.json", snapshot.to_dict())
    )
    directive_info = validate_pf_directive(
        write_json_atomic(tmp_path / "directive.json", directive.to_dict()),
        expected_snapshot=snapshot_info,
    )
    validate_pf_directive_receipt(
        write_json_atomic(tmp_path / "receipt.json", receipt.to_dict()),
        expected_directive=directive_info,
    )

    queue = VerificationQueue(
        HybridConfig(
            verification_min_future_observations=2,
            verification_support_log_predictive_ratio=2.0,
        )
    )
    queue.register(directive)
    with pytest.raises(DataReuseError, match="strictly after"):
        queue.corroborate(
            directive_id=directive.directive_id,
            proposal_id=proposal.proposal_id,
            step_id=5,
            log_predictive_likelihood_ratio=1.0,
        )
    pending = queue.corroborate(
        directive_id=directive.directive_id,
        proposal_id=proposal.proposal_id,
        step_id=6,
        log_predictive_likelihood_ratio=1.0,
    )
    assert pending.state is CandidateState.PENDING
    verified = queue.corroborate(
        directive_id=directive.directive_id,
        proposal_id=proposal.proposal_id,
        step_id=7,
        log_predictive_likelihood_ratio=1.5,
    )
    assert verified.state is CandidateState.VERIFIED

    ledger = ObservationUseLedger(
        source_run_id=snapshot.prefix.source_run_id,
        station_boundary_schedule_sha256=snapshot.prefix.station_boundary_schedule_sha256,
    )
    ledger.register_snapshot(snapshot)
    ledger.issue_directive(directive)
    ledger.record_receipt(receipt)
    with pytest.raises(DataReuseError, match="already has"):
        ledger.record_receipt(receipt)
    with pytest.raises(DataReuseError, match="strictly after"):
        ledger.record_corroboration(
            directive_id=directive.directive_id,
            proposal_id=proposal.proposal_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_candidate_id=proposal.snapshot_candidate_id,
            step_id=5,
            station_id=1,
            log_predictive_likelihood_ratio=1.0,
            future_score_sha256="a" * 64,
            current_covered_records_sha256="b" * 64,
            state="pending",
        )
    ledger.record_corroboration(
        directive_id=directive.directive_id,
        proposal_id=proposal.proposal_id,
        snapshot_id=snapshot.snapshot_id,
        snapshot_candidate_id=proposal.snapshot_candidate_id,
        step_id=6,
        station_id=2,
        log_predictive_likelihood_ratio=1.0,
        future_score_sha256="a" * 64,
        current_covered_records_sha256="b" * 64,
        state="pending",
    )
    ledger_path = write_json_atomic(tmp_path / "ledger.json", ledger.summary())
    validate_hybrid_ledger_summary(ledger_path)

    tampered = load_json(ledger_path)
    events = tampered["events"]
    assert isinstance(events, list)
    events[0]["payload"]["data_cutoff_step"] = 4  # type: ignore[index]
    tampered_path = write_json_atomic(tmp_path / "tampered-ledger.json", tampered)
    with pytest.raises(ContractError, match="content hash"):
        validate_hybrid_ledger_summary(tampered_path)


def test_receipt_rejects_mh_claim_without_acceptance_evidence(
    measurement_log: MeasurementLogInfo,
) -> None:
    snapshot = _snapshot(measurement_log)
    proposal = DirectiveProposal.from_cluster(
        _cluster(),
        proposal_kernel={
            "family": "defensive_truncated_gaussian_position",
            "position_sigma_xyz_m": [0.2, 0.2, 0.4],
            "defensive_weight": 0.1,
            "candidate_weight": 1.0,
        },
    )
    directive = build_pf_directive(
        snapshot,
        config=HybridConfig(mode=HybridMode.PROPOSAL_ONLY_MH),
        pf_resolved_config_sha256="e" * 64,
        proposals=(proposal,),
        provenance={},
    )
    with pytest.raises(ContractError, match="aggregate MH counts"):
        PFDirectiveReceipt.create(
            directive=directive,
            consumer_variant="pf_hybrid_proposal_only_mh",
            status="applied",
            pf_state_sha256_before="1" * 64,
            pf_state_sha256_after="2" * 64,
            outcomes=(CandidateOutcome(proposal.proposal_id, "mh_accepted"),),
            provenance={},
        )


def test_applied_mh_receipt_allows_an_unsampled_mixture_component(
    measurement_log: MeasurementLogInfo, tmp_path: Path
) -> None:
    snapshot = _snapshot(measurement_log)
    proposal = DirectiveProposal.from_cluster(
        _cluster(),
        proposal_kernel={
            "family": "defensive_truncated_gaussian_position",
            "position_sigma_xyz_m": [0.2, 0.2, 0.4],
            "defensive_weight": 0.1,
            "candidate_weight": 1.0,
        },
    )
    directive = build_pf_directive(
        snapshot,
        config=HybridConfig(mode=HybridMode.PROPOSAL_ONLY_MH),
        pf_resolved_config_sha256="e" * 64,
        proposals=(proposal,),
        provenance={},
    )
    receipt = PFDirectiveReceipt.create(
        directive=directive,
        consumer_variant="pf_external_relocation_mwg_v1",
        status="applied",
        pf_state_sha256_before="1" * 64,
        pf_state_sha256_after="1" * 64,
        outcomes=(CandidateOutcome(proposal.proposal_id, "not_applied"),),
        provenance={},
    )
    snapshot_info = validate_mle_snapshot_v2(
        write_json_atomic(tmp_path / "unsampled-snapshot.json", snapshot.to_dict())
    )
    directive_info = validate_pf_directive(
        write_json_atomic(tmp_path / "unsampled-directive.json", directive.to_dict()),
        expected_snapshot=snapshot_info,
    )
    validated = validate_pf_directive_receipt(
        write_json_atomic(tmp_path / "unsampled-receipt.json", receipt.to_dict()),
        expected_directive=directive_info,
    )
    assert validated.payload["candidate_outcomes"][0]["outcome"] == "not_applied"  # type: ignore[index]


def test_quarantined_candidate_cannot_change_active_planner_weight(
    measurement_log: MeasurementLogInfo,
) -> None:
    active = _cluster()
    rejected = replace(
        active,
        snapshot_candidate_id="candidate-2",
        cluster_id=2,
        integrated_strength_cps_1m=1_000_000.0,
        centroid_xyz=(3.0, 2.0, 1.0),
    )
    snapshot = replace(_snapshot(measurement_log), clusters=(active, rejected))
    kernel = {
        "family": "defensive_truncated_gaussian_position",
        "position_sigma_xyz_m": [0.2, 0.2, 0.4],
        "defensive_weight": 0.1,
        "candidate_weight": 1.0,
    }
    directive = build_pf_directive(
        snapshot,
        config=HybridConfig(mode=HybridMode.PROPOSAL_ONLY_MH),
        pf_resolved_config_sha256="e" * 64,
        proposals=(
            DirectiveProposal.from_cluster(active, proposal_kernel=kernel),
            DirectiveProposal.from_cluster(rejected, proposal_kernel=kernel),
        ),
        provenance={},
    )
    queue = VerificationQueue(HybridConfig(mode=HybridMode.PROPOSAL_ONLY_MH))
    registered = queue.register(directive)
    queue.corroborate(
        directive_id=directive.directive_id,
        proposal_id=registered[1].proposal_id,
        step_id=6,
        log_predictive_likelihood_ratio=-4.0,
    )

    modes = HybridController._planner_external_modes(queue, (directive,))
    active_mode = next(item for item in modes if item["verification_state"] == "pending")
    quarantined_mode = next(item for item in modes if item["verification_state"] == "quarantined")
    assert active_mode["weight"] == 1.0
    assert quarantined_mode["weight"] == 1e-12


def test_pf_proposal_domain_clips_only_surface_roundoff(
    measurement_log: MeasurementLogInfo,
) -> None:
    roundoff = replace(_cluster(), centroid_xyz=(2.0, 6.000000000000001, 1.0))
    outside = replace(_cluster(), centroid_xyz=(2.0, 6.01, 1.0))
    assert HybridController._pf_domain_candidate_mean(roundoff, measurement_log) == (2.0, 6.0, 1.0)
    assert HybridController._pf_domain_candidate_mean(outside, measurement_log) is None


def test_prefix_snapshot_rejects_missing_covered_prediction(
    measurement_log: MeasurementLogInfo,
) -> None:
    prefix = _prefix(measurement_log)
    predictions = tuple(
        SnapshotPrediction(step_id=step, isotope_counts={"Cs-137": 1.0})
        for step in prefix.covered_step_ids[:-1]
    )
    with pytest.raises(DataReuseError, match="exact declared prefix"):
        MLESnapshot.create(
            trigger=_trigger(),
            prefix=prefix,
            estimator_variant="count",
            mle_result_sha256="c" * 64,
            clusters=(_cluster(),),
            predictions=predictions,
            fit_diagnostics={},
            provenance={},
        )


def test_snapshot_warm_start_is_all_or_nothing(measurement_log: MeasurementLogInfo) -> None:
    snapshot = _snapshot(measurement_log)
    with pytest.raises(ContractError, match="supplied together"):
        MLESnapshot.create(
            trigger=_trigger(),
            prefix=snapshot.prefix,
            estimator_variant="count",
            mle_result_sha256="c" * 64,
            clusters=snapshot.clusters,
            predictions=snapshot.predictions,
            fit_diagnostics={},
            provenance={},
            warm_start_snapshot_id="old-snapshot",
        )


def test_config_dataclass_remains_immutable() -> None:
    config = HybridConfig()
    with pytest.raises(ValueError):
        replace(config, require_future_only_corroboration=False)
