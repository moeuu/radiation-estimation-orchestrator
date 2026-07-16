from __future__ import annotations

from orchestrator.evaluation import SourceEstimate, match_sources, point_source_metrics


def test_matching_maximizes_cardinality_then_minimizes_distance() -> None:
    truth = [
        SourceEstimate("Cs-137", (0.0, 0.0, 0.0), 10.0, identifier="a"),
        SourceEstimate("Cs-137", (1.0, 0.0, 0.0), 10.0, identifier="b"),
    ]
    estimates = [
        SourceEstimate("Cs-137", (0.1, 0.0, 0.0), 10.0, identifier="x"),
        SourceEstimate("Cs-137", (1.1, 0.0, 0.0), 10.0, identifier="y"),
        SourceEstimate("Co-60", (0.0, 0.0, 0.0), 10.0, identifier="wrong-isotope"),
    ]
    matches = match_sources(truth, estimates, radius_m=0.5)
    assert [(item.truth_index, item.estimate_index) for item in matches] == [(0, 0), (1, 1)]


def test_point_metrics_include_ceiling_recall_and_strength_error() -> None:
    truth = [SourceEstimate("Co-60", (1.0, 1.0, 3.0), 100.0, ("ceiling",), "truth")]
    estimate = [SourceEstimate("Co-60", (1.0, 1.0, 2.9), 90.0, identifier="estimate")]
    metrics = point_source_metrics(truth, estimate, radius_m=0.5, ceiling_z_m=3.0)
    assert metrics["source_precision"] == 1.0
    assert metrics["source_recall"] == 1.0
    assert metrics["ceiling_source_recall"] == 1.0
    assert metrics["integrated_strength_error_cps_1m"]["total_absolute_error"] == 10.0


def test_strength_error_cannot_cancel_between_isotopes() -> None:
    truth = [
        SourceEstimate("Cs-137", (0.0, 0.0, 0.0), 100.0, identifier="truth-cs"),
        SourceEstimate("Co-60", (1.0, 0.0, 0.0), 100.0, identifier="truth-co"),
    ]
    estimates = [
        SourceEstimate("Cs-137", (0.0, 0.0, 0.0), 150.0, identifier="estimate-cs"),
        SourceEstimate("Co-60", (1.0, 0.0, 0.0), 50.0, identifier="estimate-co"),
    ]
    strength = point_source_metrics(truth, estimates, radius_m=0.5, ceiling_z_m=3.0)[
        "integrated_strength_error_cps_1m"
    ]
    assert strength["isotope_total_absolute_error"] == 100.0
    assert strength["per_isotope"]["Cs-137"]["total_absolute_error"] == 50.0
    assert strength["per_isotope"]["Co-60"]["total_absolute_error"] == 50.0


def test_strength_error_includes_all_unmatched_mass() -> None:
    truth = [
        SourceEstimate("Cs-137", (0.0, 0.0, 0.0), 100.0, identifier="matched-truth"),
        SourceEstimate("Cs-137", (10.0, 0.0, 0.0), 40.0, identifier="missed-truth"),
    ]
    estimates = [
        SourceEstimate("Cs-137", (0.0, 0.0, 0.0), 90.0, identifier="matched-estimate"),
        SourceEstimate("Cs-137", (20.0, 0.0, 0.0), 30.0, identifier="false-estimate"),
    ]
    strength = point_source_metrics(truth, estimates, radius_m=0.5, ceiling_z_m=3.0)[
        "integrated_strength_error_cps_1m"
    ]
    assert strength["matched_mae"] == 10.0
    assert strength["unmatched_truth_strength"] == 40.0
    assert strength["unmatched_estimate_strength"] == 30.0
    assert strength["assignment_absolute_error"] == 80.0
    assert strength["total_absolute_error"] == 80.0
