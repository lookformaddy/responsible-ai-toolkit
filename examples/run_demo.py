"""End-to-end demo: synthetic data -> fairness summary -> slice report ->
gates -> threshold suggestions -> model card -> verified audit log."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from responsible_ai.audit_log import AuditLog
from responsible_ai.audit_report import DEFAULT_GATES, build_slice_report, evaluate_gates
from responsible_ai.drift import data_quality_screen, representation_drift
from responsible_ai.explainability import partial_dependence, permutation_importance
from responsible_ai.fairness import summarize_fairness
from responsible_ai.generator import generate_synthetic
from responsible_ai.mitigation import adjust_thresholds_per_group, reweighting_suggestions
from responsible_ai.model_card import ModelCard


def main() -> None:
    data = generate_synthetic()
    summary = summarize_fairness(data.y_true, data.y_pred, data.groups)
    print("== Fairness summary ==")
    print(f"  DPD={summary.demographic_parity_difference:.3f}  "
          f"DI={summary.disparate_impact}  "
          f"EOD={summary.equal_opportunity_difference}")
    for group, metrics in summary.per_group.items():
        print(f"  {group}: n={metrics.n} acc={metrics.accuracy:.3f} "
              f"TPR={metrics.true_positive_rate} FPR={metrics.false_positive_rate} "
              f"pos_rate={metrics.positive_rate:.3f}")

    report = build_slice_report(
        data.y_true, data.y_pred,
        {"group": data.groups, "age_band": data.age_bands},
        intersect=("group", "age_band"),
    )
    print("\n== Gates ==")
    for result in evaluate_gates(report.overall, DEFAULT_GATES):
        print(f"  {result.message}")

    print("\n== Threshold adjustments (review-only) ==")
    for adj in adjust_thresholds_per_group(data.scores, data.y_true, data.groups):
        print(f"  {adj.group}: {adj.threshold_before} -> {adj.threshold_after:.3f} "
              f"(TPR {adj.tpr_before} -> {adj.tpr_after}, hit_bound={adj.hit_bound})")

    print("\n== Reweighting suggestions (first 4) ==")
    for suggestion in reweighting_suggestions(data.y_true, data.groups)[:4]:
        print(f"  {suggestion.group} label={suggestion.label}: weight={suggestion.weight:.2f}")

    def predict(rows):
        return [0.35 + 0.4 * (row["income"] / 120.0) + 0.1 * (row["tenure"] / 20.0)
                for row in rows]

    print("\n== Permutation importance (toy scorer) ==")
    print(" ", permutation_importance(predict, data.features, data.y_true, n_repeats=3))
    print("== Partial dependence on income ==")
    print(" ", partial_dependence(predict, data.features, "income", [20.0, 70.0, 120.0]))

    print("\n== Drift screen (reference vs shifted production slice) ==")
    drift = representation_drift(data.groups, data.groups[:300] + ["group_c"] * 100)
    print(f"  TVD={drift.total_variation_distance:.3f} flagged={drift.flagged}")
    print("== Data-quality screen ==")
    print(" ", data_quality_screen(data.features).as_dict())

    card = ModelCard(
        model_name="demo-screen",
        version="1.0.0",
        author="responsible-ai-toolkit demo",
        date="2026-10-25",
        intended_use="Demonstration of the responsible AI toolkit on synthetic data.",
        limitations=["Synthetic data only; not validated on any real population."],
        training_data_summary=f"{len(data.y_true)} synthetic rows (seed 7).",
        evaluation_metrics={"accuracy": round(
            sum(1 for y, p in zip(data.y_true, data.y_pred) if y == p) / len(data.y_true), 4)},
        fairness_results=summary.as_dict(),
    )
    print("\n== Model card (first 400 chars) ==")
    print(card.to_markdown()[:400] + "...")

    log = AuditLog()
    log.append("demo_audit", {"n": len(data.y_true)})
    log.append("model_card_generated", {"model": card.model_name})
    print(f"\nAudit log verified: {log.verify()}")


if __name__ == "__main__":
    main()
