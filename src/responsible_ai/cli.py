"""CLI: ``demo``, ``audit``, ``card``, ``serve``."""

from __future__ import annotations

import argparse
import json
import sys

from responsible_ai.audit_log import AuditLog
from responsible_ai.audit_report import (
    DEFAULT_GATES,
    build_slice_report,
    evaluate_gates,
)
from responsible_ai.config import Settings
from responsible_ai.fairness import summarize_fairness
from responsible_ai.generator import generate_synthetic
from responsible_ai.mitigation import adjust_thresholds_per_group
from responsible_ai.model_card import ModelCard


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def cmd_demo(_args: argparse.Namespace) -> int:
    data = generate_synthetic()
    summary = summarize_fairness(data.y_true, data.y_pred, data.groups)
    print("== Fairness summary (synthetic demo data) ==")
    _print_json(summary.as_dict())
    report = build_slice_report(
        data.y_true,
        data.y_pred,
        {"group": data.groups, "age_band": data.age_bands},
        intersect=("group", "age_band"),
    )
    print("\n== Gates ==")
    for result in evaluate_gates(report.overall, DEFAULT_GATES):
        print(f"  {result.message}")
    worst = max(
        report.intersections,
        key=lambda s: abs((s.true_positive_rate or 0) - 0.5),
        default=None,
    )
    if worst is not None:
        print(f"\nWorst intersection slice (by TPR distance): {worst.slice_id} (n={worst.n})")
    adjustments = adjust_thresholds_per_group(data.scores, data.y_true, data.groups)
    print("\n== Threshold adjustments (review-only, not applied) ==")
    for adj in adjustments:
        print(f"  {json.dumps(adj.as_dict(), sort_keys=True)}")
    log = AuditLog()
    log.append("demo_audit", {"n": len(data.y_true)})
    print(f"\nAudit log entries: {len(log)}, verified: {log.verify()}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    with open(args.input, encoding="utf-8") as fh:
        payload = json.load(fh)
    summary = summarize_fairness(
        payload["y_true"], payload["y_pred"], payload["groups"],
        min_group_size=args.min_group_size,
    )
    _print_json(summary.as_dict())
    return 0


def cmd_card(args: argparse.Namespace) -> int:
    with open(args.input, encoding="utf-8") as fh:
        payload = json.load(fh)
    card = ModelCard(**payload)
    if args.format == "json":
        print(card.to_json())
    else:
        print(card.to_markdown())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from responsible_ai.service import create_app

    app = create_app()
    if app is None:
        print("FastAPI is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1
    import uvicorn  # type: ignore[import]

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="responsible-ai")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Run the synthetic end-to-end demo")
    audit = sub.add_parser("audit", help="Fairness summary from a JSON file")
    audit.add_argument("--input", required=True, help="JSON with y_true, y_pred, groups")
    audit.add_argument("--min-group-size", type=int, default=30)
    card = sub.add_parser("card", help="Render a model card from a JSON file")
    card.add_argument("--input", required=True)
    card.add_argument("--format", choices=["markdown", "json"], default="markdown")
    serve = sub.add_parser("serve", help="Serve the HTTP API")
    settings = Settings.from_env()
    serve.add_argument("--host", default=settings.host)
    serve.add_argument("--port", type=int, default=settings.port)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "audit":
        return cmd_audit(args)
    if args.command == "card":
        return cmd_card(args)
    if args.command == "serve":
        return cmd_serve(args)
    parser.print_help()
    return 2
