# responsible-ai-toolkit

A responsible AI toolkit for production ML, built in pure Python standard
library: fairness metrics, bias audit reports with slice and intersection
analysis, model-agnostic explainability, structured model cards,
tamper-evident audit logging, fairness-aware drift screens, and
transparent (never silent) mitigation aids -- plus an optional FastAPI
surface.

**Track:** ML Engineering · **Catalog #8** of the enterprise portfolio series.

## The enterprise problem

A model that ships without a fairness audit is a governance incident
waiting for a headline. Production ML teams need to answer, for every
model version:

- **Is it fair across groups -- and across intersections?** Aggregate
  fairness hides harm: a model can look fair overall while failing one
  intersection of group and age band.
- **Would the audit survive scrutiny?** Undefined rates must not be
  silently scored as zero, small groups must not be silently dropped, and
  every evaluation must be logged in a way tampering would break.
- **Can a non-technical reviewer read the verdict?** A model card that
  embeds the fairness numbers verbatim, and gates that say PASS / FAIL /
  UNDEFINED in plain language.
- **Is fairness degrading over time?** A model that never changes still
  gets less fair when the production population drifts away from the
  audited reference.
- **What do we do about a failing audit?** Mitigation must be a reviewed
  decision with visible trade-offs, never a silent patch applied by a
  library.

This repo implements all of it, clean-room, with synthetic data only.

## Architecture overview

```
predictions + labels + groups
        │
        ├─▶ fairness ──▶ per-group metrics + cross-group summary
        │                  (DPD/DPR, disparate impact, equal opportunity,
        │                   equalized odds, predictive parity, Wilson CIs)
        ├─▶ audit_report ──▶ slices + intersections, worst-slice flagging,
        │                     configurable pass/fail gates
        ├─▶ explainability ──▶ permutation importance, partial dependence
        ├─▶ model_card ──▶ Markdown + JSON, fairness embedded verbatim
        ├─▶ audit_log ──▶ hash-chained, tamper-evident evaluation log
        ├─▶ drift ──▶ representation drift, label-rate shifts, data quality
        └─▶ mitigation ──▶ threshold adjustment (bounded), reweighting
                            suggestions -- review-only, never applied
```

See [docs/architecture.md](docs/architecture.md) for the full component map,
data flow, and design trade-offs.

## How to run it

```bash
python -m venv /tmp/venv && source /tmp/venv/bin/activate
pip install -r requirements.txt
pip install -e .

# run the tests (API tests auto-skip without FastAPI)
pytest -q

# end-to-end demo: synthetic data, audit, gates, mitigation, model card
python -m responsible_ai demo
# or: make demo / python examples/run_demo.py

# fairness summary from a JSON file (y_true, y_pred, groups)
python -m responsible_ai audit --input eval.json

# render a model card from a JSON file
python -m responsible_ai card --input card.json --format markdown

# serve the HTTP API
python -m responsible_ai serve            # -> http://127.0.0.1:8000
# POST /fairness  {"y_true","y_pred","groups","min_group_size"}
# POST /report    {"y_true","y_pred","groups","attributes"}
# POST /model-card {"model_name","version","author","date","intended_use",...}
# POST /drift     {"reference_groups","production_groups"}
# GET  /audit/verify  GET /health
```

Docker:

```bash
docker build -t responsible-ai-toolkit .
docker run -p 8000:8000 responsible-ai-toolkit
```

Configuration is environment-driven; copy `.env.example` to `.env` to
override the minimum group size, drift/label-shift flag thresholds, host,
port, and audit-log path.

## Key design decisions and trade-offs

- **Pure stdlib core.** Every metric is hand-computed and traceable to
  code a reviewer can read in one sitting; no numpy/pandas, no model
  framework dependency. The toolkit works on predictions and labels, not
  on a model object, so it audits any model.
- **Undefined rates stay `None`, never 0.** A TPR with no positive labels
  is not 0 -- it does not exist. Gates return UNDEFINED for such metrics,
  which is a finding, not a pass.
- **Small groups are flagged, never dropped.** A slice below the minimum
  group size still appears in the report with its warning attached; a
  flagged small slice winning worst-slice ranking is a signal to collect
  more data, not to ignore it.
- **The four-fifths rule is a screen, not a legal finding.** A disparate
  impact below 0.8 triggers investigation; it does not establish
  discrimination, which depends on context, sample size, and justification.
- **The audit log detects tampering, it does not prevent it.** Each entry
  hashes the previous entry's hash, so any edit, deletion, or reorder
  breaks verification at exactly one index. Production use anchors
  periodic hashes externally or writes to an append-only store.
- **Mitigation is decision support, not a fix.** Threshold adjustment
  equalizes opportunity at the cost of overall accuracy and false
  positive rates; reweighting changes training emphasis. Both return data
  structures for human review and re-audit -- neither mutates a model.
- **Partial dependence describes the model, not the world.** Averaging
  predictions over counterfactual rows misleads when features are
  strongly correlated; the helper documents this and stays numeric-only.
- **FastAPI is optional.** The service module import-guards FastAPI and
  pydantic; without them the library, CLI, and non-API tests all work.

## What it demonstrates

For a Director-level interview, this repo is evidence of: shipping
governance as code rather than as a slide deck (metrics, gates, model
cards, audit logs as one coherent system); implementing fairness math
from first principles with honest handling of undefined and small-sample
cases; designing for the reviewer, not just the metric (verbatim numbers,
plain-language gates, visible trade-offs); operating a model responsibly
over time (drift and data-quality screens with a fairness lens); and
knowing where a tool's authority ends (mitigation aids that refuse to
silently patch a model, and an audit log whose limits are stated openly).

## CI

The CI definition ships at [docs/github-actions-ci.yml](docs/github-actions-ci.yml).
It activates by moving it to `.github/workflows/ci.yml` (kept out of
`.github/workflows/` here because the GitHub App used for publishing lacks
the Workflows permission).

## Clean-room note

All code is original and all data is synthetic (generic fictional
domains, synthetic group labels). No employer code, data, or internal
designs are included.
