# Architecture

## Component map

| Module | Responsibility |
| --- | --- |
| `fairness.py` | Per-group confusion metrics + cross-group summary (DPD/DPR, disparate impact, equal opportunity, equalized odds, predictive parity); Wilson CIs; small-group warnings. |
| `audit_report.py` | Slice + intersection reports, worst-slice flagging, configurable pass/fail gates (default set included). |
| `explainability.py` | Deterministic permutation importance (seeded RNG) and partial-dependence summaries over a caller-supplied predict function. |
| `model_card.py` | `ModelCard` with required-field enforcement; Markdown + JSON renderers; dict round-trip. |
| `audit_log.py` | Append-only hash-chained log (SHA-256, genesis hash, JSONL persistence) with verification returning the first bad index. |
| `drift.py` | Representation drift (per-group share change, total variation distance), label-rate shifts, per-group data-quality screens. |
| `mitigation.py` | Review-only aids: per-group threshold adjustment within bounds, capped reweighting suggestions. Neither mutates a model. |
| `generator.py` | Deterministic synthetic evaluation data with a deliberate group skew, for demos and tests only. |
| `config.py` | Environment-driven settings with validated defaults. |
| `observability.py` | Structured JSON-line logging (counts and labels only, never raw personal data). |
| `service.py` | Optional FastAPI app (import-guarded; `None` without FastAPI): `/fairness`, `/report`, `/model-card`, `/drift`, `/audit/verify`, `/health`. |
| `cli.py` | `demo` / `audit` / `card` / `serve` commands. |

## Data flow (demo path)

```
generate_synthetic ──▶ y_true / y_pred / scores / groups / age_bands
                            │
              ┌─────────────┼───────────────────────┐
              ▼             ▼                       ▼
     summarize_fairness  build_slice_report   adjust_thresholds
     (per-group + cross) (slices + intersect)  (review-only)
              │             │                       │
              └──────▶ evaluate_gates ◀─────────────┘
                            │
                            ▼
              ModelCard (fairness embedded verbatim)
                            │
                            ▼
              AuditLog.append + verify (hash chain)
```

## Design trade-offs

- **Undefined rates stay `None`, never 0.** Imputing 0 for a TPR with no
  positive labels would let a group "pass" a gate on a metric that does not
  exist for it. Gates return UNDEFINED in that case, which is a finding,
  not a pass.
- **Stdlib-only core** trades vectorised speed for auditability and zero
  install friction; evaluation sets of tens of thousands of rows are fast,
  million-row sets would want a vectorised backend behind the same APIs.
- **Small slices are flagged, never dropped.** Dropping small slices is how
  the worst harm gets hidden; a flagged small slice winning worst-slice
  ranking is itself a signal to collect more data, not to ignore it.
- **Four-fifths rule is a screen, not a legal finding.** A disparate-impact
  ratio below 0.8 triggers investigation; it does not establish
  discrimination, which depends on context, sample size, and justification.
- **Hash chain detects tampering, it does not prevent it.** Anyone who can
  rewrite the whole file can rewrite the chain. Production use anchors
  periodic hashes externally (e.g. a transparency log) or writes to an
  append-only store.
- **Mitigation is post-processing decision support, not a fix.** Threshold
  adjustment equalizes opportunity at the cost of overall accuracy and
  false positive rates; reweighting changes training emphasis. Both
  require human review and re-audit after application -- the module
  returns data structures, never a silently patched model.
- **Partial dependence describes the model, not the world.** Averaging
  predictions over counterfactual rows is misleading when features are
  strongly correlated; the helper documents this and stays numeric-only.
- **No auth on the demo service.** Production deployment adds
  authentication, rate limiting, and a real store in front of the API.
