# Consensus-review measurement harness

Three scripts, all runnable with `python3` (needs `numpy`, `scipy`, `matplotlib`).

## 1. `measure_correlation.py` — ingest REAL panel outputs (the live front-end)

This is the script wired to real model outputs. Give it what an N-model review panel
actually emits plus your ground-truth labels, and it returns the quantities Section 9 of
the paper defines: per-model false-positive/false-negative rates, inter-model error
correlations, the population correlation and implied hallucination floor with bootstrap CIs,
and the consolidation sweep. No simulation is involved.

### Input format (JSONL: one JSON object per line)

```json
{
  "item_id": "repo@sha#hunk3",
  "stratum": "seeded",
  "ground_truth": [ {"location": "src/pay.py:42", "class": "security"} ],
  "reviews": {
    "gpt-x":    [ {"location": "src/pay.py:42", "class": "security"} ],
    "claude-y": [ {"location": "src/pay.py:42", "class": "security"},
                  {"location": "src/pay.py:900", "class": "logic"} ],
    "gemini-z": [ ]
  }
}
```

- `location`: any stable string (file:line, symbol, hunk id). `--match` sets strictness
  (`exact`, `file`, or `line3` which tolerates off-by-a-few line numbers).
- `class`: a category; equivalence groups live in `CLASS_EQUIV` at the top of the script.
- `stratum`: one of `seeded | historical | expert | halluc_dep` (used for the P2 test).
- A model listed with `[]` = "reviewed, found nothing". A model **absent** from an item
  = "did not review it"; that cell is dropped.

Scoring per item/model: a reported finding that matches a ground-truth finding is a true
positive; one that matches none is a false positive (hallucination); a ground-truth finding
matched by no report is a false negative (miss).

### Run

```bash
python3 measure_correlation.py --make-example example_reviews.jsonl   # writes a sample
python3 measure_correlation.py --reviews example_reviews.jsonl --bootstrap 2000
python3 measure_correlation.py --reviews YOUR_DATA.jsonl --alpha 0.6 --tiers 2
```

### Two estimators, and why it matters

- **False-negative correlation** is measured over an unbiased universe (every ground-truth
  finding is enumerated).
- **False-positive correlation** must be measured at the **item level** (every reviewed item
  is a unit, event = "did this model hallucinate at all on this item"). Measuring it only
  over *reported* spurious findings conditions on "someone fired" and biases the estimate
  toward zero. The harness reports both and treats the item-level figure as primary. On the
  synthetic example (known injected correlation) the item-level estimator recovers it; the
  reported-findings estimator collapses to near zero. This is the measurement pitfall
  discussed in Section 9.4 of the paper.

## 2. `simulate_consensus.py` — agent-level Monte-Carlo (no real data needed)

Simulates a correlated panel + consolidator + monitoring tiers to explore residuals across
N, ρ, k, and tier count. Validates the closed form to <1%. Use it to plan a study (e.g.,
what N and ρ you would need) before collecting real outputs.

## 3. `error_model.py` — the closed-form analytical model

Produces the paper's Tables 1–3 and Figure 1 (residual vs N, ρ, and tiers) plus the
Monte-Carlo cross-check.

## Suggested pilot workflow

1. Pick a small labeled corpus (seeded mutants + a few history-localized real bugs + a
   hallucinated-dependency stratum).
2. Run your N diverse models over each diff; capture each model's findings into the JSONL
   schema above.
3. `python3 measure_correlation.py --reviews pilot.jsonl --bootstrap 2000`.
4. Read off ρ_FP (item-level), ρ_FN, and the implied floor CI. If the floor sits above 1e-2
   for your reachable panel, that is the empirical answer to whether autonomous review can
   meet a regulated-change bar (prediction P4).
