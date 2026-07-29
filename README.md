# Consensus-Layer Autonomous Code Review

Reproducibility repository for the paper *"Consensus-Layer Autonomous Code Review: A Layered
Multi-Model Monitoring Architecture and a Correlation-Bounded Error Model for
Hallucination-Resistant, Human-On-the-Loop Review"* (Srinath Gopinath).

The paper studies whether a panel of independent review models on diverse base models, plus a
consolidating feedback-reviewer and recursive monitoring tiers, can reduce hallucinated review
findings enough to move the human from *in-the-loop* to *on-the-loop*. The central result is a
**correlation floor**: residual hallucination is bounded below by a quantity set by inter-model
error correlation, invariant to both panel size and the number of monitoring tiers. The paper's
centerpiece is a live protocol for measuring that correlation, and this repo contains the code
that implements it end to end.

## Layout

```
paper/     Consensus_Review_Paper.docx / .pdf   the paper
src/       error_model.py                        closed-form analytical model (Tables 1-3, Fig 1)
           simulate_consensus.py                 agent-level Monte-Carlo of the full pipeline
           measure_correlation.py                ingest REAL panel outputs -> rho + floor
data/      example_reviews.jsonl                 runnable sample in the input schema
figures/   fig_residual.png, sim_*.png          generated evidence figures
docs/      HARNESS_README.md                     schema + CLI + pilot workflow
           RESULTS_*.txt                         captured runs (evidence)
           MASTER_PROMPT_*.md                    the prompt used to draft the paper (provenance)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. analytical model: residual hallucination vs N, rho, tiers (+ Monte-Carlo check)
python3 src/error_model.py

# 2. agent-level simulation of panel + consolidator + monitoring tiers
python3 src/simulate_consensus.py --N 7 --rho 0.15 --tiers 2

# 3. measure correlation from REAL model outputs (or the bundled example)
python3 src/measure_correlation.py --make-example data/example_reviews.jsonl
python3 src/measure_correlation.py --reviews data/example_reviews.jsonl --bootstrap 2000
```

## The measurement front-end (`measure_correlation.py`)

This is the script wired to real panel output. Feed it one JSON record per reviewed diff, each
carrying the ground-truth findings and every model's reported findings (schema in
`docs/HARNESS_README.md`), and it returns the per-model false-positive/false-negative rates,
the inter-model error correlations, the population fit and implied hallucination floor with
item-clustered bootstrap CIs, and the consolidation sweep. No simulation in the path, so a pilot
on a few hundred labeled diffs plugs straight in.

Note on estimation: false-negative correlation is measured over an unbiased universe (every
ground-truth finding is enumerated). False-positive correlation must be measured at the item
level, because measuring it only over *reported* spurious findings conditions on "someone fired"
and biases the estimate toward zero. Both are reported; the item-level figure is primary. On the
bundled example (known injected correlation) the item-level estimator recovers it while the
reported-findings estimator collapses to near zero.

## Reproducing the figures

`error_model.py` writes `fig_residual.png`; `simulate_consensus.py` writes
`sim_residual_vs_N.png` and `sim_residual_vs_tiers.png`.

## Status and honesty note

All quantitative results in the paper are analytical or simulated. The live correlation study
(Section 9) is specified but not executed, on the principle that a simulator which assumes a
correlation can only return it. `measure_correlation.py` is the tool that would run that study on
real model outputs.

## License

MIT. See `LICENSE`.
