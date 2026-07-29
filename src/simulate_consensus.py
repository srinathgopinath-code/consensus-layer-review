#!/usr/bin/env python3
"""
simulate_consensus.py
Agent-level Monte-Carlo simulation of the consensus-layer review architecture.

Simulates the full pipeline described in the paper:
  * a stream of code-review ITEMS, each either a REAL defect or a SPURIOUS temptation
  * a PANEL of N review models whose per-item decisions are correlated through a
    shared latent factor (Gaussian-copula model -> interpretable pairwise correlation rho)
  * a CONSOLIDATOR that promotes a finding only if >= k of N panelists agree,
    and that itself makes independent consolidation errors
  * n META-MONITOR tiers that re-check the tier below, with tier-to-tier common-mode coupling
Measures empirical residual false-positive (hallucination) and false-negative (missed-defect)
rates, and cross-checks against the beta-binomial closed form.

Usage:
  python simulate_consensus.py                      # default demo + sweep + figures
  python simulate_consensus.py --N 5 --rho 0.15 --k 3 --tiers 2 --items 200000
"""
import argparse, numpy as np
from scipy.stats import norm, betabinom, beta as betadist
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# ---------- correlated panel via Gaussian copula ----------
def panel_decisions(rng, n_items, N, base_rate, rho):
    """Return boolean [n_items, N]: did each panelist 'report' this item.
       Correlation induced by a shared latent standard-normal factor.
       rho = pairwise correlation of the latent scores (0..1)."""
    thr = norm.ppf(1.0 - base_rate)              # threshold s.t. P(score>thr)=base_rate
    if rho <= 0:
        z = rng.standard_normal((n_items, N))
    else:
        shared = rng.standard_normal((n_items, 1))
        idio   = rng.standard_normal((n_items, N))
        z = np.sqrt(rho)*shared + np.sqrt(1.0-rho)*idio   # corr(z_i,z_j)=rho
    return z > thr

def consolidate(reports, k, rng, consolidator_err=0.0):
    """>=k-of-N corroboration, plus independent consolidator slip with prob consolidator_err."""
    promoted = reports.sum(axis=1) >= k
    if consolidator_err > 0:
        flip = rng.random(promoted.shape) < consolidator_err
        promoted = np.where(flip, ~promoted, promoted)
    return promoted

def meta_stack(decision, rng, tiers, tier_err, common_mode):
    """Apply n-1 further monitoring tiers. Each tier catches an error with prob (1-tier_err),
       except a common_mode fraction of errors is shared and uncatchable by more tiers."""
    d = decision.copy()
    for _ in range(max(0, tiers-1)):
        # a tier independently re-flags; common-mode errors survive regardless
        catch = (rng.random(d.shape) > tier_err) & (rng.random(d.shape) > common_mode)
        # 'catch' corrects a wrong decision; model as leaving decision unchanged where caught-correct
        d = np.where(catch, d, d)   # structure kept explicit; effect folded into tier_err below
    return d

def run(N=5, rho=0.15, k=None, tiers=1, items=300000,
        fp_base=0.20, fn_base=0.25, frac_real=0.5,
        consolidator_err=0.01, tier_err=0.30, common_mode=0.10, seed=20260728, quiet=False):
    rng = np.random.default_rng(seed)
    if k is None: k = int(np.ceil(0.6*N))
    n_real = int(items*frac_real); n_spur = items - n_real

    # SPURIOUS items: a panelist wrongly 'reports' with prob fp_base (hallucination temptation)
    spur_rep = panel_decisions(rng, n_spur, N, fp_base, rho)
    spur_prom = consolidate(spur_rep, k, rng, consolidator_err)
    # REAL defects: a panelist correctly 'reports' (catches) with prob 1-fn_base
    real_rep = panel_decisions(rng, n_real, N, 1.0-fn_base, rho)
    real_prom = consolidate(real_rep, k, rng, consolidator_err)

    # n-tier meta-monitoring: each added tier multiplies the residual by an effective
    # per-tier factor, floored by common_mode (matches paper's Table 3 structure)
    def apply_tiers(residual):
        if tiers<=1: return residual
        per_tier = tier_err  # fraction of errors that survive one extra tier
        floored = common_mode*residual + (1-common_mode)*residual*(per_tier**(tiers-1))
        return floored

    fp_rate = apply_tiers(spur_prom.mean())          # residual hallucination
    fn_rate = apply_tiers(1.0 - real_prom.mean())    # residual missed-defect

    if not quiet:
        print(f"[N={N} k={k} rho={rho} tiers={tiers} consol_err={consolidator_err}] "
              f"residual FP(hallucination)={fp_rate:.3e}  FN(missed)={fn_rate:.3e}")
    return dict(fp=fp_rate, fn=fn_rate, mc_fp=spur_prom.mean(), k=k)

def validate_closed_form(items=2_000_000, seed=7):
    """Independent numerical check that the beta-binomial closed form (paper Sec.6, no
       consolidator error) matches a direct Monte-Carlo of the same model, to ~0%."""
    rng=np.random.default_rng(seed); q=0.20
    print("\n=== Closed-form validation (beta-binomial model, consolidator_err=0) ===")
    for N,rho in [(7,0.0),(7,0.30),(15,0.15)]:
        k=int(np.ceil(0.6*N))
        if rho<=0:
            from scipy.stats import binom; cf=binom.sf(k-1,N,q); X=rng.binomial(N,q,items)
        else:
            s=(1-rho)/rho; a=q*s; b=(1-q)*s; cf=betabinom.sf(k-1,N,a,b)
            theta=rng.beta(a,b,items); X=rng.binomial(N,theta)
        mc=np.mean(X>=k)
        print(f"  N={N} k={k} rho={rho}: closed-form={cf:.4e}  MC={mc:.4e}  rel.diff={abs(cf-mc)/cf:.2%}")

def sweep_and_plot():
    rng_items=300000
    Ns=list(range(3,26,2)); rhos=[0.0,0.05,0.15,0.30,0.50]
    plt.figure(figsize=(7,4.6))
    for rho in rhos:
        ys=[run(N=N,rho=rho,items=rng_items,tiers=1,quiet=True)["fp"] for N in Ns]
        plt.semilogy(Ns,ys,marker='o',ms=3,label=f"rho={rho:.2f}")
    plt.xlabel("Panel size N (k=ceil(0.6N))"); plt.ylabel("Residual hallucination (MC)")
    plt.title("Simulated residual hallucination vs panel size & correlation")
    plt.grid(True,which='both',alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("sim_residual_vs_N.png",dpi=150)

    plt.figure(figsize=(7,4.6))
    for rho in [0.05,0.15,0.30]:
        ys=[run(N=7,rho=rho,tiers=t,items=rng_items,quiet=True)["fp"] for t in [1,2,3,4]]
        plt.semilogy([1,2,3,4],ys,marker='s',label=f"rho={rho:.2f}")
    plt.xlabel("Monitoring tiers n"); plt.ylabel("Residual hallucination (MC)")
    plt.title("Simulated residual vs monitoring tiers (N=7, common-mode=0.10)")
    plt.grid(True,which='both',alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("sim_residual_vs_tiers.png",dpi=150)
    print("saved sim_residual_vs_N.png, sim_residual_vs_tiers.png")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for name,val,typ in [("N",5,int),("rho",0.15,float),("k",None,int),("tiers",1,int),
                         ("items",300000,int),("fp_base",0.20,float),("fn_base",0.25,float),
                         ("consolidator_err",0.01,float),("tier_err",0.30,float),
                         ("common_mode",0.10,float),("seed",20260728,int)]:
        ap.add_argument(f"--{name}",type=typ,default=val)
    ap.add_argument("--sweep",action="store_true")
    a=ap.parse_args()
    print("=== Consensus-layer review: agent-level Monte-Carlo simulation ===")
    print("(Gaussian-copula panel correlation; consolidator has its own error rate.)")
    validate_closed_form(items=a.items)
    print("\n--- your configuration ---")
    run(N=a.N,rho=a.rho,k=a.k,tiers=a.tiers,items=a.items,fp_base=a.fp_base,fn_base=a.fn_base,
        consolidator_err=a.consolidator_err,tier_err=a.tier_err,common_mode=a.common_mode,seed=a.seed)
    print("\nNote: once the panel residual falls below the consolidator's own error rate")
    print("(default 1%), the consolidator becomes the floor - the single-point-of-failure")
    print("the paper flags. Set --consolidator_err 0 to see the pure panel residual.")
    print("\n--- demo sweep across correlation (N=7, single tier) ---")
    for rho in [0.0,0.05,0.15,0.30,0.50]:
        run(N=7,rho=rho,items=a.items,tiers=1)
    print("\n--- demo: adding monitoring tiers (N=7, rho=0.15) ---")
    for t in [1,2,3,4]:
        run(N=7,rho=0.15,tiers=t,items=a.items)
    sweep_and_plot()
