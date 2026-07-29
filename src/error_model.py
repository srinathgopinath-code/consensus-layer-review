"""
Consensus-Layer Review: analytical + Monte-Carlo error model.
Beta-binomial (shared-latent) model of correlated panelist errors.
"""
import numpy as np
from scipy.stats import betabinom, binom
from scipy.special import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(20260728)

def beta_params(mean, rho):
    """Beta(a,b) with given mean and intra-class correlation rho (0..1)."""
    if rho <= 0:
        return None  # degenerate -> independent Bernoulli
    # rho = 1/(a+b+1) => a+b = (1-rho)/rho
    s = (1.0 - rho) / rho
    a = mean * s
    b = (1.0 - mean) * s
    return a, b

def residual_fp(N, k, q, rho):
    """P(>=k of N panelists independently 'report' a spurious/hallucinated finding).
       q = per-model hallucination (false-positive) rate for that item.
       rho = intra-class error correlation across panelists (shared training/data)."""
    if rho <= 0:
        return binom.sf(k-1, N, q)  # P(X>=k)
    a, b = beta_params(q, rho)
    return betabinom.sf(k-1, N, a, b)  # survival >= k

def residual_fn(N, k, m, rho):
    """P(a real bug is MISSED: fewer than k panelists catch it).
       m = per-model miss rate; catch rate = 1-m."""
    catch = 1.0 - m
    if rho <= 0:
        # X = number catching ~ Bin(N, catch); missed if X < k
        return binom.cdf(k-1, N, catch)
    a, b = beta_params(catch, rho)
    return betabinom.cdf(k-1, N, a, b)

# ---- Table 1: residual hallucination (FP) vs N and rho, majority threshold ----
q = 0.20          # per-model hallucination rate (~19.7% slopsquatting, USENIX'25)
print("=== Residual hallucination prob P(>=k) — majority k=ceil(0.6N) ===")
print(f"per-model q={q}")
Ns = [3,5,7,9,15,25,51]
rhos = [0.0, 0.05, 0.15, 0.30, 0.50]
header = "N\\rho | " + " | ".join(f"{r:>7.2f}" for r in rhos)
print(header)
table_fp = {}
for N in Ns:
    k = int(np.ceil(0.6*N))
    row=[]
    for r in rhos:
        v = residual_fp(N,k,q,r)
        row.append(v); table_fp[(N,r)]=v
    print(f"N={N:<3d}(k={k:<2d}) | " + " | ".join(f"{v:7.1e}" for v in row))

# ---- correlation floor: limit as N->inf with fractional threshold alpha ----
from scipy.stats import beta as betadist
alpha=0.6
print(f"\n=== Correlation FLOOR  P(theta>={alpha}) as N->inf (independent of tiers) ===")
for r in [0.05,0.15,0.30,0.50]:
    a,b=beta_params(q,r)
    floor=betadist.sf(alpha,a,b)
    print(f"rho={r:.2f}: floor={floor:.3e}")

# ---- Table 2: n-tier recursion ----
# Each monitoring tier is a consensus stage. If tiers were independent, residual multiplies.
# Realistically tiers reuse base models -> tier-to-tier correlation c limits gain.
print("\n=== n-tier stacking: independent-tiers vs correlated-tiers (per-tier residual e) ===")
def stack(e_single, n, tier_corr):
    """Independent tiers: e^n. Correlated tiers: floored by shared component."""
    indep = e_single**n
    # shared-failure floor: fraction tier_corr of failures are common-mode, uncatchable by more tiers
    floored = tier_corr*e_single + (1-tier_corr)*(e_single**n)
    return indep, floored
e_single = table_fp[(5,0.15)]  # N=5, rho=0.15 single-stage residual
for n in [1,2,3,4]:
    ind,flo = stack(e_single,n,tier_corr=0.10)
    print(f"n={n}: independent-tiers={ind:.2e}   correlated(c=0.10)={flo:.2e}")

# ---- Monte-Carlo cross-check of the analytical FP for one cell ----
def mc_fp(N,k,q,rho,trials=2_000_000):
    if rho<=0:
        X=rng.binomial(N,q,size=trials)
    else:
        a,b=beta_params(q,rho)
        theta=rng.beta(a,b,size=trials)
        X=rng.binomial(N,theta)
    return np.mean(X>=k)
N,k,r=7,int(np.ceil(0.6*7)),0.30
ana=residual_fp(N,k,q,r); mc=mc_fp(N,k,q,r)
print(f"\n=== Monte-Carlo cross-check  N={N},k={k},rho={r} ===")
print(f"analytical={ana:.4e}   monte-carlo={mc:.4e}   rel.diff={abs(ana-mc)/ana:.2%}")

# ---- Figure: residual FP vs N for several rho ----
plt.figure(figsize=(7,4.6))
Ncont=np.arange(3,52,2)
for r in [0.0,0.05,0.15,0.30,0.50]:
    ys=[residual_fp(N,int(np.ceil(0.6*N)),q,r) for N in Ncont]
    lab = f"ρ={r:.2f}" + (" (independent)" if r==0 else "")
    plt.semilogy(Ncont,ys,marker='o',ms=3,label=lab)
plt.xlabel("Panel size N  (corroboration threshold k = ⌈0.6N⌉)")
plt.ylabel("Residual hallucination prob.  P(X≥k)")
plt.title("Residual hallucinated-finding rate vs panel size and error correlation ρ\n(per-model q=0.20)")
plt.grid(True,which='both',alpha=0.3); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_residual.png",dpi=150)
print("\nsaved fig_residual.png")
