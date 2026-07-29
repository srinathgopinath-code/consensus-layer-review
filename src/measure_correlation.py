#!/usr/bin/env python3
"""
measure_correlation.py
Ingest REAL per-model code-review outputs + ground-truth labels and estimate the
quantities the paper's Section 9 protocol calls for:
  * per-model marginal false-positive (hallucination) and false-negative (miss) rates
  * pairwise inter-model error correlation matrices rho_FP and rho_FN (phi + tetrachoric)
  * population correlation rho via a beta-binomial fit, and the implied floor P(theta>=alpha)
  * consolidated FP/FN under >=k-of-N corroboration, swept over k, with n-tier extension
  * bootstrap confidence intervals (clustered by item)

This is the live-execution front-end for the architecture: it accepts the same JSONL that a
real N-model panel would emit, so a pilot on a few hundred labeled diffs plugs straight in.

------------------------------------------------------------------------------------------
INPUT SCHEMA  (one JSON object per line; JSONL)
{
  "item_id": "repo@sha#hunk3",
  "stratum": "seeded" | "historical" | "expert" | "halluc_dep",
  "ground_truth": [ {"location": "path/file.py:42", "class": "security"}, ... ],
  "reviews": {
     "gpt-x":      [ {"location": "path/file.py:42", "class": "security"}, ... ],
     "claude-y":   [ ... ],
     "gemini-z":   [ ... ]
  }
}
Notes:
 * "location" is any stable string (file:line, symbol, or hunk id). Matching mode controls
   how strictly locations must agree (see --match).
 * "class" is a finding category; equivalence groups are configurable in CLASS_EQUIV.
 * A model key present with [] means "reviewed, found nothing". A model key ABSENT for an
   item means "did not review this item" and that (item, model) cell is dropped.
------------------------------------------------------------------------------------------

USAGE
  python measure_correlation.py --reviews data.jsonl                 # full report
  python measure_correlation.py --reviews data.jsonl --alpha 0.6 --tiers 2 --bootstrap 2000
  python measure_correlation.py --make-example example.jsonl         # write a runnable sample
"""
import argparse, json, itertools, sys, math
import numpy as np
from scipy.stats import betabinom, norm
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import multivariate_normal

# ---- configurable equivalence of finding classes (findings match if classes share a group) ----
CLASS_EQUIV = [
    {"security","vuln","injection","authz","secret"},
    {"logic","correctness","bug"},
    {"dependency","import","package","supply-chain"},
    {"perf","performance"},
    {"style","naming","nit"},
]
def class_key(c):
    c=(c or "").strip().lower()
    for g in CLASS_EQUIV:
        if c in g: return frozenset(g)
    return c  # its own singleton group

def loc_key(loc, mode):
    s=(loc or "").strip()
    if mode=="exact": return s
    if mode=="file":  return s.split(":")[0]
    if mode=="line3":  # collapse line numbers to +-3 buckets to tolerate off-by-a-few
        if ":" in s:
            f,_,ln=s.rpartition(":")
            try: return f+":"+str(int(ln)//3)
            except ValueError: return s
        return s
    return s

def finding_id(f, mode):
    return (loc_key(f.get("location"),mode), class_key(f.get("class")))

# ---------------- ingestion & scoring ----------------
def load(path):
    items=[]
    with open(path) as fh:
        for ln in fh:
            ln=ln.strip()
            if ln: items.append(json.loads(ln))
    return items

def build_event_matrices(items, match_mode):
    """Return, across all items:
       FP_events: list of arrays [n_models] over spurious 'slots' (candidate hallucinations)
       FN_events: list of arrays [n_models] over ground-truth findings (1 = MISSED)
       plus the ordered model list and per-item cluster ids for bootstrap.
    """
    models=sorted({m for it in items for m in it.get("reviews",{})})
    midx={m:i for i,m in enumerate(models)}
    FP_rows=[]; FP_item=[]; FN_rows=[]; FN_item=[]
    for qi,it in enumerate(items):
        reviews=it.get("reviews",{})
        present=[m for m in models if m in reviews]  # models that reviewed this item
        if not present: continue
        # sets of reported finding-ids per model
        rep={m:set(finding_id(f,match_mode) for f in reviews.get(m,[])) for m in present}
        gt=set(finding_id(f,match_mode) for f in it.get("ground_truth",[]))
        # ---- FN slots: one per GT finding; event = model MISSED it ----
        for g in gt:
            row=np.full(len(models), np.nan)
            for m in present: row[midx[m]] = 0.0 if g in rep[m] else 1.0
            FN_rows.append(row); FN_item.append(qi)
        # ---- FP slots: union of reported findings NOT in GT; event = model reported it ----
        spurious=set().union(*rep.values()) - gt if rep else set()
        for s in spurious:
            row=np.full(len(models), np.nan)
            for m in present: row[midx[m]] = 1.0 if s in rep[m] else 0.0
            FP_rows.append(row); FP_item.append(qi)
    return (models,
            np.array(FP_rows) if FP_rows else np.empty((0,len(models))), np.array(FP_item),
            np.array(FN_rows) if FN_rows else np.empty((0,len(models))), np.array(FN_item))

def build_item_matrices(items, match_mode):
    """UNBIASED item-level events (every reviewed item is in the universe, including items
       where no model erred). Entry = 1 if model committed >=1 error of that type on the item,
       NaN if the model did not review the item. Avoids the 'conditioned on >=1 report'
       selection bias that deflates slot-level FP correlation."""
    models=sorted({m for it in items for m in it.get("reviews",{})})
    midx={m:i for i,m in enumerate(models)}
    FP=[]; FN=[]
    for it in items:
        reviews=it.get("reviews",{}); present=[m for m in models if m in reviews]
        if not present: continue
        gt=set(finding_id(f,match_mode) for f in it.get("ground_truth",[]))
        rowfp=np.full(len(models),np.nan); rowfn=np.full(len(models),np.nan)
        for m in present:
            rep=set(finding_id(f,match_mode) for f in reviews.get(m,[]))
            rowfp[midx[m]] = 1.0 if (rep-gt) else 0.0          # hallucinated at least once
            rowfn[midx[m]] = 1.0 if (gt-rep) else 0.0          # missed at least one real finding
        FP.append(rowfp); FN.append(rowfn)
    return (models,
            np.array(FP) if FP else np.empty((0,len(models))),
            np.array(FN) if FN else np.empty((0,len(models))))

# ---------------- correlation estimators ----------------
def phi_pair(x,y):
    m=~np.isnan(x)&~np.isnan(y); x,y=x[m],y[m]
    if len(x)<3 or x.std()==0 or y.std()==0: return np.nan
    return float(np.corrcoef(x,y)[0,1])

def tetrachoric_pair(x,y):
    """MLE tetrachoric correlation from the 2x2 table via bivariate-normal thresholds."""
    m=~np.isnan(x)&~np.isnan(y); x,y=x[m].astype(int),y[m].astype(int)
    n=len(x)
    if n<8: return np.nan
    n11=int(np.sum((x==1)&(y==1))); n10=int(np.sum((x==1)&(y==0)))
    n01=int(np.sum((x==0)&(y==1))); n00=int(np.sum((x==0)&(y==0)))
    if min(n11+n10,n01+n00,n11+n01,n10+n00)==0: return np.nan
    hx=norm.ppf((n11+n10)/n); hy=norm.ppf((n11+n01)/n)  # thresholds
    def negll(r):
        r=max(min(r,0.999),-0.999)
        cov=[[1,r],[r,1]]
        p11=multivariate_normal(mean=[0,0],cov=cov).cdf([hx,hy])
        p11=min(max(p11,1e-9),1-1e-9)
        p10=norm.cdf(hx)-p11; p01=norm.cdf(hy)-p11; p00=1-p11-p10-p01
        eps=1e-9
        return -(n11*math.log(p11+eps)+n10*math.log(max(p10,eps))+
                 n01*math.log(max(p01,eps))+n00*math.log(max(p00,eps)))
    res=minimize_scalar(negll,bounds=(-0.95,0.95),method="bounded")
    return float(res.x)

def mean_pairwise(mat, fn):
    M=mat.shape[1]; vals=[]
    for i,j in itertools.combinations(range(M),2):
        v=fn(mat[:,i],mat[:,j])
        if not np.isnan(v): vals.append(v)
    return (float(np.mean(vals)) if vals else np.nan, vals)

def marginal_rates(mat):
    return np.nanmean(mat,axis=0)

# ---------------- beta-binomial population fit + floor ----------------
def counts_per_slot(mat):
    """For each slot, k reporters out of n present (drop NaN)."""
    ks=[]; ns=[]
    for row in mat:
        present=row[~np.isnan(row)]
        ns.append(len(present)); ks.append(int(np.nansum(row)))
    return np.array(ks),np.array(ns)

def fit_betabinom(ks,ns):
    """MLE of Beta(a,b) mean q and correlation rho from (k of n) slot counts."""
    if len(ks)==0: return np.nan,np.nan
    def negll(params):
        lq,ls=params; q=1/(1+math.exp(-lq)); s=math.exp(ls)  # s=a+b
        a=q*s; b=(1-q)*s; 
        return -np.sum(betabinom.logpmf(ks,ns,a,b))
    from scipy.optimize import minimize
    best=None
    for q0 in (0.1,0.3): 
        for s0 in (2.0,10.0):
            r=minimize(negll,[math.log(q0/(1-q0)),math.log(s0)],method="Nelder-Mead")
            if best is None or r.fun<best.fun: best=r
    lq,ls=best.x; q=1/(1+math.exp(-lq)); s=math.exp(ls); rho=1/(s+1)
    return q,rho

def implied_floor(q,rho,alpha):
    if any(np.isnan([q,rho])) or rho<=0: return np.nan
    from scipy.stats import beta as B
    s=(1-rho)/rho; a=q*s; b=(1-q)*s
    return float(B.sf(alpha,a,b))

# ---------------- consolidation ----------------
def consolidate_rates(FP,FN,k,tiers,common_mode,tier_err):
    def resid(mat,positive):  # positive=True: promoted spurious=hallucination; False: FN if <k catch
        ks,ns=counts_per_slot(mat)
        if len(ks)==0: return np.nan
        if positive: base=np.mean(ks>=k)              # hallucination promoted
        else:        base=np.mean(ks<max(1,k))        # real finding missed by consensus
        if tiers>1:
            base=common_mode*base+(1-common_mode)*base*(tier_err**(tiers-1))
        return float(base)
    return resid(FP,True), resid(FN,False)

# ---------------- report ----------------
def bootstrap_ci(items, match_mode, alpha, stat_fn, B, seed=1):
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(B):
        samp=[items[i] for i in rng.integers(0,len(items),len(items))]
        v=stat_fn(samp)
        if v is not None and not np.isnan(v): vals.append(v)
    if not vals: return (np.nan,np.nan)
    return (float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5)))

def report(items, match_mode="line3", alpha=0.6, tiers=1, common_mode=0.10,
           tier_err=0.30, k=None, bootstrap=0, tetra=True):
    models,FP,FPi,FN,FNi=build_event_matrices(items,match_mode)
    N=len(models); 
    if k is None: k=int(np.ceil(0.6*N))
    print(f"\n=== measure_correlation report ===")
    print(f"items={len(items)}  models(N)={N}  match={match_mode}  k={k}  alpha={alpha}")
    print(f"models: {', '.join(models)}")
    print(f"FP slots (candidate hallucinations)={FP.shape[0]}  FN slots (ground-truth findings)={FN.shape[0]}")

    if FP.shape[0]:
        fp_marg=marginal_rates(FP)
        print("\n-- marginal per-model hallucination rate (on spurious slots) --")
        for m,r in zip(models,fp_marg): print(f"   {m:20s} {r:.3f}")
    if FN.shape[0]:
        fn_marg=marginal_rates(FN)
        print("-- marginal per-model MISS rate (on ground-truth findings) --")
        for m,r in zip(models,fn_marg): print(f"   {m:20s} {r:.3f}")

    # unbiased item-level matrices (primary for FP correlation & floor)
    _,FPi_mat,FNi_mat=build_item_matrices(items,match_mode)
    rho_fp_item,_=mean_pairwise(FPi_mat,phi_pair) if FPi_mat.shape[0] else (np.nan,[])
    rho_fn_item,_=mean_pairwise(FNi_mat,phi_pair) if FNi_mat.shape[0] else (np.nan,[])
    rho_fn_slot,_=mean_pairwise(FN,phi_pair) if FN.shape[0] else (np.nan,[])
    rho_fp_slot,_=mean_pairwise(FP,phi_pair) if FP.shape[0] else (np.nan,[])
    print(f"\n-- inter-model error correlation (mean pairwise phi) --")
    print(f"   rho_FP (item-level, unbiased) ={rho_fp_item:.3f}")
    print(f"   rho_FN (slot-level, unbiased) ={rho_fn_slot:.3f}   rho_FN (item-level)={rho_fn_item:.3f}")
    print(f"   rho_FP (slot-level, conditioned on >=1 report; biased low)={rho_fp_slot:.3f}")
    if tetra:
        rho_fp_it,_=mean_pairwise(FPi_mat,tetrachoric_pair) if FPi_mat.shape[0] else (np.nan,[])
        rho_fn_st,_=mean_pairwise(FN,tetrachoric_pair) if FN.shape[0] else (np.nan,[])
        print(f"   rho_FP (item, tetrachoric)={rho_fp_it:.3f}   rho_FN (slot, tetrachoric)={rho_fn_st:.3f}")

    # population fit + floor from UNBIASED item-level hallucination counts (0..N models per item)
    ks=np.nansum(FPi_mat,axis=1).astype(int); ns=np.sum(~np.isnan(FPi_mat),axis=1)
    q,rho=fit_betabinom(ks,ns) if FPi_mat.shape[0] else (np.nan,np.nan)
    floor=implied_floor(q,rho,alpha)
    print(f"\n-- beta-binomial population fit (item-level hallucination side) --")
    print(f"   q(mean per-model)={q:.3f}  rho(pop)={rho:.3f}  implied floor P(theta>={alpha})={floor:.3e}")

    fp_res,fn_res=consolidate_rates(FP,FN,k,tiers,common_mode,tier_err)
    print(f"\n-- consolidated residuals at k={k}, tiers={tiers} --")
    print(f"   residual hallucination (FP)={fp_res:.3e}   residual miss (FN)={fn_res:.3e}")

    print(f"\n-- consolidation sweep over k (tiers=1) --")
    for kk in range(1,N+1):
        a,b=consolidate_rates(FP,FN,kk,1,common_mode,tier_err)
        print(f"   k={kk:2d}: residual FP={a:.3e}  residual FN={b:.3e}")

    if bootstrap:
        def stat_floor(samp):
            _,fpi,_=build_item_matrices(samp,match_mode)
            if fpi.shape[0]==0: return np.nan
            ks=np.nansum(fpi,axis=1).astype(int); ns=np.sum(~np.isnan(fpi),axis=1)
            q,rho=fit_betabinom(ks,ns)
            return implied_floor(q,rho,alpha)
        lo,hi=bootstrap_ci(items,match_mode,alpha,stat_floor,bootstrap)
        print(f"\n-- bootstrap 95% CI on implied floor ({bootstrap} resamples, item-clustered) --")
        print(f"   [{lo:.3e}, {hi:.3e}]")

    # per-stratum rho_FP (the P2 prediction: halluc_dep should be highest)
    strata=sorted({it.get('stratum','?') for it in items})
    if len(strata)>1:
        print("\n-- rho_FP by stratum, item-level unbiased (tests prediction P2) --")
        for st in strata:
            sub=[it for it in items if it.get('stratum')==st]
            _,fpi,_=build_item_matrices(sub,match_mode)
            r,_=mean_pairwise(fpi,phi_pair) if fpi.shape[0] else (np.nan,[])
            print(f"   {st:14s} rho_FP={r:.3f}  (items={fpi.shape[0]})")

# ---------------- example data generator ----------------
def make_example(path, n_items=240, seed=11):
    """Synthesize a labeled dataset in the schema with KNOWN injected correlation,
       so the harness can be exercised end-to-end and its estimates sanity-checked."""
    rng=np.random.default_rng(seed)
    models=["gpt-x","claude-y","gemini-z","llama-w","mistral-v"]
    strata=["seeded","historical","expert","halluc_dep"]
    # injected correlation per stratum (halluc_dep highest, per prediction P2)
    rho_by={"seeded":0.10,"historical":0.15,"expert":0.12,"halluc_dep":0.45}
    fp_base=0.20; fn_base=0.25
    with open(path,"w") as fh:
        for i in range(n_items):
            st=strata[i%len(strata)]; rho=rho_by[st]
            n_gt=rng.integers(1,4)
            gt=[{"location":f"src/f{i}.py:{10*(g+1)}","class":rng.choice(["security","logic","dependency"])} for g in range(n_gt)]
            # ---- per-item shared latents (drawn ONCE, outside the model loop) ----
            fn_shared={id(g):rng.standard_normal() for g in gt}          # one per real finding
            n_tempt=int(rng.integers(1,4))                              # fixed spurious slots per item
            tempt=[]
            for t in range(n_tempt):
                cls="dependency" if st=="halluc_dep" else str(rng.choice(["security","logic","perf"]))
                tempt.append({"location":f"src/f{i}.py:{500+t}","class":cls,
                              "latent":rng.standard_normal()})            # one per spurious slot
            reviews={}
            for m in models:
                found=[]
                # catches of real findings (miss with correlated prob)
                for g in gt:
                    z=math.sqrt(rho)*fn_shared[id(g)]+math.sqrt(1-rho)*rng.standard_normal()
                    miss = z > norm.ppf(1-fn_base)
                    if not miss: found.append({"location":g["location"],"class":g["class"]})
                # hallucinations: each model reports each spurious slot w/ correlated prob
                for tp in tempt:
                    z=math.sqrt(rho)*tp["latent"]+math.sqrt(1-rho)*rng.standard_normal()
                    if z>norm.ppf(1-fp_base):
                        found.append({"location":tp["location"],"class":tp["class"]})
                reviews[m]=found
            fh.write(json.dumps({"item_id":f"item{i}","stratum":st,"ground_truth":gt,"reviews":reviews})+"\n")
    print(f"wrote {path} ({n_items} items, {len(models)} models, injected rho by stratum={rho_by})")

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="Ingest real per-model review outputs; estimate rho and floor.")
    ap.add_argument("--reviews",help="JSONL of items with ground_truth + per-model reviews")
    ap.add_argument("--make-example",metavar="PATH",help="write a runnable synthetic example and exit")
    ap.add_argument("--match",default="line3",choices=["exact","file","line3"])
    ap.add_argument("--alpha",type=float,default=0.6)
    ap.add_argument("--k",type=int,default=None)
    ap.add_argument("--tiers",type=int,default=1)
    ap.add_argument("--common_mode",type=float,default=0.10)
    ap.add_argument("--tier_err",type=float,default=0.30)
    ap.add_argument("--bootstrap",type=int,default=0)
    ap.add_argument("--no-tetra",action="store_true")
    a=ap.parse_args()
    if a.make_example:
        make_example(a.make_example); sys.exit(0)
    if not a.reviews:
        ap.error("provide --reviews FILE.jsonl (or --make-example PATH to generate a sample)")
    items=load(a.reviews)
    report(items,match_mode=a.match,alpha=a.alpha,tiers=a.tiers,common_mode=a.common_mode,
           tier_err=a.tier_err,k=a.k,bootstrap=a.bootstrap,tetra=not a.no_tetra)
