#!/usr/bin/env python
# coding: utf-8
"""
=============================================================================
 cps_realdata_whiten.py  --  Whiten the CSU innovations for the F6 test

 WHY THIS EXISTS
 ---------------
 The first F6 pass loaded R2_innovations.npz and treated EVERY array in it as
 a segment.  But that file stores FOUR predictor tiers per segment:
     T0_diff  (pure differencing -> lag-1 autocorr ~ -0.5 BY CONSTRUCTION)
     T1_ar    (single-channel AR, under-fit)
     T2_varx  (full MIMO VARX least squares -- THE innovation, R4 uses this)
     T3_gru   (nonlinear predictor, if torch available)
 Pooling all four inflated the NOATTACK baseline exposure to ~0.39 and hid the
 attack-induced tail effect the companion is about.

 This script produces a CLEAN, single-tier innovation file whose nominal
 baseline is as white as the estimator can make it, plus a per-segment
 whiteness tag so F6 can (a) measure attack-induced exposure against a white
 baseline and (b) isolate the well-whitened segments.

 TWO MODES
 ---------
   --from_existing   (fast, no CAN log needed)
       Read results/R2_innovations.npz, keep ONE tier (default T2_varx,
       channel 0), tag each segment's baseline whiteness, export.

   --refit           (default; needs the R0 cache)
       Re-fit the MIMO VARX (T2) over a sweep of orders p, pick the order that
       minimises the pooled baseline exposure functional (whiter = lower),
       export that order's innovations.  This is the knob that "raises the
       VARX order until the NOATTACK baseline -> 0".

 OUTPUT (results/)
   R2_innovations_white.npz
       segments      : object array of 1-D innovations (chosen tier/order)
       baseline_maxr : per-segment  max_{l<=L} |rhat_l|/S   (NOATTACK exposure)
       order, tier, expose_lags : scalars (provenance)
   Rw_whitening_diag.csv
       per (order-or-tier, segment): baseline_maxr, ljung_p, kurtosis, n

 Then run F6 on it:
   python cps_companion_F6.py --innov_file results/R2_innovations_white.npz ...
=============================================================================
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np

import cps_realdata as RD


# ----- the exposure functional, IDENTICAL to F6 -----------------------------
def rhat_vec(e, lags):
    """per-lag |r_l|/S profile of a 1-D series (mean removed, S = variance)."""
    e = np.asarray(e, float)
    e = e - e.mean()
    S = float(np.mean(e * e)) or 1.0
    return np.array([float(np.mean(e[:-l] * e[l:]) / S) for l in range(1, lags + 1)])

def baseline_maxr(e, lags):
    return float(np.max(np.abs(rhat_vec(e, lags))))


# ----- mode 1: re-tier an existing R2_innovations.npz -----------------------
def from_existing(cfg, tier, lags):
    inn, p_star = RD.load_innovations(cfg)
    if tier not in inn:
        avail = ", ".join(sorted(inn))
        raise SystemExit(f"tier '{tier}' not in file; available: {avail}")
    segs = inn[tier]
    out, diag = [], []
    for si, E in sorted(segs.items()):
        e0 = np.atleast_2d(np.asarray(E))
        e0 = (e0 if e0.shape[0] >= e0.shape[1] else e0.T)[:, 0]
        mr = baseline_maxr(e0, lags)
        out.append(e0.astype(float))
        diag.append(dict(order=int(p_star), tier=tier, segment=si, n=len(e0),
                         baseline_maxr=mr,
                         ljung_p=RD.ljung_box(e0, cfg.lb_lags)["pvalue"],
                         kurtosis=RD.excess_kurtosis(e0)))
    return out, [d["baseline_maxr"] for d in diag], int(p_star), diag


# ----- mode 2: refit VARX over an order sweep, pick the whitest --------------
def refit(cfg, orders, lags, min_len_mult=3):
    D = RD.load_cache(cfg)
    segs = D["segs"]
    if not segs:
        raise SystemExit("R0 cache has 0 stationary segments; run R0 first.")
    y_use = RD.usable_outputs(cfg)

    # per order: fit every segment (channel-0 innovation), record baseline_maxr
    per_order = {}   # p -> list of (si, e0, maxr, ljung, kurt, n)
    for p in orders:
        recs = []
        for si, (s, e) in enumerate(segs):
            Y, U, yk, uk = RD._segment_matrices(D, s, e, y_cols=y_use)
            N = Y.shape[0]
            need = p * (Y.shape[1] + (0 if U is None else U.shape[1])) + 2
            if N <= max(need, min_len_mult * p):
                continue
            try:
                E, _, _ = RD.varx_innovations(Y, U, p=p, ridge=cfg.ridge)
            except ValueError:
                continue
            e0 = E[:, 0]
            recs.append((si, e0.astype(float), baseline_maxr(e0, lags),
                         RD.ljung_box(e0, cfg.lb_lags)["pvalue"],
                         RD.excess_kurtosis(e0), len(e0)))
        if recs:
            per_order[p] = recs
            med = float(np.median([r[2] for r in recs]))
            frac_white = float(np.mean([r[2] < 0.05 for r in recs]))
            print(f"  [whiten] p={p:3d}  segs={len(recs):3d}  "
                  f"median baseline_maxr={med:.3f}  "
                  f"frac(<0.05)={frac_white:.2f}  "
                  f"median LB-p={np.median([r[3] for r in recs]):.3f}")

    if not per_order:
        raise SystemExit("no order produced usable innovations; segments too short.")

    # choose the order with the lowest MEDIAN baseline_maxr (whitest);
    # tie-break toward the SMALLER order (fewer parameters).
    best_p = min(per_order, key=lambda p: (round(np.median(
        [r[2] for r in per_order[p]]), 4), p))
    recs = sorted(per_order[best_p], key=lambda r: r[0])
    print(f"  [whiten] selected order p*={best_p} "
          f"(lowest median baseline_maxr)")

    out = [r[1] for r in recs]
    base = [r[2] for r in recs]
    diag = []
    for p, rlist in per_order.items():
        for (si, e0, mr, lb, ku, n) in rlist:
            diag.append(dict(order=p, tier="T2_varx", segment=si, n=n,
                             baseline_maxr=mr, ljung_p=lb, kurtosis=ku))
    return out, base, best_p, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["refit", "from_existing"], default="refit")
    ap.add_argument("--tier", default="T2_varx",
                    help="from_existing: which tier to keep (T2_varx recommended)")
    ap.add_argument("--orders", default="40,60,80,100,120",
                    help="refit: VARX orders to sweep")
    ap.add_argument("--expose_lags", type=int, default=20)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out_file", default="R2_innovations_white.npz")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    cfg = RD.Config()
    if a.out_dir: cfg.out_dir = a.out_dir
    if a.cache:   cfg.cache = a.cache
    cfg.dirs()
    if a.quick:
        a.orders = "40,60"

    t0 = time.time()
    if a.mode == "from_existing":
        segs, base, order, diag = from_existing(cfg, a.tier, a.expose_lags)
        tier = a.tier
    else:
        orders = [int(x) for x in a.orders.split(",")]
        segs, base, order, diag = refit(cfg, orders, a.expose_lags)
        tier = "T2_varx"

    base = np.asarray(base, float)
    outp = Path(cfg.out_dir) / a.out_file
    np.savez_compressed(
        outp,
        segments=np.array(segs, dtype=object),
        baseline_maxr=base,
        order=np.array([order]),
        tier=np.array([tier]),
        expose_lags=np.array([a.expose_lags]),
    )
    # diagnostic CSV
    if diag:
        keys = list(diag[0].keys())
        with open(Path(cfg.out_dir) / "Rw_whitening_diag.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(diag)

    n_white = int(np.sum(base < 0.05))
    print("\n" + "=" * 68)
    print(f"  wrote {outp}")
    print(f"  tier={tier}  order={order}  segments={len(segs)}  "
          f"expose_lags={a.expose_lags}")
    print(f"  baseline_maxr: median={np.median(base):.3f}  "
          f"mean={base.mean():.3f}  min={base.min():.3f}  max={base.max():.3f}")
    print(f"  well-whitened segments (baseline_maxr<0.05): "
          f"{n_white}/{len(base)} ({100*n_white/len(base):.0f}%)")
    print(f"  -> {cfg.out_dir}/Rw_whitening_diag.csv")
    print(f"  done in {time.time()-t0:.0f}s")
    print("=" * 68)
    print(f"\nNext:\n  python cps_companion_F6.py --innov_file {outp} "
          f"--out_dir {cfg.out_dir}")


if __name__ == "__main__":
    main()
