#!/usr/bin/env python
# coding: utf-8
"""
=============================================================================
 cps_companion_F6.py  --  F6 v3: attack-induced exposure on real CSU CAN data

 Changes over F6 v2 (the three fixes requested after the NOATTACK finding):

  (1) TIER-AWARE, WHITENESS-FIRST loader.  R2_innovations.npz stores four
      predictor tiers per segment (T0_diff, T1_ar, T2_varx, T3_gru); the v2
      loader pooled ALL of them, and T0_diff (pure differencing, lag-1 ~ -0.5)
      inflated the baseline.  We now load ONE tier only:
        - a clean file with a 'segments' key (from cps_realdata_whiten.py) is
          used directly, together with its per-segment 'baseline_maxr';
        - a raw R2 file has ONLY its T2_varx tier selected (fallbacks T3_gru,
          then T1_ar), never T0_diff.

  (2) ATTACK-INDUCED delta.  Exposure is reported as
        Dr_l = rhat_l(attacked) - rhat_l(nominal),   max_drhat = max_l |Dr_l|,
      so the estimator's own residual correlation cancels and only what the
      ATTACK adds/removes is measured.  Absolute max_rhat is kept for reference.
      (Detector power already compares to a nominal null, so it is delta-like.)

  (3) WHITENESS GATING.  Segments are split by baseline whiteness
      (baseline_maxr < --white_thresh, default 0.05).  The SUMMARY and the
      pooled detector power are reported for BOTH the full set and the
      well-whitened subset -- the proof-of-principle cohort where Prop.2's
      precondition (a white nominal innovation) actually holds.

 Reuses the hysteresis scheduler / detectors from cps_companion.py.
 Outputs: F6b_functional.csv, F6b_power.csv (with a 'cohort' column),
          F6b_summary.csv (with a 'cohort' column).  No torch, no learning.

 Usage:
   python cps_companion_F6.py --innov_file results/R2_innovations_white.npz
   python cps_companion_F6.py --innov_file results/R2_innovations.npz   # raw ok
=============================================================================
"""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import numpy as np

import cps_experiments as E
import cps_companion as CC


# ---------------------------------------------------------------------------
def rhat_profile(e, lags):
    """per-lag rhat_l / S of a 1-D series (mean removed, S = variance)."""
    e = np.asarray(e, float); e = e - e.mean()
    S = float(np.mean(e * e)) or 1.0
    return np.array([float(np.mean(e[:-l] * e[l:]) / S) for l in range(1, lags + 1)])

def psi_of(z_dep, g, k0):
    z2 = z_dep[k0:] ** 2; gk = g[k0:]
    return float(z2[gk].sum() / z2.sum()) if gk.any() and z2.sum() > 0 else 0.0

def windows_stat(x, W, fn, two_sided):
    if len(x) < 2 * W:
        return np.empty(0)
    return E.block_stats(np.asarray(x, float), W, fn, two_sided)


# ----- tier-aware, whiteness-first loader ----------------------------------
def load_clean_segments(path, expose_lags, tier_pref=("T2_varx", "T3_gru", "T1_ar")):
    """Return (segments, baseline_maxr, provenance).  One tier only."""
    p = Path(path)
    d = np.load(p, allow_pickle=True)
    keys = list(d.files)
    # (a) clean file from cps_realdata_whiten.py
    if "segments" in keys:
        segs = [np.asarray(s, float).ravel() for s in d["segments"]]
        if "baseline_maxr" in keys:
            base = list(np.asarray(d["baseline_maxr"], float))
        else:
            base = [float(np.max(np.abs(rhat_profile(s, expose_lags)))) for s in segs]
        tier = str(d["tier"][0]) if "tier" in keys else "clean"
        order = int(d["order"][0]) if "order" in keys else -1
        print(f"  [F6] clean file: tier={tier} order={order} "
              f"{len(segs)} segment(s)")
        return segs, base, dict(tier=tier, order=order, source="clean")
    # (b) raw R2 file: pick ONE tier
    tiers = {}
    for k in keys:
        if k == "p_star" or "_seg" not in k:
            continue
        t, si = k.rsplit("_seg", 1)
        tiers.setdefault(t, {})[int(si)] = d[k]
    if tiers:
        chosen = next((t for t in tier_pref if t in tiers), sorted(tiers)[0])
        print(f"  [F6] raw R2 file: tiers found {sorted(tiers)} -> using '{chosen}' "
              f"(T0_diff/T1_ar are NOT innovations and are skipped)")
        segs, base = [], []
        for si, E_ in sorted(tiers[chosen].items()):
            a = np.atleast_2d(np.asarray(E_, float))
            e0 = (a if a.shape[0] >= a.shape[1] else a.T)[:, 0]
            segs.append(e0)
            base.append(float(np.max(np.abs(rhat_profile(e0, expose_lags)))))
        return segs, base, dict(tier=chosen, order=-1, source="raw_r2")
    # (c) plain arrays
    segs = [np.asarray(d[k], float).ravel() for k in keys if k != "p_star"]
    base = [float(np.max(np.abs(rhat_profile(s, expose_lags)))) for s in segs]
    print(f"  [F6] plain file: {len(segs)} array(s) as segments")
    return segs, base, dict(tier="plain", order=-1, source="plain")


# ----- scheduler configs (reuse hysteresis + dither from cps_companion) -----
def build_configs(L_grid, f_grid, dither_grid):
    cfgs = [("MEMLESS", dict(kind="memless"))]
    for L in L_grid:
        for f in f_grid:
            cfgs.append((f"CORR_L{L}_f{f}", dict(kind="corr", L=L, f=f)))
    for sd in dither_grid:
        cfgs.append(("DITHER_blind" if sd < 0 else f"DITHER_s{sd}",
                     dict(kind="dither", sigma=sd)))
    return cfgs

def fire_config(spec, z_cal, z_dep, G, k0, rng):
    if spec["kind"] == "memless":
        s_cal = np.where(np.arange(len(z_cal)) >= k0, np.abs(z_cal), np.nan)
        s_dep = np.where(np.arange(len(z_dep)) >= k0, np.abs(z_dep), np.nan)
        return E.fire_threshold(s_dep, E.thr_conformal(s_cal, G), k0)
    if spec["kind"] == "corr":
        eps = CC.calibrate_hysteresis(z_cal, spec["f"], spec["L"], G, k0)
        return CC.fire_hysteresis(z_dep, eps, spec["f"], spec["L"], k0)
    if spec["kind"] == "dither":
        s_cal = CC.score_dither(z_cal, spec["sigma"], rng)
        s_dep = CC.score_dither(z_dep, spec["sigma"], rng)
        s_cal[:k0] = np.nan; s_dep[:k0] = np.nan
        return E.fire_threshold(s_dep, E.thr_conformal(s_cal, G), k0)
    raise KeyError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--innov_file", default="results/R2_innovations_white.npz")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--budgets", default="0.02,0.05,0.10,0.20,0.30,0.50")
    ap.add_argument("--L_grid", default="2,4,6,10,20")
    ap.add_argument("--f_grid", default="0.0,0.15,0.3,0.5,0.7")
    ap.add_argument("--dither_grid", default="0.5,1.0,2.0,-1.0")
    ap.add_argument("--expose_lags", type=int, default=20)
    ap.add_argument("--sgn_windows", default="250,500,1000,2000")
    ap.add_argument("--k0", type=int, default=50)
    ap.add_argument("--cal_frac", type=float, default=0.5)
    ap.add_argument("--min_len", type=int, default=1500)
    ap.add_argument("--white_thresh", type=float, default=0.05,
                    help="segment is 'well-whitened' if baseline_maxr < this")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.budgets = "0.02,0.10,0.30"; a.L_grid = "4,10"; a.f_grid = "0.0,0.3"
        a.dither_grid = "1.0,-1.0"; a.sgn_windows = "250,500"; a.min_len = 800

    budgets = [float(x) for x in a.budgets.split(",")]
    L_grid = [int(x) for x in a.L_grid.split(",")]
    f_grid = [float(x) for x in a.f_grid.split(",")]
    dither_grid = [float(x) for x in a.dither_grid.split(",")]
    Ws = [int(x) for x in a.sgn_windows.split(",")]
    k0 = a.k0
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)

    segs_all, base_all, prov = load_clean_segments(a.innov_file, a.expose_lags)
    keep = [i for i, z in enumerate(segs_all) if len(z) >= a.min_len]
    segs = [segs_all[i] for i in keep]
    base = [base_all[i] for i in keep]
    n_seg = len(segs)
    if n_seg == 0:
        print("[F6] no usable segments."); return
    n_white = int(np.sum(np.array(base) < a.white_thresh))
    print(f"[F6] {n_seg} usable segment(s); baseline_maxr median "
          f"{np.median(base):.3f}; well-whitened (<{a.white_thresh}): "
          f"{n_white}/{n_seg} ({100*n_white/n_seg:.0f}%)")

    configs = build_configs(L_grid, f_grid, dither_grid)
    acf = E.GSGN["acf1"]; ts_acf = E.GSGN_TWOSIDED["acf1"]
    lb = E.GSGN["ljungbox"]; ts_lb = E.GSGN_TWOSIDED["ljungbox"]

    func_rows = []
    seg_white = {}                                   # segment id -> bool
    # window accumulators, split by cohort
    null = {("all", W): [] for W in Ws}; null.update({("white", W): [] for W in Ws})
    null_lbacc = {("all", W): [] for W in Ws}; null_lbacc.update({("white", W): [] for W in Ws})
    att = {}                                          # (cohort,name,G,W) -> [arrays]
    att_lb = {}

    t0 = time.time()
    for si, z in enumerate(segs):
        cut = int(a.cal_frac * len(z))
        z_cal, z_dep = z[:cut], z[cut:]
        if len(z_dep) < 2 * k0 + 50:
            continue
        S = float(np.var(z_cal)) or float(np.var(z))
        rng = np.random.default_rng(1000 + si)
        is_white = base[si] < a.white_thresh
        seg_white[si] = is_white
        cohorts = ["all"] + (["white"] if is_white else [])

        # NOATTACK baseline profile (per lag) on the deployment half
        rv0 = rhat_profile(z_dep[k0:], a.expose_lags)
        func_rows.append(dict(exp="F6b", segment=si, cohort=("white" if is_white else "nonwhite"),
                              config="NOATTACK", budget=0.0, S=S, rate=0.0, psi=0.0,
                              baseline_maxr=base[si], max_rhat=float(np.max(np.abs(rv0))),
                              max_drhat=0.0,
                              **{f"r{l+1}": rv0[l] for l in range(min(5, len(rv0)))},
                              **{f"dr{l+1}": 0.0 for l in range(min(5, len(rv0)))}))
        for W in Ws:
            v = windows_stat(z_dep[k0:], W, acf, ts_acf)
            vlb = windows_stat(z_dep[k0:], W, lb, ts_lb)
            for c in cohorts:
                null[(c, W)].append(v); null_lbacc[(c, W)].append(vlb)

        for G in budgets:
            for name, spec in configs:
                g = fire_config(spec, z_cal, z_dep, G, k0, rng)
                zc = CC.received(z_dep, g)[k0:]
                rv = rhat_profile(zc, a.expose_lags)
                drv = rv - rv0                             # ATTACK-INDUCED delta
                func_rows.append(dict(
                    exp="F6b", segment=si, cohort=("white" if is_white else "nonwhite"),
                    config=name, budget=G, S=S, rate=float(g[k0:].mean()),
                    psi=psi_of(z_dep, g, k0), baseline_maxr=base[si],
                    max_rhat=float(np.max(np.abs(rv))),
                    max_drhat=float(np.max(np.abs(drv))),
                    **{f"r{l+1}": rv[l] for l in range(min(5, len(rv)))},
                    **{f"dr{l+1}": drv[l] for l in range(min(5, len(drv)))}))
                for W in Ws:
                    va = windows_stat(zc, W, acf, ts_acf)
                    vl = windows_stat(zc, W, lb, ts_lb)
                    for c in cohorts:
                        att.setdefault((c, name, G, W), []).append(va)
                        att_lb.setdefault((c, name, G, W), []).append(vl)
        if (si + 1) % 10 == 0:
            print(f"  [F6] {si+1}/{n_seg} segments ({time.time()-t0:.0f}s)")

    # ---- detector nulls + pooled power, per cohort ----
    def pool(lst):
        arrs = [x for x in lst if x.size]
        return np.concatenate(arrs) if arrs else np.empty(0)
    thr_acf, far_acf, thr_lb = {}, {}, {}
    for (c, W), lst in null.items():
        vv = pool(lst)
        if vv.size:
            thr_acf[(c, W)] = float(np.quantile(vv, 0.99))
            far_acf[(c, W)] = float(np.mean(vv > thr_acf[(c, W)]))
    for (c, W), lst in null_lbacc.items():
        vl = pool(lst)
        if vl.size:
            thr_lb[(c, W)] = float(np.quantile(vl, 0.99))

    power_rows = []
    for (c, name, G, W), lst in att.items():
        if (c, W) not in thr_acf:
            continue
        va = pool(lst); vl = pool(att_lb[(c, name, G, W)])
        if va.size == 0:
            continue
        power_rows.append(dict(
            exp="F6b", cohort=c, config=name, budget=G, W=W,
            power_acf1=float(np.mean(va > thr_acf[(c, W)])),
            power_ljungbox=(float(np.mean(vl > thr_lb[(c, W)]))
                            if ((c, W) in thr_lb and vl.size) else np.nan),
            far_acf1=far_acf.get((c, W), np.nan), n_windows=int(va.size)))

    def _write(rows, name):
        if not rows:
            print(f"  [F6] no rows for {name}"); return
        keys = sorted({k for r in rows for k in r})
        pth = Path(a.out_dir) / f"{name}.csv"
        with open(pth, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        print(f"  -> {pth}  ({len(rows)} rows)")
    _write(func_rows, "F6b_functional")
    _write(power_rows, "F6b_power")

    # =====================================================================
    #  SUMMARY  (printed twice: cohort = all, then white-only)
    #  Headline exposure is the ATTACK-INDUCED delta max_drhat.
    # =====================================================================
    ff = {}
    for r in func_rows:
        ff[(r["segment"], r["budget"], r["config"])] = r
    all_ids = sorted({r["segment"] for r in func_rows})
    white_ids = [s for s in all_ids if seg_white.get(s, False)]
    corr_names = [n for n, _ in configs if n.startswith("CORR")]
    dith_names = [n for n, _ in configs if n.startswith("DITHER")]
    pooled_power = {(r["cohort"], r["config"], r["budget"], r["W"]): r["power_acf1"]
                    for r in power_rows}
    summary = []

    def emit(cohort, seg_ids):
        if not seg_ids:
            print(f"\n[F6] cohort '{cohort}': no segments."); return
        cal_set = [s for s in seg_ids if s % 2 == 0] or seg_ids
        hold_set = [s for s in seg_ids if s % 2 == 1] or seg_ids
        Wref = max([W for W in Ws if (cohort, W) in thr_acf], default=Ws[-1])
        print("\n" + "=" * 88)
        print(f"F6 SUMMARY  cohort={cohort}  (n={len(seg_ids)}; "
              f"exposure = ATTACK-INDUCED max_l|Dr_l|; power at W={Wref})")
        print("=" * 88)
        print(f"{'budget':>7} {'MEM_dr':>7} {'MEM_pow':>8} {'MEM_psi':>8} "
              f"{'CORRtun_dr':>11} {'CORRtun_psi':>12} {'best(L,f)':>12} "
              f"{'CORRxfer_dr':>12} {'DITH_dr':>8}")
        def val(s, G, cfg, key):
            r = ff.get((s, G, cfg)); return r[key] if r else np.nan
        def mean_dr(cfg, segset, G):
            v = [val(s, G, cfg, "max_drhat") for s in segset if (s, G, cfg) in ff]
            return float(np.nanmean(v)) if v else np.nan
        for G in budgets:
            mem_dr = mean_dr("MEMLESS", seg_ids, G)
            mem_psi = float(np.nanmean([val(s, G, "MEMLESS", "psi") for s in seg_ids]))
            mem_pow = pooled_power.get((cohort, "MEMLESS", G, Wref), np.nan)
            tun_dr, tun_psi, picks = [], [], {}
            for s in seg_ids:
                cand = [(val(s, G, c, "max_drhat"), val(s, G, c, "psi"), c)
                        for c in corr_names if (s, G, c) in ff]
                if not cand:
                    continue
                dr, ps, c = min(cand, key=lambda t: t[0])
                tun_dr.append(dr); tun_psi.append(ps); picks[c] = picks.get(c, 0) + 1
            tdr = float(np.nanmean(tun_dr)) if tun_dr else np.nan
            tps = float(np.nanmean(tun_psi)) if tun_psi else np.nan
            bp = max(picks, key=picks.get).replace("CORR_", "") if picks else "-"
            best_cfg = min(corr_names, key=lambda c: (mean_dr(c, cal_set, G)
                           if not np.isnan(mean_dr(c, cal_set, G)) else 1e9))
            xfer = mean_dr(best_cfg, hold_set, G)
            dith = float(np.nanmean([min(val(s, G, c, "max_drhat") for c in dith_names
                                     if (s, G, c) in ff) for s in seg_ids
                                     if any((s, G, c) in ff for c in dith_names)]))
            print(f"{G:>7.2f} {mem_dr:>7.3f} {mem_pow:>8.3f} {mem_psi:>8.3f} "
                  f"{tdr:>11.3f} {tps:>12.3f} {bp:>12} {xfer:>12.3f} {dith:>8.3f}")
            summary.append(dict(cohort=cohort, budget=G, memless_dr=mem_dr,
                                memless_power=mem_pow, memless_psi=mem_psi,
                                corr_tuned_dr=tdr, corr_tuned_psi=tps,
                                corr_tuned_bestpick=bp, corr_transfer_cfg=best_cfg,
                                corr_transfer_dr=xfer, dither_dr=dith,
                                n_segments=len(seg_ids)))

    emit("all", all_ids)
    emit("white", white_ids)
    _write(summary, "F6b_summary")
    print("\nColumns: *_dr = ATTACK-INDUCED exposure max_l|rhat_l(att)-rhat_l(nom)| "
          "(lower=stealthier, 0=no signature added); MEM_pow = lag-1 whiteness power "
          "of the memoryless rule vs a nominal null; CORRtun = best (L,f) per segment; "
          "CORRxfer = one config fit on half the segments, applied to the other half.")
    print(f"\nProvenance: {prov}.  Done in {time.time()-t0:.0f}s. "
          f"Paste BOTH SUMMARY blocks, or send {a.out_dir}/F6b_*.csv")


if __name__ == "__main__":
    main()
