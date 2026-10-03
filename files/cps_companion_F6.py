#!/usr/bin/env python
# coding: utf-8
"""
=============================================================================
 cps_companion_F6.py  --  Upgraded F6: exposure on real CSU CAN innovations

 Fixes the two problems found in the first F6 pass:
   (1) the interleave split DECIMATES the series (its lag-1 = the original's
       lag-2), so it under-reports exposure -> we use a CONTIGUOUS held-out
       calibration split, which preserves lag structure and is deployable.
   (2) a single CORR config transplanted from synthetic N2 did not transfer
       -> we TUNE (L, f) per deployment and also run an honest single-config
       TRANSFER test (fit on half the segments, apply to the held-out half).

 It reuses the validated hysteresis scheduler, loader and detectors from
 cps_companion.py / cps_experiments.py.  No torch, no learning.

 Modes reported, per budget:
   - NOATTACK   : whiteness of the VARX innovations themselves (control:
                  how much exposure is baseline vs. attack-induced)
   - MEMLESS    : the letter's rule (B3)
   - CORR-tuned : best (L, f) per segment  (per-deployment tuning)
   - CORR-xfer  : one (L, f) fit on a calibration set of segments, applied to
                  held-out segments (does one config generalise on real data?)
   - DITHER     : randomisation baseline, best sigma_d per segment

 Metrics: exposure functional max_l|r_l|/S and per-lag r_l (l=1..expose_lags);
 damage Psi; realised rate; and pooled whiteness DETECTOR POWER (lag-1 acf and
 Ljung-Box) at several windows, with nulls calibrated on the real nominal stream.

 Outputs (results/):
   F6b_functional.csv  -- per segment x budget x config: rate, psi, max_rhat, r1..rK
   F6b_power.csv       -- per config x budget x W: pooled acf1 / ljungbox power + nominal FAR
   F6b_summary.csv     -- compact per-budget headline (also printed to stdout)

 Usage:
   python cps_companion_F6.py --innov_file R2_innovations.npz
   python cps_companion_F6.py --innov_file R2_innovations.npz --quick
=============================================================================
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import cps_experiments as E
import cps_companion as CC


# ----------------------------------------------------------------------------
def psi_of(z_dep, g, k0):
    z2 = z_dep[k0:] ** 2
    gk = g[k0:]
    return float(z2[gk].sum() / z2.sum()) if gk.any() and z2.sum() > 0 else 0.0


def rhat_vec(zc, lags, S):
    """per-lag r_l/S on a received stream (list length = lags)."""
    zc = np.asarray(zc, float)
    return [float(np.mean(zc[:-l] * zc[l:]) / S) for l in range(1, lags + 1)]


def windows_stat(x, W, fn, two_sided):
    """non-overlapping window statistics; [] if the stream is too short."""
    if len(x) < 2 * W:
        return np.empty(0)
    return E.block_stats(np.asarray(x, float), W, fn, two_sided)


# ----------------------------------------------------------------------------
def build_configs(L_grid, f_grid, dither_grid):
    cfgs = [("MEMLESS", dict(kind="memless"))]
    for L in L_grid:
        for f in f_grid:
            cfgs.append((f"CORR_L{L}_f{f}", dict(kind="corr", L=L, f=f)))
    for sd in dither_grid:
        tag = "DITHER_blind" if sd < 0 else f"DITHER_s{sd}"
        cfgs.append((tag, dict(kind="dither", sigma=sd)))
    return cfgs


def fire_config(spec, z_cal, z_dep, G, k0, rng):
    """Return the deployment firing mask for a given config spec."""
    if spec["kind"] == "memless":
        s_cal = np.where(np.arange(len(z_cal)) >= k0, np.abs(z_cal), np.nan)
        s_dep = np.where(np.arange(len(z_dep)) >= k0, np.abs(z_dep), np.nan)
        eps = E.thr_conformal(s_cal, G)
        return E.fire_threshold(s_dep, eps, k0)
    if spec["kind"] == "corr":
        eps = CC.calibrate_hysteresis(z_cal, spec["f"], spec["L"], G, k0)
        return CC.fire_hysteresis(z_dep, eps, spec["f"], spec["L"], k0)
    if spec["kind"] == "dither":
        sd = spec["sigma"]
        s_cal = CC.score_dither(z_cal, sd, rng)
        s_dep = CC.score_dither(z_dep, sd, rng)
        s_cal[:k0] = np.nan
        s_dep[:k0] = np.nan
        eps = E.thr_conformal(s_cal, G)
        return E.fire_threshold(s_dep, eps, k0)
    raise KeyError(spec)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--innov_file", default="R2_innovations.npz")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--budgets", default="0.02,0.05,0.10,0.20,0.30,0.50")
    ap.add_argument("--L_grid", default="2,4,6,10,20")
    ap.add_argument("--f_grid", default="0.0,0.15,0.3,0.5,0.7")
    ap.add_argument("--dither_grid", default="0.5,1.0,2.0,-1.0")  # -1 = blind
    ap.add_argument("--expose_lags", type=int, default=20)
    ap.add_argument("--sgn_windows", default="250,500,1000,2000")
    ap.add_argument("--k0", type=int, default=50)
    ap.add_argument("--cal_frac", type=float, default=0.5)
    ap.add_argument("--min_len", type=int, default=1500)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    if a.quick:
        a.budgets = "0.02,0.10,0.30"
        a.L_grid = "4,10"
        a.f_grid = "0.0,0.3"
        a.dither_grid = "1.0,-1.0"
        a.sgn_windows = "250,500"
        a.min_len = 800

    budgets = [float(x) for x in a.budgets.split(",")]
    L_grid = [int(x) for x in a.L_grid.split(",")]
    f_grid = [float(x) for x in a.f_grid.split(",")]
    dither_grid = [float(x) for x in a.dither_grid.split(",")]
    Ws = [int(x) for x in a.sgn_windows.split(",")]
    k0 = a.k0
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)

    segs = CC._load_innovations(a.innov_file)
    segs = [z for z in segs if len(z) >= a.min_len]
    n_seg = len(segs)
    print(f"[F6] {n_seg} usable segment(s) (>= {a.min_len} samples); "
          f"lengths min/med/max = "
          f"{min(map(len,segs))}/{int(np.median([len(z) for z in segs]))}/"
          f"{max(map(len,segs))}")
    if n_seg == 0:
        print("[F6] nothing to do."); return

    configs = build_configs(L_grid, f_grid, dither_grid)
    acf = E.GSGN["acf1"]; ts_acf = E.GSGN_TWOSIDED["acf1"]
    lb = E.GSGN["ljungbox"]; ts_lb = E.GSGN_TWOSIDED["ljungbox"]

    func_rows = []
    # pooled window-stat accumulators
    null_acf = {W: [] for W in Ws}          # nominal (no-attack) windows
    null_lb = {W: [] for W in Ws}
    att_acf = {}                            # (cfg_name, G, W) -> list
    att_lb = {}

    t0 = time.time()
    for si, z in enumerate(segs):
        cut = int(a.cal_frac * len(z))
        z_cal, z_dep = z[:cut], z[cut:]
        if len(z_dep) < 2 * k0 + 50:
            continue
        S = float(np.var(z_cal)) or float(np.var(z))
        rng = np.random.default_rng(1000 + si)

        # ---- NOATTACK baseline (whiteness of the innovations themselves) ----
        rv0 = rhat_vec(z_dep[k0:], a.expose_lags, S)
        func_rows.append(dict(exp="F6b", segment=si, config="NOATTACK",
                              budget=0.0, S=S, rate=0.0, psi=0.0,
                              max_rhat=float(np.max(np.abs(rv0))),
                              **{f"r{l+1}": rv0[l] for l in range(min(5, len(rv0)))}))
        for W in Ws:
            v = windows_stat(z_dep[k0:], W, acf, ts_acf)
            null_acf[W].append(v)
            null_lb[W].append(windows_stat(z_dep[k0:], W, lb, ts_lb))

        # ---- every config at every budget ----
        for G in budgets:
            for name, spec in configs:
                g = fire_config(spec, z_cal, z_dep, G, k0, rng)
                zc = CC.received(z_dep, g)[k0:]
                rv = rhat_vec(zc, a.expose_lags, S)
                func_rows.append(dict(
                    exp="F6b", segment=si, config=name, budget=G, S=S,
                    rate=float(g[k0:].mean()), psi=psi_of(z_dep, g, k0),
                    max_rhat=float(np.max(np.abs(rv))),
                    **{f"r{l+1}": rv[l] for l in range(min(5, len(rv)))}))
                for W in Ws:
                    att_acf.setdefault((name, G, W), []).append(
                        windows_stat(zc, W, acf, ts_acf))
                    att_lb.setdefault((name, G, W), []).append(
                        windows_stat(zc, W, lb, ts_lb))
        if (si + 1) % 10 == 0:
            print(f"  [F6] {si+1}/{n_seg} segments  ({time.time()-t0:.0f}s)")

    # ---- detector nulls (1% FAR) and pooled power ----
    thr_acf, far_acf, thr_lb, far_lb = {}, {}, {}, {}
    for W in Ws:
        vv = np.concatenate([x for x in null_acf[W] if x.size]) if any(
            x.size for x in null_acf[W]) else np.empty(0)
        if vv.size:
            thr_acf[W] = float(np.quantile(vv, 0.99)); far_acf[W] = float(np.mean(vv > thr_acf[W]))
        vl = np.concatenate([x for x in null_lb[W] if x.size]) if any(
            x.size for x in null_lb[W]) else np.empty(0)
        if vl.size:
            thr_lb[W] = float(np.quantile(vl, 0.99)); far_lb[W] = float(np.mean(vl > thr_lb[W]))

    power_rows = []
    for (name, G, W), lst in att_acf.items():
        if W not in thr_acf:
            continue
        va = np.concatenate([x for x in lst if x.size]) if any(x.size for x in lst) else np.empty(0)
        vl = np.concatenate([x for x in att_lb[(name, G, W)] if x.size]) if any(
            x.size for x in att_lb[(name, G, W)]) else np.empty(0)
        if va.size == 0:
            continue
        power_rows.append(dict(
            exp="F6b", config=name, budget=G, W=W,
            power_acf1=float(np.mean(va > thr_acf[W])),
            power_ljungbox=float(np.mean(vl > thr_lb[W])) if (W in thr_lb and vl.size) else np.nan,
            far_acf1=far_acf.get(W, np.nan), n_windows=int(va.size)))

    # ---- write raw ----
    def _write(rows, name):
        if not rows:
            print(f"  [F6] no rows for {name}"); return
        keys = sorted({k for r in rows for k in r})
        p = Path(a.out_dir) / f"{name}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  -> {p}  ({len(rows)} rows)")
    _write(func_rows, "F6b_functional")
    _write(power_rows, "F6b_power")

    # =====================================================================
    #  SUMMARY: per-deployment tuning + transfer test, printed for pasting
    # =====================================================================
    import statistics as st
    ff = {}
    for r in func_rows:
        ff.setdefault((r["segment"], r["budget"], r["config"]), r)

    seg_ids = sorted({r["segment"] for r in func_rows})
    corr_names = [n for n, _ in configs if n.startswith("CORR")]
    dith_names = [n for n, _ in configs if n.startswith("DITHER")]

    # transfer: choose the single CORR config with lowest MEAN max_rhat on the
    # even segments (calibration set); evaluate on odd (held-out) segments.
    cal_set = [s for s in seg_ids if s % 2 == 0]
    hold_set = [s for s in seg_ids if s % 2 == 1] or seg_ids

    def mean_maxr(cfg, segset, G):
        vals = [ff[(s, G, cfg)]["max_rhat"] for s in segset if (s, G, cfg) in ff]
        return float(np.mean(vals)) if vals else np.nan

    pooled_power = {(r["config"], r["budget"], r["W"]): r["power_acf1"] for r in power_rows}
    Wref = max([W for W in Ws if W in thr_acf], default=Ws[0])

    summary = []
    print("\n" + "=" * 78)
    print(f"F6 SUMMARY  (contiguous split; exposure=max_l|r_l|/S over l=1..{a.expose_lags};"
          f" power at W={Wref})")
    print("=" * 78)
    header = (f"{'budget':>7} {'NOATK_r':>8} {'MEM_r':>7} {'MEM_pow':>8} "
              f"{'CORRtun_r':>10} {'CORRtun_psi':>11} {'best(L,f)':>12} "
              f"{'CORRxfer_r':>11} {'DITH_r':>7} {'MEM_psi':>8}")
    print(header)
    for G in budgets:
        noatk = float(np.mean([ff[(s, 0.0, "NOATTACK")]["max_rhat"]
                               for s in seg_ids if (s, 0.0, "NOATTACK") in ff]))
        mem_r = float(np.mean([ff[(s, G, "MEMLESS")]["max_rhat"]
                               for s in seg_ids if (s, G, "MEMLESS") in ff]))
        mem_psi = float(np.mean([ff[(s, G, "MEMLESS")]["psi"]
                                 for s in seg_ids if (s, G, "MEMLESS") in ff]))
        mem_pow = pooled_power.get(("MEMLESS", G, Wref), float("nan"))
        # per-deployment tuning: best CORR per segment by max_rhat
        tun_r, tun_psi, picks = [], [], {}
        for s in seg_ids:
            cand = [(ff[(s, G, c)]["max_rhat"], ff[(s, G, c)]["psi"], c)
                    for c in corr_names if (s, G, c) in ff]
            if not cand:
                continue
            mr, ps, c = min(cand, key=lambda t: t[0])
            tun_r.append(mr); tun_psi.append(ps); picks[c] = picks.get(c, 0) + 1
        tun_r_m = float(np.mean(tun_r)) if tun_r else float("nan")
        tun_psi_m = float(np.mean(tun_psi)) if tun_psi else float("nan")
        best_pick = max(picks, key=picks.get).replace("CORR_", "") if picks else "-"
        # transfer: pick config on cal_set, eval on hold_set
        best_cfg = min(corr_names, key=lambda c: (mean_maxr(c, cal_set, G)
                       if not np.isnan(mean_maxr(c, cal_set, G)) else 1e9))
        xfer_r = mean_maxr(best_cfg, hold_set, G)
        dith_r = float(np.mean([min(ff[(s, G, c)]["max_rhat"] for c in dith_names
                                    if (s, G, c) in ff)
                                for s in seg_ids
                                if any((s, G, c) in ff for c in dith_names)]))
        print(f"{G:>7.2f} {noatk:>8.3f} {mem_r:>7.3f} {mem_pow:>8.3f} "
              f"{tun_r_m:>10.3f} {tun_psi_m:>11.3f} {best_pick:>12} "
              f"{xfer_r:>11.3f} {dith_r:>7.3f} {mem_psi:>8.3f}")
        summary.append(dict(budget=G, noattack_maxr=noatk, memless_maxr=mem_r,
                            memless_power=mem_pow, memless_psi=mem_psi,
                            corr_tuned_maxr=tun_r_m, corr_tuned_psi=tun_psi_m,
                            corr_tuned_bestpick=best_pick,
                            corr_transfer_cfg=best_cfg, corr_transfer_maxr=xfer_r,
                            dither_best_maxr=dith_r))
    _write(summary, "F6b_summary")
    print("\nColumns: *_r = exposure functional max_l|r_l|/S (lower=stealthier); "
          "MEM_pow = lag-1 whiteness power of the letter's rule; "
          "CORRtun = best (L,f) per segment; CORRxfer = one config fit on half "
          "the segments, applied to the other half; DITH = best dither per seg.")
    print(f"\nDone in {time.time()-t0:.0f}s. Paste the SUMMARY block above, or send "
          f"{a.out_dir}/F6b_*.csv")


if __name__ == "__main__":
    main()
