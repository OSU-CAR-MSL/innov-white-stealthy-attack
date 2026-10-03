#!/usr/bin/env python
# coding: utf-8
"""
=============================================================================
 cps_companion.py  --  Companion-paper experiment engine (F1 ... F6)

 Paper : "Optimal Attack Scheduling under Sign-Sensitive Detection:
          Why Stealth Requires Memory"   (companion to the L-CSS letter)

 This module REUSES the validated machinery of cps_experiments.py (plant,
 vectorised simulation, divergence/damage, magnitude scores, detectors,
 conformal threshold, Gsgn statistics, N_eff).  It adds only what is new to
 the companion:

     * BLOCK_L      -- threshold + hold-L memory scheduler (transparent baseline)
     * CORR         -- the proposed corrected-score scheduler (causal one-pass)
     * F1 ... F6    -- the six companion experiments

 Scope decisions (per plan):
   - NO autoencoder / PCA / learning-based schedulers.  torch is never imported.
   - Regimes are N1 (Gaussian) and N2 (heavy-tailed) ONLY.  No MM, no TV.
   - Every scheduler is sign-invariant (magnitude-measurable), so Thm. 1 of the
     letter holds pathwise and Gmag FAR stays nominal for all of them (F-Gmag
     is a one-line control, not a swept axis).

 Every experiment writes results/<Fk>_companion.csv (tidy long format) and a
 _meta.json.  Aggregation/figures are downstream (collect_companion.py).

 Usage
 -----
   python cps_companion.py --exp F1                 # dither frontier
   python cps_companion.py --exp all                # F1..F5 (F6 needs data)
   python cps_companion.py --exp F6 --innov_file csu_innovations.npz
   python cps_companion.py --exp array:0            # SLURM array element
   python cps_companion.py --exp all --quick        # smoke test (~2 min)
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ---- reuse the letter's engine (no torch: we never touch the AE) -------------
import cps_experiments as E
from cps_experiments import (
    A_P1, divergence, damage_metrics,
    score_magnitude, score_energy, score_diff, score_ar, fit_ar,
    thr_conformal, fire_threshold, fire_top_rate,
    block_stats, GSGN, GSGN_TWOSIDED, stat_acf1,
    eff_sample_size, save_rows,
)

# ---- bridge to the letter's simulate_stream/seed_of, which take an E.Config --
_BRIDGE = None          # set in main() from the companion config


def simulate_stream(regime, T, seed, cfg=None, burn=None):
    return E.simulate_stream(regime, T, seed, _BRIDGE, burn)


_EXTRA_SEED = {"dither": 960_000}


def seed_of(regime, kind, i=0, cfg=None):
    try:
        return E.seed_of(regime, kind, i, _BRIDGE)
    except KeyError:
        # kind not in the engine's seed table (e.g. 'dither' on older engines)
        return _BRIDGE.seed + E._REG_OFF[regime] + _EXTRA_SEED.get(kind, 970_000) + i


def score_dither(z, sigma_d, rng):
    """Dithered score s_k = |z_k| + sigma_d * xi_k, xi independent (F1).
    sigma_d < 0 returns an innovation-independent (blind) score.  Defined
    locally so the companion does not depend on the engine having E9."""
    m = np.abs(np.asarray(z, dtype=float))
    if sigma_d < 0:
        return rng.standard_normal(len(m))
    return m + float(sigma_d) * rng.standard_normal(len(m))


# =============================================================================
# S0.  Configuration (companion-specific grids)
# =============================================================================
@dataclass
class CompCfg:
    seed: int = 0
    # stream lengths -- identical protocol to the letter
    T_burn: int = 4_000
    N_cal: int = 5_000            # calibration record (N_eff story lives at 5000)
    T_dep: int = 50_000
    T_train: int = 60_000         # AR fit stream (disjoint)
    n_mc: int = 50                # deployment seeds
    n_cal_draws: int = 200        # calibration draws (F3: variance is the point)

    budgets: tuple = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
    regimes: tuple = ("N1", "N2")

    # scheduler windows / memory
    L: int = 50                   # generic score window (energy/AR)
    ar_order: int = 5
    L_grid: tuple = (1, 2, 4, 6, 10, 20, 50)     # memory-length ablation
    f_grid: tuple = (1.0, 0.7, 0.5, 0.3, 0.15, 0.0)  # CORR continue-gate (1=B3, 0=block)
    exposure_lags: int = 10       # L in |r_ell| <= delta  (ell = 1..L)

    # dither (F1)
    dither_grid: tuple = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, -1.0)  # -1 = blind

    # detectors
    fa_target: float = 0.01
    sgn_windows: tuple = (500, 2000, 5000, 20_000)
    n_null_windows: int = 400
    neff_lags: int = 200

    out_dir: str = "results"

    def dirs(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)


def quick(c: CompCfg) -> CompCfg:
    c.T_burn, c.N_cal, c.T_dep, c.T_train = 500, 2_000, 8_000, 12_000
    c.n_mc, c.n_cal_draws = 8, 30
    c.budgets = (0.02, 0.10, 0.30)
    c.L_grid = (1, 4, 10)
    c.f_grid = (1.0, 0.5, 0.0)
    c.dither_grid = (0.0, 1.0, -1.0)
    c.sgn_windows, c.n_null_windows = (500, 2000), 120
    c.neff_lags = 80
    return c


# =============================================================================
# S1.  The two new schedulers
# =============================================================================
# ---- BLOCK_L : fire on a magnitude threshold, then hold for L steps ----------
#   The transparent memory baseline.  Removes firing/non-firing boundaries
#   (letter Eq. r1 = -2(T10+T01)) by firing in runs of length L, so exposure
#   per firing falls ~1/L while Psi stays above blind.  It is the beta -> inf
#   limit of CORR.
def fire_block(z, thr, L, k0):
    a = np.abs(z)
    g = np.zeros(len(z), dtype=bool)
    last = -10 ** 9
    for i in np.flatnonzero(a > thr):
        if i < k0:
            continue
        if i >= last + L:
            g[i:i + L] = True
            last = i
    g[:k0] = False
    return g


def block_thr_for_rate(z, L, rate, k0, iters=45):
    a = np.abs(z)
    lo, hi = 0.0, float(a.max())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = fire_block(z, mid, L, k0).mean()
        lo, hi = (mid, hi) if r > rate else (lo, mid)
    return hi


# ---- CORR : the proposed corrected-score scheduler (hysteresis form) ---------
#   A threshold on a corrected score whose correction is a function of the past
#   window of magnitudes and decisions (matching the structure theorem):
#     * START a flip when |z_k| > eps_hi           (a rare tail crossing)
#     * CONTINUE for up to L steps while |z_k| > eps_lo = f * eps_hi
#   so an isolated tail spike is extended into a short run, removing the
#   firing/non-firing boundary that carries r_ell = -2(T10+T01).  The effective
#   threshold is lowered (eps_hi -> eps_lo) only when the past L decisions put
#   us inside a run, i.e. the "correction" c_k depends on past magnitudes and
#   past decisions alone -> magnitude-measurable, Thm. 1 holds, model-free.
#     f = 1  (eps_lo = eps_hi)  => correction inactive => exactly B3.
#     f -> 0                    => approaches the BLOCK rule of length L.
#   eps_hi is calibrated so the TOTAL realised rate equals Gbar (budget-neutral,
#   non-cascading), so the comparison against B3/dither is at matched budget.
def fire_hysteresis(z, eps_hi, f, L, k0):
    a = np.abs(z)
    eps_lo = f * eps_hi
    n = len(z)
    g = np.zeros(n, dtype=bool)
    run = 0
    for k in range(k0, n):
        if a[k] > eps_hi:
            g[k] = True; run = 1
        elif 0 < run < L and a[k] > eps_lo:
            g[k] = True; run += 1
        else:
            run = 0
    return g


def calibrate_hysteresis(z_cal, f, L, Gbar, k0, iters=45):
    """Bisection on eps_hi so the realised rate of the hysteresis rule = Gbar
    (rate is monotone decreasing in eps_hi)."""
    a = np.abs(z_cal)
    lo, hi = 0.0, float(a.max())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = fire_hysteresis(z_cal, mid, f, L, k0)[k0:].mean()
        lo, hi = (mid, hi) if r > Gbar else (lo, mid)
    return hi


# =============================================================================
# S2.  Exposure functional  r_ell = E[z^c_k z^c_{k+ell}]  (the constraint object)
# =============================================================================
def received(z, gamma):
    return np.where(gamma, -z, z)


def dmg(z, es, gamma, K, k0):
    """psi, tr(D), realised rate, excess MSE, mean|z| at firing -- computed
    locally so the companion does not depend on the engine's damage_metrics
    version (older ones omit psi/tr_D)."""
    z = np.asarray(z, float)
    gk = gamma[k0:]
    z2 = z[k0:] ** 2
    psi = float(z2[gk].sum() / z2.sum()) if gk.any() and z2.sum() > 0 else 0.0
    d = divergence(z, gamma, K)                      # (T, 2) divergence path
    dK = d[k0:]
    trD = float(np.mean(np.sum(dK * dK, axis=1)))
    ea = es[k0:] + dK
    mse = float(np.mean(np.sum(ea * ea, axis=1)))
    nom = es[k0:]
    mse_nom = float(np.mean(np.sum(nom * nom, axis=1)))
    fired = np.abs(z[k0:])[gk]
    return dict(psi=psi, tr_D=trD, rate=float(gk.mean()),
                excess=mse - mse_nom,
                mean_abs_z_fired=float(fired.mean()) if gk.any() else 0.0)


def exposure_rhat(zc, lags, S):
    """max_l |r_l|/S and the per-lag vector, on the received sequence."""
    zc = np.asarray(zc, dtype=float)
    out = []
    for l in range(1, lags + 1):
        out.append(float(np.mean(zc[:-l] * zc[l:]) / S))
    out = np.array(out)
    return float(np.max(np.abs(out))), out


# =============================================================================
# S3.  Shared helpers
# =============================================================================
def gsgn_nulls(regime, cfg, k0):
    """Empirical (1-fa) thresholds + realised nominal FAR for every Gsgn test
    and window, from independent nominal chunks."""
    Wmax = max(cfg.sgn_windows)
    chunks = [simulate_stream(regime, Wmax + cfg.L,
                              seed_of(regime, "chunk", j), cfg)
              for j in range(cfg.n_null_windows)]
    z_nom = [c["z"][k0:] for c in chunks]
    thr, far = {}, {}
    for tname, fn in GSGN.items():
        for W in cfg.sgn_windows:
            v = np.concatenate([block_stats(zz, W, fn, GSGN_TWOSIDED[tname])
                                for zz in z_nom])
            thr[(tname, W)] = float(np.quantile(v, 1 - cfg.fa_target))
            far[(tname, W)] = float(np.mean(v > thr[(tname, W)]))
    return thr, far


def gsgn_power(z_att_list, thr, cfg):
    """Power of each (test, W) on a list of attacked received chunks."""
    out = {}
    for tname, fn in GSGN.items():
        for W in cfg.sgn_windows:
            v = np.concatenate([block_stats(zz, W, fn, GSGN_TWOSIDED[tname])
                                for zz in z_att_list])
            out[(tname, W)] = float(np.mean(v > thr[(tname, W)]))
    return out


def fit_ar_coef(regime, cfg):
    tr = simulate_stream(regime, cfg.T_train, seed_of(regime, "train"), cfg)
    return fit_ar(np.abs(tr["z"]), cfg.ar_order), tr["S_design"]


def plant_S_K():
    _, S, K = E.solve_kf(E.A_P1, E.C_P1, E.Q_P1, E.R_P1)
    return S, K


# =============================================================================
# F1.  Dithering is not a stealth dial   (matches E9)
# =============================================================================
def run_F1(cfg: CompCfg):
    S, K = plant_S_K()
    k0 = cfg.L
    rows = []
    for regime in cfg.regimes:
        thr, far = gsgn_nulls(regime, cfg, k0)
        for sd in cfg.dither_grid:
            for G in cfg.budgets:
                psis, trDs, exc, rates = [], [], [], []
                z_att = {W: [] for W in cfg.sgn_windows}
                for s in range(cfg.n_mc):
                    cal = simulate_stream(regime, cfg.N_cal + cfg.L,
                                          seed_of(regime, "cal", s), cfg)
                    dep = simulate_stream(regime, cfg.T_dep,
                                          seed_of(regime, "dep", s), cfg)
                    rng = np.random.default_rng(seed_of(regime, "dither", s))
                    s_cal = score_dither(cal["z"], sd, rng)
                    s_dep = score_dither(dep["z"], sd, rng)
                    s_cal[:k0] = np.nan
                    s_dep[:k0] = np.nan
                    eps = thr_conformal(s_cal, G)
                    g = fire_threshold(s_dep, eps, k0)
                    dm = dmg(dep["z"], dep["es"], g, K, k0)
                    psis.append(dm["psi"]); trDs.append(dm["tr_D"])
                    exc.append(dm["excess"])
                    rates.append(dm["rate"])
                    zc = received(dep["z"], g)[k0:]
                    for W in cfg.sgn_windows:
                        z_att[W].append(zc)
                for W in cfg.sgn_windows:
                    pw = float(np.mean(block_stats(np.concatenate(z_att[W]),
                               W, stat_acf1, True) > thr[("acf1", W)]))
                    rows.append(dict(
                        exp="F1", regime=regime, scheduler="DITHER",
                        sigma_d_over_sqrtS=sd, blind=bool(sd < 0), budget=G, W=W,
                        psi=float(np.mean(psis)), tr_D=float(np.mean(trDs)),
                        excess_mse=float(np.mean(exc)),
                        realized_rate=float(np.mean(rates)),
                        power_acf1=pw, far_nominal=far[("acf1", W)]))
        print(f"  [F1 {regime}] done")
    return save_rows(rows, "F1_dither", cfg=_BRIDGE, meta=dict(dither_grid=list(cfg.dither_grid)))


# =============================================================================
# F2.  Found memory rules at matched realised rate
# =============================================================================
F2_SCHEDS = ["MEMLESS", "BLOCK", "AR", "ENERGY", "MAGDIFF"]


def _score_for(name, z, arcoef, cfg):
    if name in ("MEMLESS", "BLOCK"):
        return score_magnitude(z).copy()          # BLOCK handled specially
    if name == "AR":
        return score_ar(z, arcoef, cfg.ar_order, cfg.L)
    if name == "ENERGY":
        return score_energy(z, cfg.L)
    if name == "MAGDIFF":
        return score_diff(z)
    raise KeyError(name)


def run_F2(cfg: CompCfg):
    S, K = plant_S_K()
    k0 = cfg.L
    rows = []
    for regime in cfg.regimes:
        arcoef, _ = fit_ar_coef(regime, cfg)
        for G in cfg.budgets:
            for s in range(cfg.n_mc):
                dep = simulate_stream(regime, cfg.T_dep,
                                      seed_of(regime, "dep", s), cfg)
                for name in F2_SCHEDS:
                    if name == "BLOCK":
                        thr = block_thr_for_rate(dep["z"], cfg.L, G, k0)
                        g = fire_block(dep["z"], thr, cfg.L, k0)
                    else:
                        sc = _score_for(name, dep["z"], arcoef, cfg)
                        sc[:k0] = np.nan
                        g = fire_top_rate(sc, G, k0)     # matched realised rate
                    dm = dmg(dep["z"], dep["es"], g, K, k0)
                    zc = received(dep["z"], g)[k0:]
                    mx, rv = exposure_rhat(zc, cfg.exposure_lags, S)
                    # mean burst length
                    gk = g[k0:]
                    runs = 1 + int(np.sum(gk[1:] != gk[:-1])) if gk.any() else 1
                    burst = float(np.sum(gk) / max(runs / 2.0, 1.0))
                    rows.append(dict(
                        exp="F2", regime=regime, scheduler=name, budget=G,
                        seed=s, psi=dm["psi"], excess_mse=dm["excess"],
                        tr_D=dm["tr_D"], realized_rate=dm["rate"],
                        mean_abs_z_fired=dm["mean_abs_z_fired"],
                        mean_burst_len=burst, max_rhat=mx,
                        r1=rv[0], r2=rv[1] if len(rv) > 1 else np.nan,
                        r3=rv[2] if len(rv) > 2 else np.nan))
        print(f"  [F2 {regime}] done")
    return save_rows(rows, "F2_found_memory", cfg=_BRIDGE, meta=dict(scheds=F2_SCHEDS))


# =============================================================================
# F3.  The price of memory in calibration  (N_eff + realised-rate spread)
# =============================================================================
def run_F3(cfg: CompCfg):
    S, K = plant_S_K()
    k0 = cfg.L
    rows = []
    for regime in cfg.regimes:
        for L in cfg.L_grid:
            for G in cfg.budgets:
                rates = []
                neffs = []
                viol = 0
                for s in range(cfg.n_cal_draws):
                    cal = simulate_stream(regime, cfg.N_cal + cfg.L,
                                          seed_of(regime, "cal", s), cfg)
                    dep = simulate_stream(regime, cfg.T_dep,
                                          seed_of(regime, "dep", s), cfg)
                    if L <= 1:
                        eps = thr_conformal(np.where(np.arange(len(cal["z"])) >= k0,
                                            np.abs(cal["z"]), np.nan), G)
                        gd = fire_threshold(
                            np.where(np.arange(len(dep["z"])) >= k0,
                                     np.abs(dep["z"]), np.nan), eps, k0)
                        gc = fire_threshold(
                            np.where(np.arange(len(cal["z"])) >= k0,
                                     np.abs(cal["z"]), np.nan), eps, k0)
                    else:
                        thr = block_thr_for_rate(cal["z"], L, G, k0)
                        gc = fire_block(cal["z"], thr, L, k0)
                        gd = fire_block(dep["z"], thr, L, k0)
                    r = float(gd[k0:].mean())
                    rates.append(r)
                    if r > G:
                        viol += 1
                    neffs.append(eff_sample_size(gc[k0:].astype(float),
                                                 cfg.neff_lags))
                rates = np.array(rates)
                Neff = float(np.mean(neffs))
                rows.append(dict(
                    exp="F3", regime=regime, scheduler=("MEMLESS" if L <= 1
                                                        else "BLOCK"),
                    L=L, budget=G,
                    N_cal=cfg.N_cal, N_eff=Neff, N_eff_over_N=Neff / cfg.N_cal,
                    eff_exceedances=G * Neff,
                    rate_mean=float(rates.mean()), rate_std=float(rates.std()),
                    rate_std_iid=math.sqrt(G * (1 - G) / cfg.N_cal),
                    rate_std_neff=math.sqrt(G * (1 - G) / max(Neff, 1.0)),
                    conformal_violation=viol / cfg.n_cal_draws))
        print(f"  [F3 {regime}] done")
    return save_rows(rows, "F3_price_of_memory", cfg=_BRIDGE, meta=dict(L_grid=list(cfg.L_grid),
                     n_cal_draws=cfg.n_cal_draws))


# =============================================================================
# F4.  The corrected rule vs dither vs memoryless  (the centerpiece)
# =============================================================================
def run_F4(cfg: CompCfg):
    """CORR (hysteresis) vs MEMLESS vs DITHER, all at matched budget.
    Knob is the continue-gate f (1=B3, ->0 = block).  Reports psi, exposure
    functional, and Gsgn power vs (L, f, budget, W)."""
    S, K = plant_S_K()
    k0 = cfg.L
    rows = []
    for regime in cfg.regimes:
        thr, far = gsgn_nulls(regime, cfg, k0)
        for G in cfg.budgets:
            # ---- CORR grid over (L, f); include MEMLESS as f=1 any-L ----------
            for L in cfg.L_grid:
                for f in cfg.f_grid:
                    memless = (f >= 1.0) or (L <= 1)
                    if memless and not (L == cfg.L_grid[0] and f >= 1.0):
                        continue          # MEMLESS logged once (first L, f=1)
                    psis, trDs, rates, mxs, rmat = [], [], [], [], []
                    z_att = {W: [] for W in cfg.sgn_windows}
                    for s in range(cfg.n_mc):
                        cal = simulate_stream(regime, cfg.N_cal + cfg.L,
                                              seed_of(regime, "cal", s), cfg)
                        dep = simulate_stream(regime, cfg.T_dep,
                                              seed_of(regime, "dep", s), cfg)
                        if memless:
                            eps = thr_conformal(np.where(
                                np.arange(len(cal["z"])) >= k0,
                                np.abs(cal["z"]), np.nan), G)
                            g = fire_threshold(np.where(
                                np.arange(len(dep["z"])) >= k0,
                                np.abs(dep["z"]), np.nan), eps, k0)
                        else:
                            eps = calibrate_hysteresis(cal["z"], f, L, G, k0)
                            g = fire_hysteresis(dep["z"], eps, f, L, k0)
                        dm = dmg(dep["z"], dep["es"], g, K, k0)
                        psis.append(dm["psi"]); trDs.append(dm["tr_D"])
                        rates.append(dm["rate"])
                        zc = received(dep["z"], g)[k0:]
                        mx, rv = exposure_rhat(zc, cfg.exposure_lags, S)
                        mxs.append(mx); rmat.append(rv)
                        for W in cfg.sgn_windows:
                            z_att[W].append(zc)
                    prow = {}
                    for tname, fn in GSGN.items():
                        for W in cfg.sgn_windows:
                            v = block_stats(np.concatenate(z_att[W]), W, fn,
                                            GSGN_TWOSIDED[tname])
                            prow[(tname, W)] = float(np.mean(v > thr[(tname, W)]))
                    rmean = np.mean(np.array(rmat), axis=0)
                    for W in cfg.sgn_windows:
                        rows.append(dict(
                            exp="F4", regime=regime,
                            scheduler=("MEMLESS" if memless else "CORR"),
                            L=(1 if memless else L), f_cont=(1.0 if memless else f),
                            budget=G, W=W,
                            psi=float(np.mean(psis)), tr_D=float(np.mean(trDs)),
                            realized_rate=float(np.mean(rates)),
                            max_rhat=float(np.mean(mxs)), r1=float(rmean[0]),
                            power_acf1=prow[("acf1", W)],
                            power_ljungbox=prow[("ljungbox", W)],
                            far_acf1=far[("acf1", W)]))
            print(f"  [F4 {regime} G={G}] done")
    return save_rows(rows, "F4_corrected_rule", cfg=_BRIDGE,
                     meta=dict(L_grid=list(cfg.L_grid), f_grid=list(cfg.f_grid)))


# =============================================================================
# F5.  Achievable region  (post-processing of F1/F2/F4 -- emit a unified scatter)
# =============================================================================
def run_F5(cfg: CompCfg):
    """Concatenate the (psi, max_rhat, power) points from F1/F2/F4 CSVs into a
    single tidy file for the region figure.  Requires F1,F2,F4 already run."""
    import csv
    out = []
    src = {"F1_dither": "DITHER", "F2_found_memory": None,
           "F4_corrected_rule": None}
    for fname in src:
        p = Path(cfg.out_dir) / f"{fname}.csv"
        if not p.exists():
            print(f"  [F5] missing {p}, skip"); continue
        with open(p) as f:
            for r in csv.DictReader(f):
                def g(k):
                    return r.get(k, "")
                out.append(dict(
                    source=fname, regime=g("regime"), scheduler=g("scheduler"),
                    budget=g("budget"), W=g("W"),
                    L=g("L"), beta=g("beta"), sigma_d=g("sigma_d_over_sqrtS"),
                    psi=g("psi"),
                    max_rhat=g("max_rhat"),
                    power_acf1=g("power_acf1")))
    if not out:
        print("  [F5] nothing to assemble"); return None
    return save_rows(out, "F5_region", cfg=_BRIDGE, meta=dict(note="union of F1/F2/F4 points"))


# =============================================================================
# F6.  Real vehicle data -- is the constraint active on CAN residuals?
# =============================================================================
def _load_innovations(path):
    """Load VARX-recovered innovations from .npz or .csv, tolerantly.

    .npz: uses key 'segments' or 'innovations' if present (object array or 2-D);
    otherwise treats every stored array as data -- each 1-D array is one segment,
    each 2-D array is split into segments along its longer axis.
    .csv: columns 'segment','z' (one row per sample).
    Prints what it found so the layout is visible in the job log.
    Returns a list of 1-D float arrays (one per stationary segment).
    """
    p = Path(path)
    if p.suffix.lower() == ".npz":
        d = np.load(p, allow_pickle=True)
        print(f"  [F6] npz keys: {list(d.files)}")
        segs = []
        keys = d.files
        prefer = [k for k in ("segments", "innovations", "z", "innov") if k in keys]
        keys = prefer + [k for k in keys if k not in prefer]
        for k in keys:
            arr = d[k]
            if arr.dtype == object:                      # object array of arrays
                for s in arr:
                    a = np.asarray(s, float).ravel()
                    if a.size:
                        segs.append(a)
            elif arr.ndim == 1:
                if arr.size:
                    segs.append(np.asarray(arr, float))
            elif arr.ndim == 2:
                # split along the SHORTER axis -> each slice is a time series
                if arr.shape[0] <= arr.shape[1]:
                    for r in arr:
                        segs.append(np.asarray(r, float))
                else:
                    for c in arr.T:
                        segs.append(np.asarray(c, float))
            if prefer and k == prefer[0]:
                break                                    # a named key wins alone
        print(f"  [F6] loaded {len(segs)} segment(s); lengths "
              f"{[len(x) for x in segs][:10]}{' ...' if len(segs) > 10 else ''}")
        return segs
    # CSV
    import csv
    segs = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            segs.setdefault(r.get("segment", "0"), []).append(float(r["z"]))
    out = [np.asarray(v, float) for _, v in sorted(segs.items())]
    print(f"  [F6] loaded {len(out)} segment(s) from CSV")
    return out


def run_F6(cfg: CompCfg, innov_file=None):
    if not innov_file:
        print("  [F6] --innov_file not given; skipping (needs VARX-recovered "
              "CSU innovations, e.g. csu_innovations.npz).")
        return None
    segs = _load_innovations(innov_file)
    k0 = cfg.L
    rows = []
    for si, z in enumerate(segs):
        if len(z) < 4 * cfg.L:
            continue
        S = float(np.var(z))
        # exchangeable split: even indices calibrate, odd deploy (also try
        # contiguous halves for the robustness note)
        for split in ("interleave", "contiguous"):
            if split == "interleave":
                zc_cal, zc_dep = z[0::2], z[1::2]
            else:
                h = len(z) // 2
                zc_cal, zc_dep = z[:h], z[h:]
            for G in cfg.budgets:
                # MEMLESS reference
                eps = thr_conformal(
                    np.where(np.arange(len(zc_cal)) >= k0, np.abs(zc_cal), np.nan), G)
                gd = fire_threshold(
                    np.where(np.arange(len(zc_dep)) >= k0, np.abs(zc_dep), np.nan),
                    eps, k0)
                zrec = received(zc_dep, gd)[k0:]
                mx0, rv0 = exposure_rhat(zrec, cfg.exposure_lags, S)
                psi0 = float(np.sum(np.abs(zc_dep[k0:])[gd[k0:]] ** 2) /
                             np.sum(np.abs(zc_dep[k0:]) ** 2)) if gd[k0:].any() else 0.0
                # CORR (hysteresis) at a representative (L, f)
                Lc, fc = min(cfg.L, 10), 0.3
                epsc = calibrate_hysteresis(zc_cal, fc, Lc, G, k0)
                gc = fire_hysteresis(zc_dep, epsc, fc, Lc, k0)
                zrecc = received(zc_dep, gc)[k0:]
                mxc, rvc = exposure_rhat(zrecc, cfg.exposure_lags, S)
                rows.append(dict(
                    exp="F6", segment=si, split=split, budget=G, S=S,
                    scheduler_ref="MEMLESS", psi_memless=psi0, max_rhat_memless=mx0,
                    r1_memless=rv0[0],
                    scheduler_corr="CORR_L10_f0.3", max_rhat_corr=mxc, r1_corr=rvc[0],
                    realized_rate_memless=float(gd[k0:].mean()),
                    realized_rate_corr=float(gc[k0:].mean())))
        print(f"  [F6] segment {si} done")
    return save_rows(rows, "F6_realdata", cfg=_BRIDGE, meta=dict(n_segments=len(segs),
                     innov_file=str(innov_file)))


# =============================================================================
# S9.  CLI + SLURM array
# =============================================================================
EXPERIMENTS = {"F1": run_F1, "F2": run_F2, "F3": run_F3,
               "F4": run_F4, "F5": run_F5, "F6": run_F6}
ARRAY_ORDER = ["F4", "F2", "F3", "F1", "F5"]   # F4 first (longest); F6 separate


def build_parser():
    p = argparse.ArgumentParser(description="Companion experiments F1-F6")
    p.add_argument("--exp", default="F1",
                   help="F1..F6 | all | array:<idx> (order: " +
                        ",".join(ARRAY_ORDER) + ")")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_mc", type=int, default=None)
    p.add_argument("--n_cal_draws", type=int, default=None)
    p.add_argument("--N_cal", type=int, default=None)
    p.add_argument("--T_dep", type=int, default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--innov_file", default=None, help="F6: VARX innovations file")
    return p


def config_from_args(a) -> CompCfg:
    c = CompCfg()
    if a.quick:
        c = quick(c)
    for k in ("seed", "n_mc", "n_cal_draws", "N_cal", "T_dep", "out_dir"):
        v = getattr(a, k, None)
        if v is not None:
            setattr(c, k, v)
    c.dirs()
    return c


def _bridge_cfg(c: CompCfg):
    """An E.Config carrying the fields simulate_stream/seed_of read."""
    b = E.Config()
    b.T_burn, b.N_cal, b.T_dep, b.T_train = c.T_burn, c.N_cal, c.T_dep, c.T_train
    b.L, b.ar_order, b.seed = c.L, c.ar_order, c.seed
    b.skip_ae = True
    b.out_dir = c.out_dir
    return b


def main(argv=None):
    a = build_parser().parse_args(argv)
    c = config_from_args(a)
    global _BRIDGE
    _BRIDGE = _bridge_cfg(c)

    if a.exp.startswith("array:"):
        todo = [ARRAY_ORDER[int(a.exp.split(":")[1])]]
    elif a.exp == "all":
        todo = ARRAY_ORDER            # F6 run explicitly (needs data)
    else:
        todo = [a.exp]

    print("=" * 72)
    print(f"cps_companion | exp={todo} | regimes={c.regimes} | N_cal={c.N_cal} "
          f"T_dep={c.T_dep} n_mc={c.n_mc}")
    print("=" * 72)
    for name in todo:
        t0 = time.time()
        print(f"\n### {name} ###")
        if name == "F6":
            EXPERIMENTS[name](c, innov_file=a.innov_file)
        else:
            EXPERIMENTS[name](c)
        print(f"### {name} done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
