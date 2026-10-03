#!/usr/bin/env python3
# =============================================================================
#  F8 -- The proposed schedule vs. the historical-data attack literature,
#        under the detector class that literature does not constrain.
#
#  THE ARGUMENT BEING TESTED
#  -------------------------
#  Li & Yang (TSMC 2021), Shang & Chen (Automatica 2021) and Ren, Yang & Zhang
#  (Automatica 2023) all impose STRICT STEALTHINESS against a chi^2 detector of
#  window tau.  Their stealth conditions constrain
#      * the marginal law of the corrupted innovation                  (Shang), or
#      * whiteness at lags 1..tau-1 only                        (Li, Ren)
#  and leave the received autocovariance FREE at lag tau and beyond.  By
#  construction their transmitted streams are therefore MA or AR processes:
#      Li     z~_k = T z_{k-tau} + L z_k + b_k        -> corr at lag tau
#      Shang  z~_k = sum_{i<=tau} T_i z_{k-i} + b_k   -> corr at ALL lags <= tau
#      Ren    z~_k = H z~_{k-tau} + T z_k             -> AR(1) at lag tau
#  The sign flip of this paper carries ZERO received autocovariance whenever the
#  innovation is conditionally sign symmetric -- i.e. exactly under Gaussian
#  noise (Prop. 1) -- for EVERY schedule and every budget.
#
#  So under Gaussian noise the two families separate on a detector neither
#  certificate covers: the innovation-whiteness monitor.
#
#  TWO HONEST QUALIFICATIONS THIS SCRIPT IS BUILT TO EXPOSE, NOT HIDE
#  -----------------------------------------------------------------
#  (1) The baselines have NO BUDGET: they corrupt every transmission (rate 1.0).
#      "At any budget" is therefore a category error.  We sweep OUR budget and
#      report damage alongside exposure so the comparison is a Pareto statement,
#      not a damage claim -- the baselines WILL dominate on damage.
#  (2) Detection depends on the monitor's lag count m.  Ren's and Li's
#      correlation sits at lag tau ALONE, so a Ljung-Box monitor with m < tau
#      misses them.  We sweep m in {1,2,5,10,20,30} against tau to exhibit the
#      threshold.  This is the mirror image of Ren et al. Remark 7 ("the attacker
#      needs to know the window length of the chi^2-detector"): the DEFENDER
#      needs m >= tau, and unlike the attacker the defender chooses m freely.
#
#  ATTACK PARAMETERS
#  -----------------
#  Rather than re-deriving each paper's optimality proof (and risking an algebra
#  slip), each baseline's parameters are obtained by MAXIMIZING EMPIRICAL DAMAGE
#  SUBJECT TO THAT PAPER'S OWN STEALTH CONSTRAINT on a training stream.  This is
#  faithful to the published model class and stealth definition, and gives every
#  baseline its best available attack on our plant.
#
#  OUTPUT ->  F8_baseline_whiteness.csv
# =============================================================================
from __future__ import annotations
import math, argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2dist

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                      # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(fn):
            return fn
        return deco if not a else a[0]

from cps_companion_F7 import (A, C, Q, R, S_INNOV, KGAIN, AF, TRLYAP,
                              simulate_stream, fire_hysteresis,
                              thr_conformal, mean_run_length)


# =============================================================================
#  S1.  Damage -- one recursion for every attack family
#       d_k = A d_{k-1} + K (z_k - z~_k)
# =============================================================================
@njit(cache=True)
def _trD(err, k0, Amat, Kf):
    d0 = 0.0; d1 = 0.0; acc = 0.0
    n = err.shape[0]
    for k in range(k0, n):
        n0 = Amat[0, 0] * d0 + Amat[0, 1] * d1 + Kf[0] * err[k]
        n1 = Amat[1, 0] * d0 + Amat[1, 1] * d1 + Kf[1] * err[k]
        d0 = n0; d1 = n1
        acc += d0 * d0 + d1 * d1
    return acc / (n - k0)


def tr_D(z, z_tilde, k0):
    return _trD(np.ascontiguousarray(z - z_tilde, dtype=np.float64), int(k0),
                np.ascontiguousarray(A), np.ascontiguousarray(KGAIN.ravel()))


# =============================================================================
#  S2.  The three baseline attack families (their own model + stealth constraint)
# =============================================================================
def atk_li_yang(z, T, Lc, sig_b, tau, rng):
    """Li & Yang 2021 Eq. (10):  z~_k = T z_{k-tau} + L z_k + b_k
       stealth (Rem. 5): T^2 S + L^2 S + Var(b) = S."""
    zt = Lc * z.copy()
    zt[tau:] += T * z[:-tau]
    if sig_b > 0:
        zt += sig_b * rng.standard_normal(len(z))
    return zt


def atk_shang_chen(z, Tvec, sig_b, rng):
    """Shang & Chen 2021 Eq. (12):  z~_k = sum_{i=0}^{tau} T^[i] z_{k-i} + b_k
       stealth (Eq. 14): sum_i (T^[i])^2 S + Var(b) = S."""
    zt = np.zeros_like(z)
    for i, Ti in enumerate(Tvec):
        if i == 0:
            zt += Ti * z
        else:
            zt[i:] += Ti * z[:-i]
    if sig_b > 0:
        zt += sig_b * rng.standard_normal(len(z))
    return zt


@njit(cache=True)
def _ren_core(z, H, T, tau):
    n = z.shape[0]
    zt = np.zeros(n)
    for k in range(n):
        v = T * z[k]
        if k >= tau:
            v += H * zt[k - tau]
        zt[k] = v
    return zt


def atk_ren(z, H, T, tau):
    """Ren, Yang & Zhang 2023 Eq. (8b):  z~_k = H z~_{k-tau} + T z_k
       stealth (Eq. 9b): H^2 S + T^2 S = S."""
    return _ren_core(np.ascontiguousarray(z, dtype=np.float64),
                     float(H), float(T), int(tau))


# ------------------------- damage-optimal parameters under each constraint ---
def fit_li_yang(z_tr, tau, k0):
    """max damage s.t. T^2 + L^2 + sig_b^2/S = 1.  b adds no damage-correlated
       content, so the optimum has sig_b = 0; verified by the 2-D search."""
    rng = np.random.default_rng(0)
    best = (-1, None)
    for th in np.linspace(0, 2 * math.pi, 721):
        T, Lc = math.sin(th), math.cos(th)
        d = tr_D(z_tr, atk_li_yang(z_tr, T, Lc, 0.0, tau, rng), k0)
        if d > best[0]:
            best = (d, (T, Lc, 0.0))
    return best[1], best[0]


def fit_shang_chen(z_tr, tau, k0):
    """max damage s.t. sum_i T_i^2 + sig_b^2/S = 1."""
    rng = np.random.default_rng(0)
    m = tau + 1

    def neg(u):
        v = u / max(np.linalg.norm(u), 1e-12)
        return -tr_D(z_tr, atk_shang_chen(z_tr, v, 0.0, rng), k0)

    best = (-1, None)
    for seed in range(6):
        r = np.random.default_rng(seed)
        u0 = r.standard_normal(m)
        u0[0] = -abs(u0[0]) - 1.0                      # start near T_0 = -1
        res = minimize(neg, u0, method="Nelder-Mead",
                       options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6))
        v = res.x / max(np.linalg.norm(res.x), 1e-12)
        d = -res.fun
        if d > best[0]:
            best = (d, v)
    return best[1], best[0]


def fit_ren(z_tr, tau, k0):
    """max damage s.t. H^2 + T^2 = 1."""
    best = (-1, None)
    for th in np.linspace(0, 2 * math.pi, 721):
        H, T = math.sin(th), math.cos(th)
        if abs(H) >= 0.999:                            # |H|=1 -> non-stationary
            continue
        d = tr_D(z_tr, atk_ren(z_tr, H, T, tau), k0)
        if d > best[0]:
            best = (d, (H, T))
    return best[1], best[0]


# =============================================================================
#  S3.  Detectors --  G_mag (their chi^2)  vs  G_sgn (whiteness)
# =============================================================================
def stat_chi2(w, S):
    """Windowed chi^2 -- the detector all three baselines are designed against."""
    return float(np.sum(w * w) / S)


def stat_acf(w, lag):
    w = w - w.mean()
    den = float(w @ w)
    if den <= 0:
        return 0.0
    return float(w[:-lag] @ w[lag:] / den)


def stat_ljungbox(w, m):
    n = len(w)
    w = w - w.mean()
    den = float(w @ w)
    if den <= 0 or n <= m + 1:
        return 0.0
    q = 0.0
    for l in range(1, m + 1):
        r = float(w[:-l] @ w[l:] / den)
        q += r * r / (n - l)
    return float(n * (n + 2) * q)


def empirical_null(z_nom, W, statfn, two_sided, n_win, far):
    # requires len(z_nom) >= n_win * W; the caller sizes T_nom accordingly
    """Threshold calibrated on NOMINAL data to false-alarm rate `far`.
       Asymptotic nulls are not used: under N2 the innovation is white but not
       independent, so var(r_hat) != 1/n and the chi^2 null is mis-sized."""
    vals = []
    for i in range(n_win):
        a = i * W
        if a + W > len(z_nom):
            break
        s = statfn(z_nom[a:a + W])
        vals.append(abs(s) if two_sided else s)
    vals = np.array(vals)
    return float(np.quantile(vals, 1.0 - far)), vals


def power_of(streams, W, statfn, two_sided, thr):
    """Power averaged over non-overlapping windows WITHIN each stream, so no
       window ever straddles two independent realisations."""
    hit = tot = 0
    for z in streams:
        for i in range(len(z) // W):
            s = statfn(z[i * W:(i + 1) * W])
            if (abs(s) if two_sided else s) > thr:
                hit += 1
            tot += 1
    return hit / tot if tot else np.nan


# =============================================================================
#  S4.  Main
# =============================================================================
@dataclass
class Cfg:
    seed: int = 0
    tau: int = 10                  # chi^2 window the baselines are designed against
    T_train: int = 60000
    T_dep: int = 200000
    T_nom: int = 200000
    n_mc: int = 20
    k0: int = 60
    N_cal: int = 5000
    budgets: tuple = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
    regimes: tuple = ("N1", "N2")
    corr_L: int = 4
    corr_f: float = 0.3
    W: int = 5000                  # detector window
    lb_lags: tuple = (1, 2, 5, 10, 20, 30)
    far: float = 0.01
    n_null_windows: int = 2000
    sweep_grid: tuple = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.98)
    out: str = "F8_baseline_whiteness.csv"


def quick(c: Cfg) -> Cfg:
    c.T_train, c.T_dep, c.T_nom = 20000, 60000, 60000
    c.n_mc = 6
    c.budgets = (0.02, 0.10, 0.30)
    c.lb_lags = (1, 5, 10, 20)
    c.n_null_windows = 60
    c.sweep_grid = (0.0, 0.2, 0.5, 0.9)
    return c


def run_F8(cfg: Cfg):
    rows = []
    for regime in cfg.regimes:
        tau, k0, W = cfg.tau, cfg.k0, cfg.W

        # ---- fit each baseline's damage-optimal parameters on a training stream
        z_tr = simulate_stream(regime, cfg.T_train, 31337)["z"]
        (li_T, li_L, li_b), li_d = fit_li_yang(z_tr, tau, k0)
        sh_T, sh_d = fit_shang_chen(z_tr, tau, k0)
        (rn_H, rn_T), rn_d = fit_ren(z_tr, tau, k0)
        print(f"  [{regime}] Li (T,L)=({li_T:+.3f},{li_L:+.3f})  "
              f"Ren (H,T)=({rn_H:+.3f},{rn_T:+.3f})  "
              f"Shang T0={sh_T[0]:+.3f} |T|={np.linalg.norm(sh_T):.3f}")

        # ---- empirical nulls on nominal data -------------------------------
        need = cfg.n_null_windows * W + 5000
        z_nom = simulate_stream(regime, max(cfg.T_nom, need), 777)["z"]
        nulls = {}
        nulls[("chi2", tau)] = empirical_null(
            z_nom, W, lambda w: stat_chi2(w, S_INNOV), False,
            cfg.n_null_windows, cfg.far)[0]
        for l in (1, tau):
            nulls[("acf", l)] = empirical_null(
                z_nom, W, lambda w, l=l: stat_acf(w, l), True,
                cfg.n_null_windows, cfg.far)[0]
        for m in cfg.lb_lags:
            nulls[("lb", m)] = empirical_null(
                z_nom, W, lambda w, m=m: stat_ljungbox(w, m), False,
                cfg.n_null_windows, cfg.far)[0]

        # ---- attacks --------------------------------------------------------
        def record(name, family, budget, rate, zt_list, z_list):
            st = [x[k0:] for x in zt_list]
            za = np.concatenate(st)
            trd = float(np.mean([tr_D(z, zt, k0)
                                 for z, zt in zip(z_list, zt_list)]))
            row = dict(exp="F8", regime=regime, attack=name, family=family,
                       tau=tau, W=W, budget=budget, fire_rate=rate,
                       tr_D=trd,
                       tr_D_norm=trd / (4.0 * S_INNOV * TRLYAP),
                       acf1=abs(stat_acf(za, 1)),
                       acf_tau=abs(stat_acf(za, tau)),
                       power_chi2=power_of(st, W,
                                           lambda w: stat_chi2(w, S_INNOV),
                                           False, nulls[("chi2", tau)]),
                       power_acf1=power_of(st, W, lambda w: stat_acf(w, 1),
                                           True, nulls[("acf", 1)]),
                       power_acf_tau=power_of(st, W,
                                              lambda w: stat_acf(w, tau),
                                              True, nulls[("acf", tau)]),
                       far_target=cfg.far)
            for m in cfg.lb_lags:
                row[f"power_lb_m{m}"] = power_of(
                    st, W, lambda w, m=m: stat_ljungbox(w, m), False,
                    nulls[("lb", m)])
            rows.append(row)

        deps = [simulate_stream(regime, cfg.T_dep, 900000 + 7919 * s)["z"]
                for s in range(cfg.n_mc)]
        rngs = [np.random.default_rng(4242 + s) for s in range(cfg.n_mc)]

        # nominal control (no attack): every power must equal ~far
        record("NOMINAL", "control", np.nan, 0.0, deps, deps)

        record("LiYang2021", "baseline", np.nan, 1.0,
               [atk_li_yang(z, li_T, li_L, li_b, tau, r)
                for z, r in zip(deps, rngs)], deps)
        record("ShangChen2021", "baseline", np.nan, 1.0,
               [atk_shang_chen(z, sh_T, 0.0, r) for z, r in zip(deps, rngs)],
               deps)
        record("RenEtAl2023", "baseline", np.nan, 1.0,
               [atk_ren(z, rn_H, rn_T, tau) for z, r in zip(deps, rngs)], deps)

        # ---- ours: sign flip on a budgeted hysteresis schedule ---------------
        for G in cfg.budgets:
            zts, rates = [], []
            for s, z in enumerate(deps):
                cal = simulate_stream(regime, cfg.N_cal + k0,
                                      1000 + 7919 * s)["z"]
                lo, hi = 0.0, float(np.abs(cal).max())
                for _ in range(45):
                    mid = 0.5 * (lo + hi)
                    r = fire_hysteresis(cal, mid, cfg.corr_f, cfg.corr_L,
                                        k0)[0][k0:].mean()
                    lo, hi = (mid, hi) if r > G else (lo, mid)
                g, _ = fire_hysteresis(z, hi, cfg.corr_f, cfg.corr_L, k0)
                zts.append(np.where(g, -z, z))
                rates.append(float(g[k0:].mean()))
            record(f"OURS_CORR_G{G}", "proposed", G, float(np.mean(rates)),
                   zts, deps)
            # memoryless reference at the same budget
            zts2, rates2 = [], []
            for s, z in enumerate(deps):
                cal = simulate_stream(regime, cfg.N_cal + k0,
                                      1000 + 7919 * s)["z"]
                a = np.abs(cal).astype(float); a[:k0] = np.nan
                eps = thr_conformal(a, G)
                g = np.abs(z) > eps; g[:k0] = False
                zts2.append(np.where(g, -z, z))
                rates2.append(float(g[k0:].mean()))
            record(f"OURS_MEMLESS_G{G}", "proposed", G,
                   float(np.mean(rates2)), zts2, deps)
        # ---- parameter sweeps: the damage/exposure frontier of EACH family --
        # The only way a baseline reaches nominal whiteness power is to switch
        # its historical term off (H->0 / T_tau->0), i.e. to collapse to the
        # memoryless attack of Guo et al. and forfeit the historical data.
        for hh in cfg.sweep_grid:
            tt = math.sqrt(max(1.0 - hh * hh, 0.0))
            record(f"SWEEP_Ren_H{hh:.2f}", "sweep_ren", np.nan, 1.0,
                   [atk_ren(z, hh, -tt, tau) for z in deps], deps)
        for tt in cfg.sweep_grid:
            ll = -math.sqrt(max(1.0 - tt * tt, 0.0))
            record(f"SWEEP_Li_T{tt:.2f}", "sweep_li", np.nan, 1.0,
                   [atk_li_yang(z, -tt, ll, 0.0, tau, r)
                    for z, r in zip(deps, rngs)], deps)
        print(f"  [F8 {regime}] done")

    df = pd.DataFrame(rows)
    df.to_csv(cfg.out, index=False)
    print(f"\nwrote {cfg.out}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tau", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = Cfg()
    if a.quick:
        cfg = quick(cfg); cfg.out = "F8_baseline_whiteness_quick.csv"
    if a.tau:
        cfg.tau = a.tau
    if a.out:
        cfg.out = a.out
    run_F8(cfg)
