#!/usr/bin/env python3
# =============================================================================
#  F9 -- Windowed scores at matched firing count, reported in (Psi, pi)
#
#  WHY THIS RE-RUN EXISTS
#  ----------------------
#  F2_found_memory.csv records the exposure FUNCTIONAL max_l |r_l| but carries
#  no detector column, so Table "camps" could not be stated in the same metric
#  as the main table.  F9 reruns the same five schedulers at the same matched
#  firing count and additionally evaluates the OPERATIONAL metric
#      pi = power of a lag-one innovation-whiteness monitor of window W
#           against an empirically calibrated 1% null,
#  so that every table in Sec. V speaks in (Psi, pi).
#
#  Schedulers (identical definitions to run_F2 in cps_companion.py):
#    MEMLESS  threshold on |z_k|
#    AR       one-step magnitude predictor residual, AR(p) on |z|
#    MAGDIFF  | |z_k| - |z_{k-1}| |
#    ENERGY   windowed energy, window L
#    BLOCK    fixed hold of length L after a magnitude crossing
#    CORR     the deployed two-threshold rule, for reference
#
#  All non-BLOCK rules fire on the top ceil(Gbar * n) scores of the deployment
#  stream (matched realised rate), exactly as in run_F2.
#
#  Requires cps_companion_F7.py in the same directory (plant + simulator).
#  Output ->  F9_camps.csv
# =============================================================================
from __future__ import annotations
import math, argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd

from cps_companion_F7 import (S_INNOV, simulate_stream, fire_hysteresis,
                              thr_conformal)


# ----------------------------------------------------------------- scores ---
def _nanpad(T):
    return np.full(T, np.nan)


def score_magnitude(z):
    return np.abs(z).astype(float)


def score_diff(z):
    s = _nanpad(len(z)); m = np.abs(z)
    s[1:] = np.abs(m[1:] - m[:-1])
    return s


def score_energy(z, L):
    s = _nanpad(len(z)); e = z ** 2
    cs = np.concatenate([[0.0], np.cumsum(e)])
    s[L - 1:] = cs[L:] - cs[:-L]
    return s


def fit_ar(mag, p):
    X = np.lib.stride_tricks.sliding_window_view(mag, p)[:-1]
    y = mag[p:]
    X1 = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return coef


def score_ar(z, coef, p, L):
    s = _nanpad(len(z)); m = np.abs(z)
    X = np.lib.stride_tricks.sliding_window_view(m, p)[:-1]
    X1 = np.hstack([X, np.ones((len(X), 1))])
    s[p:] = (m[p:] - X1 @ coef) ** 2
    s[:L] = np.nan
    return s


# ------------------------------------------------------------- schedulers ---
def fire_top_rate(s, rate, k0):
    """Matched realised rate: fire on the top ceil(rate*n) scores."""
    g = np.zeros(len(s), dtype=bool)
    idx = np.where(np.isfinite(s))[0]; idx = idx[idx >= k0]
    if rate <= 0 or len(idx) == 0:
        return g
    n = min(int(math.ceil(rate * len(idx))), len(idx))
    g[idx[np.argsort(s[idx])[::-1]][:n]] = True
    return g


def fire_block(z, thr, L, k0):
    a = np.abs(z); g = np.zeros(len(z), dtype=bool); last = -10 ** 9
    for i in np.flatnonzero(a > thr):
        if i < k0:
            continue
        if i >= last + L:
            g[i:i + L] = True; last = i
    g[:k0] = False
    return g


def block_thr_for_rate(z, L, rate, k0, iters=45):
    lo, hi = 0.0, float(np.abs(z).max())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fire_block(z, mid, L, k0).mean() > rate else (lo, mid)
    return hi


# --------------------------------------------------------------- metrics ---
def energy_capture(z, g, k0):
    zz, gg = z[k0:], g[k0:]
    return float(np.sum(zz[gg] ** 2) / np.sum(zz ** 2)) if gg.any() else 0.0


def exposure_max(zc, lags, S):
    out = []
    for l in range(1, lags + 1):
        out.append(float(np.mean(zc[:-l] * zc[l:]) / S))
    return max(abs(v) for v in out), out


def mean_run_length(g, k0):
    gg = g[k0:]
    if not gg.any():
        return 0.0
    d = np.diff(np.concatenate(([0], gg.astype(np.int8), [0])))
    return float(np.mean(np.flatnonzero(d == -1) - np.flatnonzero(d == 1)))


def stat_acf1(w):
    w = w - w.mean(); den = float(w @ w)
    return 0.0 if den <= 0 else float(w[:-1] @ w[1:] / den)


def empirical_null(z_nom, W, far, n_win):
    """Threshold on |acf1| calibrated on NOMINAL data to false-alarm rate far."""
    v = [abs(stat_acf1(z_nom[i * W:(i + 1) * W]))
         for i in range(min(n_win, len(z_nom) // W))]
    return float(np.quantile(np.array(v), 1.0 - far))


def power_of(streams, W, thr):
    hit = tot = 0
    for zc in streams:
        for i in range(len(zc) // W):
            if abs(stat_acf1(zc[i * W:(i + 1) * W])) > thr:
                hit += 1
            tot += 1
    return hit / tot if tot else np.nan


# ------------------------------------------------------------------ main ---
@dataclass
class Cfg:
    N_cal: int = 5000
    T_dep: int = 50000
    T_train: int = 60000
    n_mc: int = 50
    L: int = 50                  # window for ENERGY / BLOCK (as in run_F2)
    ar_order: int = 5
    corr_L: int = 4
    corr_f: float = 0.3
    budgets: tuple = (0.02, 0.10)
    regimes: tuple = ("N2", "N1")
    W: int = 5000
    far: float = 0.01
    n_null_windows: int = 2000
    exposure_lags: int = 10
    out: str = "F9_camps.csv"


def quick(c: Cfg) -> Cfg:
    c.T_dep, c.T_train, c.n_mc = 20000, 20000, 8
    c.n_null_windows = 200
    return c


SCHEDS = ["MEMLESS", "AR", "MAGDIFF", "ENERGY", "BLOCK", "CORR"]


def run_F9(cfg: Cfg):
    rows = []
    for regime in cfg.regimes:
        k0 = cfg.L
        arcoef = fit_ar(np.abs(simulate_stream(regime, cfg.T_train, 4242)["z"]),
                        cfg.ar_order)
        need = cfg.n_null_windows * cfg.W + 5000
        thr_null = empirical_null(simulate_stream(regime, need, 777)["z"],
                                  cfg.W, cfg.far, cfg.n_null_windows)
        deps = [simulate_stream(regime, cfg.T_dep, 900000 + 7919 * s)["z"]
                for s in range(cfg.n_mc)]
        for G in cfg.budgets:
            for name in SCHEDS:
                psis, mxs, brs, rates, zcs, mzs = [], [], [], [], [], []
                for si, z in enumerate(deps):
                    if name == "BLOCK":
                        g = fire_block(z, block_thr_for_rate(z, cfg.L, G, k0),
                                       cfg.L, k0)
                    elif name == "CORR":
                        cal = simulate_stream(regime, cfg.N_cal + k0,
                                              1000 + 7919 * si)["z"]
                        lo, hi = 0.0, float(np.abs(cal).max())
                        for _ in range(45):
                            mid = 0.5 * (lo + hi)
                            r = fire_hysteresis(cal, mid, cfg.corr_f,
                                                cfg.corr_L, k0)[0][k0:].mean()
                            lo, hi = (mid, hi) if r > G else (lo, mid)
                        g, _ = fire_hysteresis(z, hi, cfg.corr_f, cfg.corr_L, k0)
                    else:
                        if name == "MEMLESS":
                            sc = score_magnitude(z)
                        elif name == "AR":
                            sc = score_ar(z, arcoef, cfg.ar_order, cfg.L)
                        elif name == "MAGDIFF":
                            sc = score_diff(z)
                        elif name == "ENERGY":
                            sc = score_energy(z, cfg.L)
                        sc = np.asarray(sc, float).copy(); sc[:k0] = np.nan
                        g = fire_top_rate(sc, G, k0)
                    zc = np.where(g, -z, z)[k0:]
                    psis.append(energy_capture(z, g, k0))
                    mxs.append(exposure_max(zc, cfg.exposure_lags, S_INNOV)[0])
                    brs.append(mean_run_length(g, k0))
                    rates.append(float(g[k0:].mean()))
                    gg = g[k0:]
                    mzs.append(float(np.abs(z[k0:][gg]).mean())
                               if gg.any() else 0.0)
                    zcs.append(zc)
                rows.append(dict(
                    exp="F9", regime=regime, scheduler=name, budget=G,
                    W=cfg.W, psi=float(np.mean(psis)),
                    power_acf1=power_of(zcs, cfg.W, thr_null),
                    max_rhat=float(np.mean(mxs)),
                    mean_burst_len=float(np.mean(brs)),
                    mean_absz_fire=float(np.mean(mzs)),
                    realized_rate=float(np.mean(rates)),
                    far_target=cfg.far, null_thr=thr_null))
                print("  [F9 %s %-8s G=%.2f] psi=%.3f pi=%.3f b=%.1f"
                      % (regime, name, G, rows[-1]["psi"],
                         rows[-1]["power_acf1"], rows[-1]["mean_burst_len"]))
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out, index=False)
    print("\nwrote", cfg.out, "(%d rows)" % len(df))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = Cfg()
    if a.quick:
        cfg = quick(cfg); cfg.out = "F9_camps_quick.csv"
    if a.out:
        cfg.out = a.out
    run_F9(cfg)
