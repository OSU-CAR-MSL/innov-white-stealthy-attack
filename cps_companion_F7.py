#!/usr/bin/env python3
# =============================================================================
#  F7 -- Conformal calibration of the hysteresis (CORR) scheduler
#
#  PURPOSE
#  -------
#  The deployed rule (18) is calibrated in cps_companion.py by BISECTION on the
#  realised rate -- i.e. the naive empirical quantile of the *rule*, not the
#  split-conformal order statistic of Eq. (13).  Sec. IV nevertheless claims
#  "Lemma 4 transfers verbatim".  This experiment (a) implements the conformal
#  calibration properly, (b) isolates *which* event the guarantee actually
#  covers, and (c) measures how the realised-rate guarantee degrades.
#
#  THE STRUCTURAL POINT THIS TESTS
#  -------------------------------
#  For the hysteresis rule
#        START    : |z_k| > eps_hi
#        CONTINUE : 0 < n_{k-1} < L  and  |z_k| > eps_lo = f * eps_hi
#  the START event is a threshold on the exchangeable scalar score |z_k|, so
#  Lemma 4 applies to it verbatim:      P(start) <= alpha.
#  The FIRING indicator is NOT such a threshold -- it depends on the run state
#  n_{k-1} -- so the guarantee does not transfer to the realised rate.  Two
#  bounds survive:
#        (deterministic)  rate <= L * start_rate          [each start licenses
#                                                          at most L-1 continues]
#        (renewal)        rate ~= start_rate * E[B]       [E[B] = mean run length,
#                                                          law-dependent]
#
#  THREE CALIBRATION VARIANTS COMPARED
#  -----------------------------------
#    BISECT      current code: bisect eps_hi so realised rate = Gbar on the
#                calibration record.  No finite-sample guarantee.
#    CONF_NAIVE  what Sec. IV says if read literally: conformal index at level
#                Gbar applied to |z|.  Over-fires by ~E[B].
#    CONF_START  the repair: allocate alpha = Gbar / Bhat (Bhat = mean run
#                length on the calibration record, fixed-point), then
#                eps_hi = conformal(|z|, alpha).  Start rate carries the exact
#                finite-sample guarantee at level alpha; the realised rate
#                targets Gbar only through the plug-in Bhat.
#
#  OUTPUT  ->  F7_conformal_calibration.csv
#
#  Self-contained: does not import cps_companion.py.  Plant, regimes and
#  calibration protocol are identical to Sec. V of the companion.
# =============================================================================
from __future__ import annotations
import math, json, argparse
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                    # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(fn):
            return fn
        return deco if not a else a[0]

# ----------------------------------------------------------------- plant ----
A = np.array([[0.95, 0.02], [0.0, 0.90]])
C = np.array([[1.0, 0.0]])
Q = 0.01 * np.eye(2)
R = 0.05


def dare(A, C, Q, R, iters=20000, tol=1e-14):
    P = np.eye(A.shape[0])
    for _ in range(iters):
        Sm = C @ P @ C.T + R
        K = P @ C.T @ np.linalg.inv(Sm)
        Pn = A @ (P - K @ C @ P) @ A.T + Q
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn
            break
        P = Pn
    Sm = C @ P @ C.T + R
    S = float(Sm.reshape(-1)[0])
    K = (P @ C.T / S).reshape(-1, 1)
    return P, S, K


PBAR, S_INNOV, KGAIN = dare(A, C, Q, R)
AF = A @ (np.eye(2) - KGAIN @ C)


def lyap_trace(A, W, iters=6000):
    X = np.zeros_like(W)
    for _ in range(iters):
        X = A @ X @ A.T + W
    return float(np.trace(X))


TRLYAP = lyap_trace(A, KGAIN @ KGAIN.T)


# ------------------------------------------------------------- simulator ----
def simulate_stream(regime: str, T: int, seed: int, burn: int = 4000):
    """Steady-state Kalman innovation stream.  N1 Gaussian, N2 variance-matched
    heavy-tailed sensor-noise mixture (identical to Sec. V)."""
    rng = np.random.default_rng(seed)
    n = T + burn
    w = rng.standard_normal((n, 2)) * math.sqrt(0.01)
    if regime == "N1":
        v = rng.standard_normal(n) * math.sqrt(R)
    elif regime == "N2":
        heavy = rng.random(n) < 0.05
        v = np.where(heavy,
                     rng.standard_normal(n) * math.sqrt(0.568),
                     rng.standard_normal(n) * math.sqrt(0.0227))
    else:
        raise ValueError(regime)
    z = _sim_core(w, v, np.ascontiguousarray(AF),
                  np.ascontiguousarray((A @ KGAIN).ravel()))
    return dict(z=z[burn:], es=None)


@njit(cache=True)
def _sim_core(w, v, AFm, AK):
    n = v.shape[0]
    z = np.empty(n)
    e0 = 0.0; e1 = 0.0
    for k in range(n):
        z[k] = e0 + v[k]
        n0 = AFm[0, 0] * e0 + AFm[0, 1] * e1 + w[k, 0] - AK[0] * v[k]
        n1 = AFm[1, 0] * e0 + AFm[1, 1] * e1 + w[k, 1] - AK[1] * v[k]
        e0 = n0; e1 = n1
    return z


# ------------------------------------------------------------ schedulers ----
@njit(cache=True)
def _fire_hyst(a, eps_hi, f, L, k0):
    eps_lo = f * eps_hi
    n = a.shape[0]
    g = np.zeros(n, dtype=np.uint8)
    st = np.zeros(n, dtype=np.uint8)
    run = 0
    for k in range(k0, n):
        if a[k] > eps_hi:
            g[k] = 1; st[k] = 1; run = 1
        elif 0 < run < L and a[k] > eps_lo:
            g[k] = 1; run += 1
        else:
            run = 0
    return g, st


def fire_hysteresis(z, eps_hi, f, L, k0):
    """Eq. (18).  Returns (gamma, start_mask) as boolean arrays."""
    g, st = _fire_hyst(np.abs(np.asarray(z, dtype=np.float64)),
                       float(eps_hi), float(f), int(L), int(k0))
    return g.astype(bool), st.astype(bool)


def mean_run_length(g, k0):
    gg = g[k0:]
    if not gg.any():
        return 1.0
    d = np.diff(np.concatenate(([0], gg.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return float(np.mean(ends - starts))


# ---------------------------------------------------------- calibration -----
def thr_conformal(scores, alpha):
    """Split-conformal order statistic, Eq. (13).
       eps = s_(j_N),  j_N = ceil((N+1)(1-alpha))."""
    s = np.sort(np.asarray(scores)[np.isfinite(scores)])
    N = len(s)
    j = math.ceil((N + 1) * (1.0 - alpha))
    j = min(max(j, 1), N)
    return float(s[j - 1])


def calib_bisect(z_cal, f, L, Gbar, k0, iters=45):
    """CURRENT CODE: bisection on the realised rate.  No guarantee."""
    lo, hi = 0.0, float(np.abs(z_cal).max())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = fire_hysteresis(z_cal, mid, f, L, k0)[0][k0:].mean()
        lo, hi = (mid, hi) if r > Gbar else (lo, mid)
    return hi, Gbar


def calib_conf_naive(z_cal, f, L, Gbar, k0):
    """Conformal index at level Gbar applied directly to |z| (Sec. IV read
       literally).  Over-fires by ~E[B]."""
    a = np.abs(z_cal).astype(float)
    a[:k0] = np.nan
    return thr_conformal(a, Gbar), Gbar


def calib_conf_start(z_cal, f, L, Gbar, k0, iters=40):
    """THE REPAIR.

    eps_hi is ALWAYS a split-conformal order statistic of |z| -- Eq. (13) -- so
    the START event {|z_k| > eps_hi} carries the exact finite-sample guarantee
    P(start) <= alpha of Lemma 4.  What is chosen by the data is the LEVEL
    alpha at which that index is taken: alpha is bisected so that the realised
    firing rate of the hysteresis rule meets the budget on the calibration
    record.

    Bisection on alpha (rather than a fixed point on alpha = Gbar / Bhat) is
    used because the realised rate is monotone in alpha, whereas the fixed-point
    map is not: once runs merge, the observed mean block length can exceed L and
    the iteration oscillates.

    What this does and does not buy:
      * START rate      -- exact distribution-free guarantee at level alpha.
      * rate <= L*alpha -- deterministic, holds pathwise.
      * realised rate   -- targets Gbar only through the data-chosen alpha; no
                           finite-sample guarantee, and it degrades as the
                           effective exceedance count alpha*N_eff falls.
    """
    a = np.abs(z_cal).astype(float)
    a[:k0] = np.nan
    N = int(np.isfinite(a).sum())
    lo, hi = 1.0 / (N + 1), min(1.0, max(Gbar, 1.0 / (N + 1)))
    # widen upper end if even alpha = Gbar under-fires (f close to 1)
    eps_hi_at = lambda al: thr_conformal(a, al)
    rate_at = lambda al: fire_hysteresis(z_cal, eps_hi_at(al), f, L,
                                         k0)[0][k0:].mean()
    if rate_at(hi) < Gbar:
        return eps_hi_at(hi), hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if rate_at(mid) > Gbar:
            hi = mid
        else:
            lo = mid
    alpha = lo
    return eps_hi_at(alpha), alpha


CALIBRATORS = {"BISECT": calib_bisect,
               "CONF_NAIVE": calib_conf_naive,
               "CONF_START": calib_conf_start}


# --------------------------------------------------------------- damage -----
@njit(cache=True)
def _dmg(z, g, k0, Amat, Kf):
    d0 = 0.0; d1 = 0.0; acc = 0.0
    n = z.shape[0]
    for k in range(k0, n):
        n0 = Amat[0, 0] * d0 + Amat[0, 1] * d1
        n1 = Amat[1, 0] * d0 + Amat[1, 1] * d1
        if g[k] == 1:
            kick = 2.0 * z[k]
            n0 += kick * Kf[0]
            n1 += kick * Kf[1]
        d0 = n0; d1 = n1
        acc += d0 * d0 + d1 * d1
    return acc / (n - k0)


def damage(z, es, g, k0):
    """tr(D_inf) (time average) and energy capture Psi for the sign flip."""
    zz = np.asarray(z, dtype=np.float64)
    gg = np.ascontiguousarray(g.astype(np.uint8))
    trD = _dmg(zz, gg, int(k0), np.ascontiguousarray(A),
               np.ascontiguousarray(KGAIN.ravel()))
    zt = zz[k0:]; gt = g[k0:]
    psi = float(np.sum(zt[gt] ** 2) / np.sum(zt ** 2)) if gt.any() else 0.0
    return trD, psi


def eff_sample_size(x, max_lag=200):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    v = float(x @ x / len(x))
    if v <= 0:
        return float(len(x))
    tot = 1.0
    for l in range(1, min(max_lag, len(x) - 1) + 1):
        r = float(x[:-l] @ x[l:] / (len(x) - l)) / v
        if r <= 0:
            break
        tot += 2.0 * r
    return float(len(x) / max(tot, 1e-9))


# ------------------------------------------------------------------ main ----
@dataclass
class Cfg:
    seed: int = 0
    N_cal: int = 5000
    T_dep: int = 50000
    n_cal_draws: int = 200
    k0: int = 50
    budgets: tuple = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
    regimes: tuple = (
        "N1", "N2")
    L_grid: tuple = (1, 2, 4, 6, 10, 20, 50)
    f_grid: tuple = (0.3, 0.15, 0.0)
    out: str = "F7_conformal_calibration.csv"


def quick(c: Cfg) -> Cfg:
    c.N_cal, c.T_dep, c.n_cal_draws = 2000, 8000, 40
    c.budgets = (0.02, 0.10, 0.30)
    c.L_grid = (1, 4, 10, 50)
    c.f_grid = (0.3,)
    c.k0 = 50
    return c


def run_F7(cfg: Cfg):
    rows = []
    cache = {}

    def streams(regime, s):
        key = (regime, s)
        if key not in cache:
            off = 0 if regime == "N1" else 104729
            cache[key] = (
                simulate_stream(regime, cfg.N_cal + cfg.k0, 1000 + 7919 * s + off),
                simulate_stream(regime, cfg.T_dep, 500000 + 7919 * s + off))
        return cache[key]

    for regime in cfg.regimes:
        cache.clear()
        for L in cfg.L_grid:
            for f in cfg.f_grid:
                if L <= 1 and f != cfg.f_grid[0]:
                    continue
                for G in cfg.budgets:
                    for cname, cfun in CALIBRATORS.items():
                        rates, starts, alphas, Bs, neffs = [], [], [], [], []
                        psis, trDs = [], []
                        viol_rate = viol_start = viol_Lbound = 0
                        for s in range(cfg.n_cal_draws):
                            cal, dep = streams(regime, s)
                            eps, alpha = cfun(cal["z"], f, L, G, cfg.k0)
                            gd, sd = fire_hysteresis(dep["z"], eps, f, L, cfg.k0)
                            gc, _ = fire_hysteresis(cal["z"], eps, f, L, cfg.k0)
                            r = float(gd[cfg.k0:].mean())
                            sr = float(sd[cfg.k0:].mean())
                            rates.append(r); starts.append(sr); alphas.append(alpha)
                            Bs.append(mean_run_length(gd, cfg.k0))
                            neffs.append(eff_sample_size(
                                gc[cfg.k0:].astype(float)))
                            if s < max(8, cfg.n_cal_draws // 10):
                                td, ps = damage(dep["z"], dep["es"], gd, cfg.k0)
                                trDs.append(td); psis.append(ps)
                            if r > G:
                                viol_rate += 1
                            if sr > alpha:
                                viol_start += 1
                            if r > L * sr + 1e-12:
                                viol_Lbound += 1
                        rates = np.array(rates); starts = np.array(starts)
                        Neff = float(np.mean(neffs))
                        rows.append(dict(
                            exp="F7", regime=regime, calibrator=cname,
                            L=L, f_cont=f, budget=G, N_cal=cfg.N_cal,
                            n_cal_draws=cfg.n_cal_draws,
                            alpha_mean=float(np.mean(alphas)),
                            # analytic conformal prediction 1 - j_N/(N+1)
                            conf_pred_start_rate=float(np.mean([
                                1.0 - min(max(math.ceil((cfg.N_cal + 1) *
                                              (1.0 - aa)), 1), cfg.N_cal)
                                / (cfg.N_cal + 1) for aa in alphas])),
                            # --- the budget constraint (5) ---
                            rate_mean=float(rates.mean()),
                            rate_std=float(rates.std()),
                            rate_bias=float(rates.mean() - G),
                            rate_rel_bias=float((rates.mean() - G) / G),
                            viol_rate_frac=viol_rate / cfg.n_cal_draws,
                            # --- the START event: does Lemma 4 hold here? ---
                            start_rate_mean=float(starts.mean()),
                            start_rate_std=float(starts.std()),
                            start_bias_vs_alpha=float(starts.mean() -
                                                      np.mean(alphas)),
                            start_bias_vs_confpred=float(starts.mean() - np.mean([
                                1.0 - min(max(math.ceil((cfg.N_cal + 1) *
                                              (1.0 - aa)), 1), cfg.N_cal)
                                / (cfg.N_cal + 1) for aa in alphas])),
                            viol_start_frac=viol_start / cfg.n_cal_draws,
                            # --- the deterministic bound rate <= L*start ---
                            viol_Lbound_frac=viol_Lbound / cfg.n_cal_draws,
                            mean_run_len=float(np.mean(Bs)),
                            # --- degradation driver ---
                            N_eff=Neff, N_eff_over_N=Neff / cfg.N_cal,
                            eff_exceedances=G * Neff,
                            eff_exceedances_start=float(np.mean(alphas)) * Neff,
                            rate_std_iid=math.sqrt(G * (1 - G) / cfg.N_cal),
                            rate_std_neff=math.sqrt(G * (1 - G) / max(Neff, 1.0)),
                            psi=float(np.mean(psis)) if psis else np.nan,
                            tr_D=float(np.mean(trDs)) if trDs else np.nan))
                print(f"  [F7 {regime} L={L} f={f}] done")
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out, index=False)
    print(f"\nwrote {cfg.out}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = Cfg()
    if a.quick:
        cfg = quick(cfg); cfg.out = a.out or "F7_conformal_calibration_quick.csv"
    elif a.out:
        cfg.out = a.out
    run_F7(cfg)
