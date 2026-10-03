#!/usr/bin/env python3
# =============================================================================
#  F10 -- Budget convergence:  how the accuracy of the realised firing rate
#         depends on the CALIBRATION RECORD LENGTH N, and why the governing
#         resource is the effective exceedance count alpha * N_eff rather
#         than N itself.
#
#  F7 fixed N = 5000 and swept (L, f, Gbar); F10 fixes the rule and sweeps N.
#  It reports, for the repaired calibrator CONF_START:
#     * bias      E[Ghat] - Gbar
#     * spread    sd(Ghat), against the iid prediction and the N_eff prediction
#     * viol_rate fraction of draws with Ghat > Gbar   (should sit near 1/2:
#                 the estimator is median-unbiased, not conservative)
#     * viol_L    fraction with Ghat > L*alpha         (must be exactly 0)
#     * N_eff, eff_exceedances = alpha * N_eff
#
#  The point of the table: at fixed N the accuracy degrades with L, and the
#  curves for different L collapse when plotted against alpha * N_eff.
#
#  Requires cps_companion_F7.py.   Output -> F10_budget_convergence.csv
# =============================================================================
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from cps_companion_F7 import (simulate_stream, fire_hysteresis, calib_conf_start,
                              mean_run_length, eff_sample_size)


@dataclass
class Cfg:
    T_dep: int = 50000
    n_cal_draws: int = 400
    k0: int = 50
    N_grid: tuple = (250, 500, 1000, 2000, 5000, 10000, 20000, 50000)
    L_grid: tuple = (4, 20, 50)
    f_cont: float = 0.0
    budgets: tuple = (0.02, 0.10)
    regimes: tuple = ("N2", "N1")
    out: str = "F10_budget_convergence.csv"


def quick(c: Cfg) -> Cfg:
    c.T_dep, c.n_cal_draws = 20000, 60
    c.N_grid = (250, 1000, 5000, 20000)
    c.L_grid = (4, 20)
    c.budgets = (0.02,)
    c.regimes = ("N2",)
    c.out = "F10_budget_convergence_quick.csv"
    return c


def run_F10(cfg: Cfg):
    rows = []
    for regime in cfg.regimes:
        # one long deployment stream per draw index, reused across (N, L, Gbar)
        deps = [simulate_stream(regime, cfg.T_dep, 500000 + 7919 * d)["z"]
                for d in range(cfg.n_cal_draws)]
        for N in cfg.N_grid:
            cals = [simulate_stream(regime, N + cfg.k0, 3000000 + 104729 * d)["z"]
                    for d in range(cfg.n_cal_draws)]
            for L in cfg.L_grid:
                for G in cfg.budgets:
                    rate, start, alph, runs, neff = [], [], [], [], []
                    vL_path = vL_alpha = 0
                    for d in range(cfg.n_cal_draws):
                        eps, al = calib_conf_start(cals[d], cfg.f_cont, L, G,
                                                   cfg.k0)
                        g, st = fire_hysteresis(deps[d], eps, cfg.f_cont, L,
                                                cfg.k0)
                        gc, _ = fire_hysteresis(cals[d], eps, cfg.f_cont, L,
                                                cfg.k0)
                        r = float(g[cfg.k0:].mean())
                        sr = float(st[cfg.k0:].mean())
                        rate.append(r); start.append(sr); alph.append(al)
                        runs.append(mean_run_length(g, cfg.k0))
                        # N_eff of the CALIBRATION record (the resource that
                        # sets calibration accuracy), so N_eff <= N by design
                        neff.append(eff_sample_size(gc[cfg.k0:].astype(float)))
                        if r > L * sr + 1e-12:
                            vL_path += 1          # pathwise bound: must be 0
                        if r > L * al + 1e-12:
                            vL_alpha += 1         # in-expectation version only
                    rate = np.array(rate); alph = np.array(alph)
                    ne = float(np.mean(neff))
                    rows.append(dict(
                        exp="F10", regime=regime, calibrator="CONF_START",
                        N_cal=N, L=L, f_cont=cfg.f_cont, budget=G,
                        n_cal_draws=cfg.n_cal_draws,
                        alpha_mean=float(alph.mean()),
                        rate_mean=float(rate.mean()),
                        rate_bias=float(rate.mean() - G),
                        rate_rel_bias=float((rate.mean() - G) / G),
                        rate_std=float(rate.std(ddof=1)),
                        abs_err_mean=float(np.abs(rate - G).mean() / G),
                        viol_rate_frac=float((rate > G).mean()),
                        viol_Lpath_frac=vL_path / cfg.n_cal_draws,
                        viol_Lalpha_frac=vL_alpha / cfg.n_cal_draws,
                        start_rate_mean=float(np.mean(start)),
                        start_bias_vs_alpha=float(np.mean(start) - alph.mean()),
                        mean_run_len=float(np.mean(runs)),
                        N_eff=ne, N_eff_over_N=ne / N,
                        # calibration-side resource: expected number of
                        # calibration magnitudes above eps_hi.  The innovations
                        # are white, so the order statistic sees all N points.
                        exc_cal=float(alph.mean()) * N,
                        # deployment-side resource: independent blocks in the
                        # realised-rate average (N_eff/N ~ 1/L)
                        eff_exceedances=float(alph.mean()) * ne,
                        std_pred_iid=float(np.sqrt(G * (1 - G) / N)),
                        std_pred_neff=float(np.sqrt(G * (1 - G) / max(ne, 1.0))),
                    ))
                    print("  [F10 %s N=%6d L=%2d G=%.2f] alpha=%.4f "
                          "rate=%.4f bias=%+.4f rel=%+.3f sd=%.4f "
                          "viol=%.2f violLpath=%.2f Neff/N=%.3f excCal=%.1f nexcNeff=%.2f"
                          % (regime, N, L, G, rows[-1]["alpha_mean"],
                             rows[-1]["rate_mean"], rows[-1]["rate_bias"],
                             rows[-1]["rate_rel_bias"], rows[-1]["rate_std"],
                             rows[-1]["viol_rate_frac"],
                             rows[-1]["viol_Lpath_frac"],
                             rows[-1]["N_eff_over_N"],
                             rows[-1]["exc_cal"],
                             rows[-1]["eff_exceedances"]))
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out, index=False)
    print("\nwrote", cfg.out, "(%d rows)" % len(df))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = quick(Cfg()) if a.quick else Cfg()
    if a.out:
        cfg.out = a.out
    run_F10(cfg)
