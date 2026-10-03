#!/usr/bin/env python3
# =============================================================================
#  make_results_figs.py -- floats for Sec. V of the companion
#
#  Place in the directory holding the result CSVs and run:
#
#      python3 make_results_figs.py              # every float
#      python3 make_results_figs.py main region  # selected ones
#
#  Figures  ->  figs/results/*.pdf
#  Tables   ->  printed to stdout as LaTeX table BODIES, so the numbers in
#               the manuscript are generated rather than transcribed.
#
#  INPUTS                                     PRODUCES
#    F1_dither.csv          blind + dither    fig_dither, blind column
#    F4_corrected_rule.csv  memoryless + CORR fig_main, fig_region, tab:main
#    F9_camps.csv           windowed scores   fig_main inset pts, tab:camps
#    F10_budget_convergence.csv               fig_budget, tab:budget
#
#  Filenames may carry a hex- prefix (as exported from the project); _find
#  tolerates that.
#
#  COVERT CRITERION.  "Covert" means the lag-one whiteness monitor is held at
#  the level it already shows against the BLIND rule at the same budget.  That
#  reference is used rather than the 1% design false-alarm rate for a reason
#  worth stating: blind sign flipping induces no lag-one correlation in mean
#  (E[sigma_k sigma_{k+1}] E[z_k z_{k+1}] = 0 for white z), but it does change
#  the fourth-order structure of the record, so a threshold calibrated on
#  NOMINAL data does not deliver exactly 1% on a blind-flipped record.  The
#  measured blind power consequently drifts upward with the budget -- 0.010 at
#  Gbar = 0.02 to 0.040 at Gbar = 0.30 (W = 5000) -- and is the empirical floor
#  of the monitor, not 0.01.
#
#  The criterion therefore also has to state its own resolution: with n_win
#  windows a difference of one window is 1/n_win, so "at the nominal level"
#  means within two Monte-Carlo standard errors of the blind level,
#      pi <= pi_blind(Gbar) + 2 sqrt(pi_blind (1 - pi_blind) / n_win).
#  Strict inequality against pi_blind would be decided by single windows.
# =============================================================================
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.0, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3, "font.family": "serif",
})
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", grey="#7F7F7F", sky="#56B4E9", black="#000000")

OUT = os.path.join("figs", "results")
os.makedirs(OUT, exist_ok=True)

W_REF    = 5000        # reporting window
N_WIN    = 500         # power windows per configuration: 50 streams x 50e3 / W
PI_FLOOR = 8e-3        # plotting floor for log axes (zero power -> here)
BUDGETS   = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]


def _find(name):
    if os.path.exists(name):
        return name
    for fn in sorted(os.listdir(".")):
        if fn.endswith(name):
            return fn
    raise FileNotFoundError(name)


def _load(name):
    return pd.read_csv(_find(name))


def _clip(p):
    return np.maximum(np.asarray(p, float), PI_FLOOR)


# --------------------------------------------------------------- data views --
def blind_table(regime="N2"):
    d = _load("F1_dither.csv")
    d = d[(d.blind == True) & (d.W == W_REF) & (d.regime == regime)]
    return d.set_index("budget")[["psi", "power_acf1"]].sort_index()


def dither_table(regime="N2"):
    d = _load("F1_dither.csv")
    return d[(d.blind == False) & (d.W == W_REF) & (d.regime == regime)] \
        .sort_values(["budget", "sigma_d_over_sqrtS"])


def memless_table(regime="N2"):
    d = _load("F4_corrected_rule.csv")
    d = d[(d.scheduler == "MEMLESS") & (d.W == W_REF) & (d.regime == regime)]
    return d.set_index("budget")[["psi", "power_acf1", "max_rhat", "r1"]].sort_index()


def corr_table(regime="N2"):
    d = _load("F4_corrected_rule.csv")
    return d[(d.scheduler == "CORR") & (d.W == W_REF) & (d.regime == regime)]


def pi_nominal(regime="N2"):
    """Empirical nominal exposure level of the monitor, per budget: the power
       it shows against the blind rule, plus two Monte-Carlo standard errors.
       See the COVERT CRITERION note in the header."""
    bl = blind_table(regime)
    pn = bl.power_acf1
    return pn, pn + 2.0 * np.sqrt(pn * (1.0 - pn) / N_WIN)


def covert_pick(regime="N2"):
    """Largest Psi among CORR settings whose exposure is at the nominal level.
       Returns a frame indexed by budget with the chosen (L, f) and metrics."""
    cd, rows = corr_table(regime), []
    pn, tol = pi_nominal(regime)
    for G in BUDGETS:
        g = cd[cd.budget == G]
        ok = g[g.power_acf1 <= tol.loc[G] + 1e-12]
        pick = (ok.loc[ok.psi.idxmax()] if len(ok)
                else g.loc[g.power_acf1.idxmin()])
        rows.append(dict(budget=G, L=int(pick.L), f=float(pick.f_cont),
                         psi=float(pick.psi), pi=float(pick.power_acf1),
                         max_rhat=float(pick.max_rhat),
                         pi_nom=float(pn.loc[G]), pi_tol=float(tol.loc[G]),
                         n_feasible=int(len(ok)), feasible=bool(len(ok))))
    return pd.DataFrame(rows).set_index("budget")


# =========================================================== FIG 1: main ====
def fig_main(regime="N2", budgets=(0.02, 0.10)):
    """Damage-exposure trade-off at a fixed budget.  The corrected rule's
       (L, f) cloud is shown against the two rules that bracket it: blind
       (no location information) and memoryless (all of it, no run
       structure).  The covert band is the region the monitor cannot
       distinguish from Gaussian."""
    bl, ml, cd = blind_table(regime), memless_table(regime), corr_table(regime)
    cmp_ = _load("F9_camps.csv")
    fig, axes = plt.subplots(1, len(budgets), figsize=(6.9, 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, G in zip(axes, budgets):
        g = cd[cd.budget == G]
        tolG = float(pi_nominal(regime)[1].loc[G])
        ax.axvspan(PI_FLOOR, tolG, color=C["green"], alpha=0.09, lw=0)
        ax.axvline(tolG, color=C["green"], lw=0.8, ls=":")
        # corrected cloud, shaded by run length L
        sc = ax.scatter(_clip(g.power_acf1), g.psi, c=np.log2(g.L), s=17,
                        cmap="viridis", alpha=0.85, edgecolor="none", zorder=3)
        # the deployed frontier: best Psi at or below each exposure level
        o = g.sort_values("power_acf1")
        ax.plot(_clip(o.power_acf1), np.maximum.accumulate(o.psi),
                color=C["grey"], lw=1.0, ls="-", zorder=2)
        # bracketing rules
        ax.scatter([_clip([bl.loc[G, "power_acf1"]])[0]], [bl.loc[G, "psi"]],
                   marker="s", s=42, color=C["black"], zorder=5)
        ax.scatter([_clip([ml.loc[G, "power_acf1"]])[0]], [ml.loc[G, "psi"]],
                   marker="*", s=90, color=C["red"], zorder=5)
        # long-window windowed scores, if F9 covers this budget
        c9 = cmp_[(cmp_.regime == regime) & (cmp_.budget == G)
                  & (cmp_.scheduler.isin(["ENERGY", "BLOCK"]))]
        if len(c9):
            ax.scatter(_clip(c9.power_acf1), c9.psi, marker="v", s=30,
                       facecolor="none", edgecolor=C["purple"], zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel(r"whiteness power $\pi$")
        ax.set_title(r"$\bar\Gamma=%.2f$" % G, pad=3)
        ax.grid(alpha=0.18, lw=0.5)
    axes[0].set_ylabel(r"energy capture $\Psi$")
    Lv = sorted(cd.L.unique())
    cb = fig.colorbar(sc, ax=axes.tolist(), pad=0.015, fraction=0.028,
                      ticks=np.log2(Lv))
    cb.ax.set_yticklabels([str(int(v)) for v in Lv])
    cb.set_label(r"run length $L$", fontsize=8)
    fig.legend(handles=[
        Line2D([], [], marker="*", ls="", color=C["red"], ms=9,
               label="memoryless [ref.]"),
        Line2D([], [], marker="s", ls="", color=C["black"], ms=5,
               label="blind"),
        Line2D([], [], marker="o", ls="", color="#3B528B", ms=5,
               label=r"corrected, all $(L,f)$ (shaded by $L$)"),
        Line2D([], [], marker="v", ls="", mfc="none", mec=C["purple"], ms=5,
               label="long-window scores"),
        Line2D([], [], color=C["green"], ls=":", label="nominal exposure"),
    ], loc="lower center", bbox_to_anchor=(0.45, -0.16), ncol=5,
        frameon=False, handletextpad=0.4, columnspacing=1.1)
    p = os.path.join(OUT, "fig_main.pdf"); fig.savefig(p); plt.close(fig)
    print("wrote", p)


# ========================================================= FIG 2: region ====
def fig_region(regime="N2"):
    """Achievable set across budgets, Psi on x and pi on y.

       This orientation is used deliberately.  With pi on x the blind and
       corrected rules both sit against the left spine and the difference in
       Psi -- the quantity of interest -- is compressed into the axis.  With
       Psi on x the blind rule becomes the lower locus and the corrected rule
       is visibly displaced to the RIGHT along it: same exposure, more damage.
       That displacement is the statement the figure has to make, and the
       annotated ratios measure it."""
    bl, ml, cd = blind_table(regime), memless_table(regime), corr_table(regime)
    cv = covert_pick(regime)
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    ax.scatter(cd.psi, _clip(cd.power_acf1), s=7, color=C["sky"], alpha=0.30,
               edgecolor="none", zorder=2, label=r"corrected, all $(L,f)$")
    ax.plot(ml.psi, _clip(ml.power_acf1), "-*", color=C["red"], ms=9,
            zorder=5, label="memoryless (damage ceiling)")
    ax.plot(bl.psi, _clip(bl.power_acf1), "-s", color=C["black"], ms=4,
            zorder=5, label="blind (nominal exposure)")
    ax.plot(cv.psi, _clip(cv.pi), "-o", color=C["blue"], ms=5, zorder=6,
            label="corrected, tuned covert")
    # how much damage the correction recovers at the SAME exposure level
    for G, dx, dy in ((0.02, 4, -10), (0.05, 2, 7), (0.10, 3, -10),
                      (0.20, 3, 7), (0.30, 3, -10), (0.50, 3, 6)):
        ax.annotate(r"$%.1f\times$" % (cv.loc[G, "psi"] / bl.loc[G, "psi"]),
                    xy=(cv.loc[G, "psi"], _clip([cv.loc[G, "pi"]])[0]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=6.3,
                    color=C["blue"])
    for G, ha, dx in ((0.02, "right", -4), (0.50, "left", 5)):
        ax.annotate(r"$\bar\Gamma{=}%.2f$" % G,
                    xy=(bl.loc[G, "psi"], _clip([bl.loc[G, "power_acf1"]])[0]),
                    xytext=(dx, -3), textcoords="offset points", fontsize=6.3,
                    color=C["grey"], ha=ha)
    ax.set_yscale("log")
    ax.set_xlabel(r"energy capture $\Psi$")
    ax.set_ylabel(r"whiteness power $\pi$")
    ax.set_xlim(-0.03, 1.02)
    ax.set_ylim(PI_FLOOR * 0.8, 6.0)
    ax.grid(alpha=0.18, lw=0.5)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.5,
              borderaxespad=0.15, fontsize=6.4, labelspacing=0.35)
    p = os.path.join(OUT, "fig_region.pdf"); fig.savefig(p); plt.close(fig)
    print("wrote", p)


# ========================================================= FIG 3: dither ====
def fig_dither(regime="N2", budget_ref=0.10):
    """Randomization is not a stealth dial: Psi is monotone in sigma_d,
       exposure is not, and the safe setting moves with the budget."""
    dt, bl = dither_table(regime), blind_table(regime)
    n1 = dither_table("N1")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.9, 2.7))

    g = dt[dt.budget == budget_ref].sort_values("sigma_d_over_sqrtS")
    x = g.sigma_d_over_sqrtS.values
    a0.plot(x, g.psi, "-o", color=C["blue"], ms=4, label=r"$\Psi$ (damage)")
    a0.plot(x, g.power_acf1, "-^", color=C["red"], ms=4,
            label=r"$\pi$ (exposure)")
    a0.axhline(bl.loc[budget_ref, "psi"], color=C["blue"], lw=0.8, ls="--")
    a0.annotate(r"blind $\Psi$", xy=(x[-1], bl.loc[budget_ref, "psi"]),
                xytext=(-30, 5), textcoords="offset points", fontsize=7,
                color=C["blue"])
    a0.axhspan(0, float(pi_nominal(regime)[1].loc[budget_ref]),
               color=C["green"], alpha=0.09, lw=0)
    imax = int(np.argmax(g.power_acf1.values))
    a0.annotate("exposure peaks\ninside the sweep",
                xy=(x[imax], g.power_acf1.values[imax]),
                xytext=(6, -4), textcoords="offset points", fontsize=7,
                color=C["red"])
    a0.set_xlabel(r"randomization scale $\sigma_{d}/\sqrt{S}$")
    a0.set_ylabel("dimensionless fraction")
    a0.set_title(r"$\bar\Gamma=%.2f$" % budget_ref, pad=3)
    a0.set_ylim(-0.03, 1.0)
    a0.grid(alpha=0.18, lw=0.5)
    a0.legend(loc="upper right", frameon=False)

    cmap = plt.get_cmap("viridis")
    for i, G in enumerate(BUDGETS):
        g = dt[dt.budget == G].sort_values("sigma_d_over_sqrtS")
        a1.plot(g.sigma_d_over_sqrtS, _clip(g.power_acf1), "-o", ms=3.2,
                color=cmap(i / (len(BUDGETS) - 1)),
                label=r"$\bar\Gamma=%.2f$" % G)
    n1hi = float(n1.power_acf1.max())
    a1.axhline(n1hi, color=C["grey"], lw=0.8, ls="--")
    a1.annotate("N1 ceiling (%.3f)" % n1hi, xy=(2.2, n1hi),
                xytext=(0, -9), textcoords="offset points", fontsize=6.5,
                color=C["grey"], ha="center")
    # the claim is the vertical spread at a COMMON setting, not the minimum
    xs = 0.5
    col = dt[np.isclose(dt.sigma_d_over_sqrtS, xs)]
    a1.axvline(xs, color=C["black"], lw=0.7, ls=":")
    a1.annotate(r"at $\sigma_{d}=%.2f\sqrt{S}$:  $\pi$ spans %.2f--%.2f"
                % (xs, col.power_acf1.min(), col.power_acf1.max()),
                xy=(xs, 1.0), xytext=(6, -2), textcoords="offset points",
                fontsize=6.5, color=C["black"])
    a1.set_yscale("log")
    a1.set_xlabel(r"randomization scale $\sigma_{d}/\sqrt{S}$")
    a1.set_ylabel(r"whiteness power $\pi$")
    a1.set_title("no common safe setting", pad=3)
    a1.grid(alpha=0.18, lw=0.5)
    a1.legend(loc="upper right", frameon=False, ncol=2, columnspacing=0.8,
              fontsize=6.3, handletextpad=0.4, labelspacing=0.3)
    a1.set_ylim(top=9.0)
    p = os.path.join(OUT, "fig_dither.pdf"); fig.savefig(p); plt.close(fig)
    print("wrote", p)


# ========================================================= FIG 4: budget ====
def fig_budget(regime="N2", budget=0.02):
    """Budget accuracy against calibration length, and the collapse.

       Left: relative bias of the realized rate against N, one curve per L.
       Right: the same points against the expected number of calibration
       exceedances N*Gbar/L.  The curves collapse, which is the content of
       the window-selection accounting."""
    d = _load("F10_budget_convergence.csv")
    d = d[(d.regime == regime) & (d.budget == budget)]
    if not len(d):
        print("fig_budget: no rows for regime=%s budget=%s" % (regime, budget))
        return
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.9, 2.7))
    cmap = plt.get_cmap("viridis")
    Ls = sorted(d.L.unique())
    for i, L in enumerate(Ls):
        g = d[d.L == L].sort_values("N_cal")
        col = cmap(i / max(len(Ls) - 1, 1))
        a0.plot(g.N_cal, np.abs(g.rate_rel_bias), "-o", ms=4, color=col,
                label=r"$L=%d$" % L)
        a1.plot(g.exc_cal, np.abs(g.rate_rel_bias), "-o", ms=4, color=col,
                label=r"$L=%d$" % L)
    for a in (a0, a1):
        a.set_xscale("log"); a.set_yscale("log")
        a.axhline(0.05, color=C["grey"], lw=0.8, ls="--")
        a.set_ylabel(r"$|\hat\Gamma-\bar\Gamma|/\bar\Gamma$")
        a.grid(alpha=0.18, lw=0.5)
    a0.annotate(r"$5\%$", xy=(d.N_cal.min(), 0.05), xytext=(0, 4),
                textcoords="offset points", fontsize=7, color=C["grey"])
    a0.set_xlabel(r"calibration record length $N$")
    a0.set_title(r"at fixed $N$, memory costs accuracy", pad=3)
    a1.set_xlabel(r"calibration exceedances $\alpha N\approx N\bar\Gamma/L$")
    a1.set_title("the curves collapse", pad=3)
    a0.legend(loc="lower left", frameon=False)
    p = os.path.join(OUT, "fig_budget.pdf"); fig.savefig(p); plt.close(fig)
    print("wrote", p)


# ============================================================== TABLES ======
def _fmt(x, n=3):
    return ("$%." + str(n) + "f$") % x


def tab_main(regime="N2"):
    bl, ml, cv = blind_table(regime), memless_table(regime), covert_pick(regime)
    print("\n%% ---- tab:main  (regime=%s, W=%d, n_win=%d) ----"
          % (regime, W_REF, N_WIN))
    print("%% covert tolerance per budget: " +
          ", ".join("%.2f:%.3f" % (G, cv.loc[G, "pi_tol"]) for G in BUDGETS))
    for G in BUDGETS:
        b, m, c = bl.loc[G], ml.loc[G], cv.loc[G]
        star = "" if c.feasible else r"\dag"
        print(r"$%.2f$ & %s & %s & %s & %s & $(%d,%.2f)$ & %s & %s%s & %s & %s \\"
              % (G, _fmt(b.psi), _fmt(b.power_acf1), _fmt(m.psi),
                 _fmt(m.power_acf1), c.L, c.f, _fmt(c.psi), _fmt(c.pi), star,
                 _fmt(c.psi / m.psi, 2),
                 ("$%.1f\\times$" % (c.psi / b.psi))))
    print(r"%% covert feasible at every budget: %s   n_feasible: %s"
          % (bool(cv.feasible.all()), list(cv.n_feasible)))


def tab_camps(regime="N2", budget=0.02):
    d = _load("F9_camps.csv")
    d = d[(d.regime == regime) & (d.budget == budget)]
    label = {"MEMLESS": r"memoryless \cite{din2026distributionfreebudgetedstealthyattack}",
             "AR": "one-step magnitude predictor",
             "MAGDIFF": "magnitude difference",
             "ENERGY": r"windowed energy ($L=50$)",
             "BLOCK": r"fixed hold ($L=50$)",
             "CORR": r"corrected \eqref{eq:hyst}"}
    print("\n%% ---- tab:camps  (regime=%s, Gbar=%.2f, W=%d) ----"
          % (regime, budget, int(d.W.iloc[0])))
    for k in ["MEMLESS", "AR", "MAGDIFF", "ENERGY", "BLOCK", "CORR"]:
        r = d[d.scheduler == k]
        if not len(r):
            continue
        r = r.iloc[0]
        if k == "CORR":
            print(r"\midrule")
        print(r"%-34s & %s & %s & %s & $%.1f$ & %s \\"
              % (label[k], _fmt(r.psi), _fmt(r.power_acf1),
                 _fmt(r.max_rhat), r.mean_burst_len,
                 _fmt(r.mean_absz_fire)))


def tab_budget(regime="N2", budget=0.02, Ns=(250, 1000, 5000, 20000, 50000)):
    """Table III body.  The spread column is the OBSERVED s.d. of the realized
       rate divided by the i.i.d. prediction sqrt(Gbar(1-Gbar)/N), which is the
       factor memory costs; it is close to sqrt(L)."""
    try:
        d = _load("F10_budget_convergence.csv")
    except FileNotFoundError:
        print("\n%% tab:budget -- run cps_companion_F10.py first")
        return
    d = d[(d.regime == regime) & (d.budget == budget)].sort_values(["L", "N_cal"])
    print("\n%% ---- tab:budget  (regime=%s, Gbar=%.2f, CONF_START) ----"
          % (regime, budget))
    for L in sorted(d.L.unique()):
        g = d[d.L == L]
        ne = float(g[g.N_cal >= 5000].N_eff_over_N.mean())
        print(r"\midrule")
        print(r"\multicolumn{7}{l}{$L=%d$ \quad ($\Neff/N=%.3f$)} \\" % (L, ne))
        for _, r in g[g.N_cal.isin(Ns)].iterrows():
            print(r"$%d$ & $%.1f$ & %s & %s & %s & $%.1f$ & $%.2f$ \\"
                  % (r.N_cal, r.exc_cal, _fmt(r.alpha_mean, 4),
                     _fmt(r.rate_mean, 4), _fmt(r.rate_rel_bias, 3),
                     r.rate_std / r.std_pred_iid, r.viol_rate_frac))
    print(r"%% pathwise Ghat <= L*(start rate) violations: %d / %d rows"
          % (int((d.viol_Lpath_frac > 0).sum()), len(d)))
    print(r"%% s.d. ratio vs sqrt(L): " + ", ".join(
        "L=%d: %.1f vs %.1f" % (L, (d[(d.L == L) & (d.N_cal >= 5000)].rate_std
                                    / d[(d.L == L) & (d.N_cal >= 5000)].std_pred_iid).mean(),
                                np.sqrt(L)) for L in sorted(d.L.unique())))


# ================================================================== main ====
ALL = dict(main=fig_main, region=fig_region, dither=fig_dither,
           budget=fig_budget, tmain=tab_main, tcamps=tab_camps,
           tbudget=tab_budget)

if __name__ == "__main__":
    want = sys.argv[1:] or list(ALL)
    for k in want:
        if k not in ALL:
            print("unknown target", k, "-- choose from", list(ALL)); continue
        ALL[k]()
