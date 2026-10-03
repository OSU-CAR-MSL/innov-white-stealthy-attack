#!/usr/bin/env python3
# =============================================================================
#  make_companion_figs2.py  --  figures for the companion (revised)
#
#  Place in the directory that holds the result CSVs (F1_dither.csv,
#  F2_found_memory.csv, F3_price_of_memory.csv, F4_corrected_rule.csv,
#  F5_region.csv).  Run:
#
#      python3 make_companion_figs2.py            # all figures
#      python3 make_companion_figs2.py region     # just one
#
#  Writes PDFs to  figs/results/ .  Print-safe, colour-blind-safe
#  (Okabe--Ito), no chartjunk.  Only matplotlib + numpy required.
#
#  Figures
#    region     fig_region.pdf     damage--exposure plane (V-E + V-F merged)
#    operating  fig_operating.pdf  blind->corrected(covert)->memoryless, all Gbar
#    camps      fig_camps.pdf      windowed-score two-camp scatter (V-D, optional)
#    price      fig_price.pdf      N_eff/N and realized-rate s.d. vs L (V-G, opt.)
# =============================================================================
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---- style -----------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.4,
    "font.family": "serif",
})
# Okabe--Ito
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", grey="#7F7F7F", sky="#56B4E9", black="#000000")

OUT = os.path.join("figs", "results")
os.makedirs(OUT, exist_ok=True)


def _find(name):
    """Locate a CSV by clean basename, tolerating a hex- prefix."""
    if os.path.exists(name):
        return name
    for fn in os.listdir("."):
        if fn.endswith(name):          # e.g. 'ffc609ec-F1_dither.csv'
            return fn
    raise FileNotFoundError(name)


def _load(name):
    with open(_find(name)) as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# =============================================================================
def fig_region(budget=0.10, regime="N2", W="5000"):
    """(pi, Psi) plane: blind & memoryless corners, corrected frontier, dither."""
    F5 = _load("F5_region.csv")
    def sub(sched):
        out = []
        for r in F5:
            if (r["regime"] == regime and r["W"] == W
                    and _f(r["budget"]) == budget and r["scheduler"] == sched
                    and _f(r["power_acf1"]) is not None):
                out.append((_f(r["power_acf1"]), _f(r["psi"]),
                            _f(r.get("sigma_d")), _f(r.get("L")),
                            _f(r.get("beta"))))
        return out

    corr = sub("CORR")
    dith = sub("DITHER")
    ml = sub("MEMLESS")

    # blind: prefer explicit blind row in F1
    blind = None
    for r in _load("F1_dither.csv"):
        if (r["regime"] == regime and r.get("blind", "").strip() == "True"
                and r["W"] == W and _f(r["budget"]) == budget):
            blind = (_f(r["power_acf1"]), _f(r["psi"]))
            break
    if blind is None:
        blind = (0.01, budget)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # corrected: upper envelope (max Psi at each pi, cumulative) as the frontier
    cs = sorted(corr, key=lambda t: t[0])
    px = [t[0] for t in cs]; py = [t[1] for t in cs]
    # efficient frontier: best Psi achievable at pi <= x
    fx, fy, best = [], [], -1
    for x, y in zip(px, py):
        best = max(best, y)
        fx.append(x); fy.append(best)
    ax.fill_between(fx, blind[1], fy, color=C["blue"], alpha=0.08, zorder=0)
    ax.plot(fx, fy, color=C["blue"], lw=1.8, zorder=4, label="corrected frontier")
    ax.scatter(px, py, s=14, color=C["blue"], alpha=0.55, zorder=3,
               edgecolor="none")

    # dither curve, ordered by sigma_d (shows the rightward loop)
    ds = sorted([t for t in dith if t[2] is not None], key=lambda t: t[2])
    if ds:
        ax.plot([t[0] for t in ds], [t[1] for t in ds], color=C["orange"],
                lw=1.4, marker="o", ms=3, zorder=3, label="dither ($\\sigma_d\\uparrow$)")

    # corners
    ax.scatter([ml[0][0]] if ml else [], [ml[0][1]] if ml else [],
               marker="s", s=55, color=C["red"], zorder=6,
               label="memoryless")
    ax.scatter([blind[0]], [blind[1]], marker="D", s=45, color=C["green"],
               zorder=6, label="blind")

    # nominal line
    ax.axvline(0.10, color=C["grey"], ls=":", lw=0.8, zorder=1)
    ax.text(0.101, ax.get_ylim()[0] + 0.02, "$\\pi=0.10$", fontsize=6.5,
            color=C["grey"], rotation=90, va="bottom")

    ax.set_xlabel("lag-one whiteness power  $\\pi$")
    ax.set_ylabel("energy capture  $\\Psi$")
    ax.set_xlim(-0.02, max(0.9, max(px + [ml[0][0]] if ml else px) + 0.03))
    ax.set_ylim(0.0, 0.7)
    ax.set_title(f"N2, $\\bar\\Gamma={budget}$, $W={W}$", fontsize=8)
    ax.legend(loc="lower right", frameon=False, handlelength=1.6)
    # arrow to the good corner
    ax.annotate("efficient\ncorner", xy=(0.03, 0.55), xytext=(0.32, 0.60),
                fontsize=6.5, color=C["blue"], ha="center",
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=0.8))
    p = os.path.join(OUT, "fig_region.pdf")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


# =============================================================================
def fig_operating(regime="N2", W="5000"):
    """Blind -> corrected(covert) -> memoryless, connected per budget."""
    F4 = [r for r in _load("F4_corrected_rule.csv")
          if r["regime"] == regime and r["W"] == W]
    budgets = sorted({_f(r["budget"]) for r in F4})
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(budgets)))
    for b, col in zip(budgets, cmap):
        rows = [r for r in F4 if _f(r["budget"]) == b]
        ml = [r for r in rows if r["scheduler"] == "MEMLESS"]
        if not ml:
            continue
        star = _f(ml[0]["psi"]); piml = _f(ml[0]["power_acf1"])
        corr = [r for r in rows if r["scheduler"] == "CORR"]
        covert = [r for r in corr if _f(r["power_acf1"]) <= 0.02]
        cov = max(covert, key=lambda r: _f(r["psi"])) if covert else \
            min(corr, key=lambda r: _f(r["power_acf1"]))
        xs = [0.01, _f(cov["power_acf1"]), piml]
        ys = [b, _f(cov["psi"]), star]
        ax.plot(xs, ys, "-", color=col, lw=1.0, alpha=0.9)
        ax.scatter([xs[0]], [ys[0]], marker="D", s=22, color=col, zorder=5)
        ax.scatter([xs[1]], [ys[1]], marker="o", s=26, color=col, zorder=5)
        ax.scatter([xs[2]], [ys[2]], marker="s", s=26, color=col, zorder=5)
        ax.text(piml + 0.01, star, f"${b:g}$", fontsize=6.5, color=col,
                va="center")
    ax.set_xlabel("lag-one whiteness power  $\\pi$")
    ax.set_ylabel("energy capture  $\\Psi$")
    ax.set_title("blind (◇) → corrected/covert (○) → memoryless (□)",
                 fontsize=7.0)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0.0, 1.0)
    p = os.path.join(OUT, "fig_operating.pdf")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


# =============================================================================
def fig_camps(regime="N2", budget=0.02):
    """Two-camp scatter of windowed scores in (max|r_hat|, Psi)."""
    import statistics as st
    F2 = _load("F2_found_memory.csv")
    names = [("MEMLESS", "memoryless", C["red"], "s"),
             ("AR", "1-step mag. pred.", C["orange"], "o"),
             ("MAGDIFF", "mag. difference", C["purple"], "o"),
             ("ENERGY", "windowed energy", C["sky"], "^"),
             ("BLOCK", "fixed hold", C["grey"], "^")]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    for key, lab, col, mk in names:
        rs = [r for r in F2 if r["regime"] == regime
              and r["scheduler"] == key and _f(r["budget"]) == budget]
        if not rs:
            continue
        x = st.mean(_f(r["max_rhat"]) for r in rs)
        y = st.mean(_f(r["psi"]) for r in rs)
        ax.scatter([x], [y], s=48, color=col, marker=mk, zorder=4,
                   edgecolor="white", linewidth=0.5, label=lab)
    # corrected point from F4 (L=4,f=0.3)
    for r in _load("F4_corrected_rule.csv"):
        if (r["regime"] == regime and r["W"] == "5000"
                and _f(r["budget"]) == budget and r["scheduler"] == "CORR"
                and _f(r["L"]) == 4 and _f(r["f_cont"]) == 0.3):
            ax.scatter([_f(r["max_rhat"])], [_f(r["psi"])], s=70, marker="*",
                       color=C["blue"], zorder=6, edgecolor="white",
                       linewidth=0.5, label="corrected")
            break
    ax.axhline(budget, color=C["green"], ls=":", lw=0.8)
    ax.text(ax.get_xlim()[1], budget, " blind $\\Psi=\\bar\\Gamma$",
            fontsize=6.5, color=C["green"], va="bottom", ha="right")
    ax.set_xlabel("exposure  $\\max_\\ell|\\hat r_\\ell|$")
    ax.set_ylabel("energy capture  $\\Psi$")
    ax.set_title(f"windowed scores, N2, $\\bar\\Gamma={budget}$ "
                 "(matched firing)", fontsize=7.5)
    ax.legend(loc="upper right", frameon=False, handlelength=1.2, ncol=1)
    p = os.path.join(OUT, "fig_camps.pdf")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


# =============================================================================
def fig_price(regime="N2", budget=0.02):
    """N_eff/N and realized-rate s.d. (obs vs iid) against L."""
    F3 = [r for r in _load("F3_price_of_memory.csv")
          if r["regime"] == regime and _f(r["budget"]) == budget]
    F3.sort(key=lambda r: _f(r["L"]))
    L = [_f(r["L"]) for r in F3]
    neff = [_f(r["N_eff_over_N"]) for r in F3]
    sd_obs = [_f(r["rate_std"]) for r in F3]
    sd_iid = [_f(r["rate_std_iid"]) for r in F3]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    a1.loglog(L, neff, "o-", color=C["blue"], label="measured")
    a1.loglog(L, [1.0 / x for x in L], "--", color=C["grey"], label="$1/L$")
    a1.set_xlabel("window $L$"); a1.set_ylabel("$N_{\\mathrm{eff}}/N$")
    a1.set_title("effective calibration fraction", fontsize=8)
    a1.legend(frameon=False)

    a2.loglog(L, sd_obs, "o-", color=C["red"], label="observed")
    a2.loglog(L, sd_iid, "--", color=C["grey"], label="i.i.d.")
    a2.set_xlabel("window $L$")
    a2.set_ylabel("realized-rate s.d.")
    a2.set_title("price is variance, not bias", fontsize=8)
    a2.legend(frameon=False)
    fig.suptitle(f"N2, $\\bar\\Gamma={budget}$, $N=5000$", fontsize=8, y=1.02)
    p = os.path.join(OUT, "fig_price.pdf")
    fig.savefig(p); plt.close(fig)
    print("wrote", p)


# =============================================================================
FIGS = {"region": fig_region, "operating": fig_operating,
        "camps": fig_camps, "price": fig_price}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for name in which:
        if name in FIGS:
            FIGS[name]()
        else:
            print("unknown figure:", name, "| choices:", *FIGS)
