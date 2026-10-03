#!/usr/bin/env python
# coding: utf-8
"""
=============================================================================
 make_companion_figs.py  --  the two figures of the companion paper

 Keep this in the SAME directory as the experiment scripts / results, e.g.
     budget_attack/Gsgn/
 It reads the tidy CSVs already produced there and writes
     figs/results/fig_dither.pdf     -> Fig. 1(a)
     figs/results/fig_region.pdf     -> Fig. 1(b)

 Usage
 -----
   python make_companion_figs.py                       # results/ -> figs/results/
   python make_companion_figs.py --in . --out figs/results
   python make_companion_figs.py --regime N2 --W 5000 --budget 0.10

 Design constraints (deliberate, do not "improve" away)
 -----------------------------------------------------
 * ONE y-axis per panel.  Psi and detector power are both dimensionless
   fractions on [0,1], so they share an axis legitimately; a twin-axis
   version of (a) would be a dual-scale chart and is not used.
 * Colour is never the only encoding: every series also carries its own
   marker shape and a direct label, so the figures survive greyscale
   printing and colour-vision deficiency.
 * Palette is the validated categorical set
   #2a78d6 / #eb6834 / #1baf7a / #4a3aa7 (blue, orange, aqua, violet),
   checked for CVD separation and lightness band on a white surface.
=============================================================================
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- validated categorical palette -----------------------------------------
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#d8d8d4"

def style():
    plt.rcParams.update({
        "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.linewidth": 0.6, "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "figure.dpi": 200, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def recede(ax):
    ax.grid(True, axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ===========================================================================
# Fig. 1(a) -- dithering is not a stealth dial
# ===========================================================================
def fig_dither(indir, outdir, regime="N2", W=5000, budget=0.10):
    f = pd.read_csv(Path(indir) / "F1_dither.csv")
    d = f[(f.regime == regime) & (f.W == W) & (f.budget == budget)].copy()
    if d.empty:
        print(f"  [dither] no rows for regime={regime} W={W} budget={budget}")
        return
    d = d.sort_values("sigma_d_over_sqrtS")
    blind = d[d.blind == True]                      # noqa: E712
    swept = d[d.blind == False]                     # noqa: E712
    x = np.arange(len(swept))
    labels = [f"{v:g}" for v in swept.sigma_d_over_sqrtS]
    if len(blind):
        x = np.append(x, len(swept) + 0.6)
        labels = labels + ["blind"]
        psi = np.append(swept.psi.values, blind.psi.values[0])
        pw = np.append(swept.power_acf1.values, blind.power_acf1.values[0])
    else:
        psi, pw = swept.psi.values, swept.power_acf1.values

    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    ax.plot(x, psi, color=BLUE, marker="o", ms=3.6, label=r"$\Psi$ (damage)",
            zorder=3)
    ax.plot(x, pw, color=ORANGE, marker="s", ms=3.6, ls="--",
            label="whiteness power", zorder=3)
    # nominal false-alarm level
    far = float(d.far_nominal.iloc[0]) if "far_nominal" in d else 0.01
    ax.axhline(far, color=MUTED, lw=0.6, ls=":", zorder=1)
    ax.annotate(f"nominal FAR ({far:g})", (x[-1], far), xytext=(-2, 4),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=6, color=MUTED)
    # mark the non-monotone peak in exposure
    j = int(np.argmax(pw))
    ax.annotate("exposure rises\nbefore it falls", (x[j], pw[j]),
                xytext=(8, 6), textcoords="offset points", fontsize=6,
                color=ORANGE, va="bottom",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.5))
    # direct labels (relief rule: identity never colour-alone)
    ax.annotate(r"$\Psi$", (x[0], psi[0]), xytext=(3, 5),
                textcoords="offset points", fontsize=7, color=BLUE)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel(r"randomization scale $\sigma_d/\sqrt{S}$")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.16), handlelength=1.6,
              columnspacing=1.0)
    recede(ax)
    p = Path(outdir) / "fig_dither.pdf"
    fig.savefig(p); plt.close(fig)
    print(f"  -> {p}")


# ===========================================================================
# Fig. 1(b) -- the achievable damage / exposure region
# ===========================================================================
def _pareto(pw, psi):
    """upper-left envelope: max psi at or below each power level."""
    o = np.argsort(pw)
    pw, psi = np.asarray(pw)[o], np.asarray(psi)[o]
    keep, best = [], -np.inf
    for i in range(len(pw)):
        if psi[i] > best:
            best = psi[i]; keep.append(i)
    return pw[keep], psi[keep]


def fig_region(indir, outdir, regime="N2", W=5000, budget=0.10):
    ind = Path(indir)
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    # NOTE: only families for which a detector power is measured appear here.
    # The windowed scores of Table II carry the exposure functional but not a
    # pooled power, so they are reported in that table rather than plotted.

    # --- corrected rule: all (L, f) at this budget -> frontier -------------
    f4 = pd.read_csv(ind / "F4_corrected_rule.csv")
    c = f4[(f4.regime == regime) & (f4.W == W) & (f4.budget == budget)]
    corr = c[c.scheduler == "CORR"]
    mem = c[c.scheduler == "MEMLESS"]
    if len(corr):
        ax.scatter(corr.power_acf1, corr.psi, s=11, facecolor="none",
                   edgecolor=BLUE, linewidths=0.7, zorder=3,
                   label="corrected $(L,f)$")
        fx, fy = _pareto(corr.power_acf1.values, corr.psi.values)
        ax.plot(fx, fy, color=BLUE, lw=1.4, zorder=4)
        ax.annotate("corrected frontier", (fx[0], fy[0]), xytext=(6, -8),
                    textcoords="offset points", fontsize=6.5, color=BLUE)

    # --- dither curve ------------------------------------------------------
    f1 = pd.read_csv(ind / "F1_dither.csv")
    d1 = f1[(f1.regime == regime) & (f1.W == W) & (f1.budget == budget)]
    dd = d1[d1.blind == False].sort_values("sigma_d_over_sqrtS")   # noqa: E712
    if len(dd):
        ax.plot(dd.power_acf1, dd.psi, color=ORANGE, marker="s", ms=3.2,
                ls="--", lw=1.2, zorder=3, label="dithered family")

    # --- reference points --------------------------------------------------
    if len(mem):
        ax.scatter(mem.power_acf1, mem.psi, marker="*", s=70, color=VIOLET,
                   zorder=5, label=r"memoryless $\gamma^\star$")
        ax.annotate(r"memoryless $\gamma^\star$",
                    (float(mem.power_acf1.iloc[0]), float(mem.psi.iloc[0])),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=6.5, color=VIOLET, ha="left")
    ax.scatter([0.01], [budget], marker="o", s=22, color=MUTED, zorder=5)
    ax.annotate("blind", (0.01, budget), xytext=(6, -1),
                textcoords="offset points", fontsize=6.5, color=MUTED)

    ax.set_xlabel("lag-one whiteness power")
    ax.set_ylabel(r"energy capture $\Psi$")
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6,
              fontsize=6.5)
    recede(ax)
    ax.grid(True, axis="x", alpha=0.5)
    p = Path(outdir) / "fig_region.pdf"
    fig.savefig(p); plt.close(fig)
    print(f"  -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="results")
    ap.add_argument("--out", dest="outdir", default="figs/results")
    ap.add_argument("--regime", default="N2")
    ap.add_argument("--W", type=int, default=5000)
    ap.add_argument("--budget", type=float, default=0.10)
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    style()
    print(f"[figs] regime={a.regime} W={a.W} budget={a.budget}")
    fig_dither(a.indir, a.outdir, a.regime, a.W, a.budget)
    fig_region(a.indir, a.outdir, a.regime, a.W, a.budget)
    print("[figs] done")


if __name__ == "__main__":
    main()
