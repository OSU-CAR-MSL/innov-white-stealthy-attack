#!/usr/bin/env python
# coding: utf-8
"""
collect_companion.py -- bundle the companion results (F1..F6) into ONE zip that
is easy to hand back for analysis, plus quick sanity summaries printed to stdout.

Usage:
    python collect_companion.py --in results --out companion_results.zip
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import zipfile
from pathlib import Path


def summarize(indir):
    try:
        import pandas as pd
    except Exception:
        print("(pandas not available; skipping summaries)")
        return
    for f in sorted(glob.glob(os.path.join(indir, "F*_*.csv"))):
        try:
            d = pd.read_csv(f)
        except Exception as e:
            print(f"  ! could not read {f}: {e}"); continue
        name = Path(f).stem
        print(f"\n== {name}  ({len(d)} rows, cols: {', '.join(d.columns[:10])}"
              f"{'...' if len(d.columns) > 10 else ''}) ==")
        if name.startswith("F4") and {"scheduler", "power_acf1", "psi"} <= set(d.columns):
            sub = d[d.get("W", 0) == d.get("W", 0).max()] if "W" in d else d
            g = (sub.groupby(["regime", "scheduler"])
                    [["psi", "power_acf1", "max_rhat"]].mean().round(3))
            print(g.to_string())
        elif name.startswith("F3") and "N_eff_over_N" in d.columns:
            print(d.groupby(["regime", "L"])[["N_eff_over_N", "rate_std",
                  "rate_std_neff", "conformal_violation"]].mean().round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="results")
    ap.add_argument("--out", default="companion_results.zip")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.indir, "F*_*.csv")) +
                   glob.glob(os.path.join(a.indir, "F*_*_meta.json")))
    if not files:
        print(f"No companion result files in {a.indir}/ (expected F*_*.csv).")
        return
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=os.path.basename(f))
    print(f"Wrote {a.out} with {len(files)} files:")
    for f in files:
        print("   ", os.path.basename(f), f"({os.path.getsize(f)} B)")
    summarize(a.indir)
    print(f"\nHand back: {a.out}  (all six experiments, tidy long CSV).")


if __name__ == "__main__":
    main()
