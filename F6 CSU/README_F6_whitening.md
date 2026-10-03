# F6 whitening + re-run — what changed and how to run

## Why
The first F6 pass loaded `R2_innovations.npz` and treated **every array** as a
segment. That file stores four predictor tiers per segment:
`T0_diff, T1_ar, T2_varx, T3_gru`. `T0_diff` is pure differencing (lag-1
autocorrelation ≈ −0.5 **by construction**), so pooling all tiers inflated the
NOATTACK baseline to ≈0.39 and buried the attack-induced effect.

Measured contrast (same data): `T0_diff` 0.500, `T1_ar` 0.264, **`T2_varx` 0.040**.
`T2_varx` (the MIMO VARX least-squares residual, the tier R4 already uses) is the
actual innovation.

## Two new scripts
- **`cps_realdata_whiten.py`** — produces a clean, single-tier, whitened
  innovation file `results/R2_innovations_white.npz` with a per-segment
  `baseline_maxr` whiteness tag.
  - `--mode refit` (default): re-fit VARX over an order sweep and pick the order
    with the lowest NOATTACK baseline exposure. This is the "raise the order
    until the baseline → 0" knob.
  - `--mode from_existing`: no CAN log needed — just keep the `T2_varx` tier of
    the existing `R2_innovations.npz`.
  - Also writes `Rw_whitening_diag.csv` (baseline_maxr, Ljung-p, kurtosis per
    order/segment) so you can see whiteness improve with order.
- **`cps_companion_F6.py` (v3)** — three fixes:
  1. **Tier-aware loader**: uses the clean `segments` file, or selects only
     `T2_varx` from a raw file — never `T0_diff`.
  2. **Attack-induced Δ**: headline exposure is
     `max_l |r̂_l(attacked) − r̂_l(nominal)|`, so residual model correlation
     cancels and only what the attack adds is measured.
  3. **Whiteness gating**: prints the SUMMARY twice — over **all** segments and
     over the **well-whitened** subset (`baseline_maxr < --white_thresh`, default
     0.05) — the proof-of-principle cohort where Prop. 2's precondition holds.

## Run it
```
sbatch run_whiten_F6.slurm
```
Paste **both SUMMARY blocks** (cohort=all and cohort=white) from the log, or send
`results/F6b_summary.csv`, `results/F6b_power.csv`, `results/Rw_whitening_diag.csv`.

## What to look for
- In `Rw_whitening_diag.csv`: does `baseline_maxr` keep dropping as order rises,
  or flatten? If p*=120 is still selected, raise the ceiling in the sweep.
- In the SUMMARY: on the **white** cohort, does `MEM_pow` (memoryless whiteness
  power) rise **above** the ~0.01 FAR — i.e. is the memoryless attack now
  detectable, as Prop. 2 predicts on a white innovation? And does `CORRtun_dr`
  (tuned corrected rule) pull the attack-induced exposure back down while keeping
  `CORRtun_psi` (damage) high — R1 on real data?
