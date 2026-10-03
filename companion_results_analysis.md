# Companion-paper experiments — results & analysis (F1–F6)

*Run: full production. N1 (Gaussian) and N2 (heavy-tailed) synthetic regimes;
calibration N=5000, deployment T=50,000, 50 Monte-Carlo seeds (F3: 200
calibration draws); real data = 90 stationary CSU CAN segments via VARX
innovations. All schedulers sign-invariant. Exposure functional
`max_rhat` = max over lags ℓ=1..10 of |r̂_ℓ|/S on the received stream;
`power_acf1` = rejection rate of a lag-1 whiteness monitor at window W.*

---

## 0. The headline, in five sentences

1. **The corrected rule works and it is the paper's result.** At the operating
   point where the letter's memoryless rule is *certainly* detected (N2, Γ̄=0.02,
   whiteness power **1.000**), the corrected scheduler CORR drops power to
   **0.014** — nominal — while keeping **86%** of the damage (Ψ 0.294 vs 0.343).
2. **Randomization (dither) is not a substitute:** Ψ falls monotonically with the
   dither scale, but exposure is non-monotone and the "safe" dither level moves
   with both the budget and the noise law — so it cannot be set without the
   distributional knowledge the framework avoids.
3. **Memory is only needed off-Gaussian:** under N1 the memoryless rule is already
   at nominal power at every budget, so CORR has nothing to correct — exactly the
   "memory necessary ⇔ not conditionally sign-symmetric" statement, visible in data.
4. **The price of memory is quantified and real:** the budget *guarantee* holds,
   but effective calibration size collapses as N_eff ≈ N/L, down to ~2 usable
   exceedances at L=50, Γ̄=0.02 — which sets a hard upper bound on useful window
   length.
5. **On real vehicle data the premise is confirmed but transfer is not:** the
   exposure constraint is strongly *active* (MEMLESS max_rhat ≈ 0.20, ~20× nominal),
   yet a single CORR configuration tuned on synthetic N2 does **not** reduce it —
   the correction must be tuned to the data's lag structure. This is an honest
   limitation that motivates the multi-lag / per-deployment tuning.

Everything below is the evidence for these five.

---

## F1 — Dithering is not a stealth dial

**What it asks in plain terms:** the easiest way to look less suspicious is to add
random noise to the firing score so you stop firing exactly on the biggest spikes.
Does turning that noise knob (σ_d) trade damage for stealth cleanly?

**What the data says (N2, heavy-tailed):**

| σ_d/√S | 0 (=B3) | 0.25 | 0.5 | 1 | 2 | 4 | blind |
|---|---|---|---|---|---|---|---|
| Ψ (damage), Γ̄=0.10, W=5000 | 0.607 | 0.485 | 0.367 | 0.238 | 0.161 | 0.129 | 0.103 |
| power, Γ̄=0.10, W=5000 | 0.255 | 0.279 | **0.798** | 0.505 | 0.130 | 0.046 | 0.016 |

Damage falls **monotonically** with σ_d — as expected. Exposure does **not**: at
Γ̄=0.10 it actually *rises* to 0.798 at σ_d=0.5 before falling. And the shape
changes with budget — the whiteness-power surface (N2, W=5000):

| Γ̄ \ σ_d | 0 | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|
| 0.02 | 1.000 | 1.000 | 0.962 | 0.283 | 0.032 | 0.016 |
| 0.10 | 0.255 | 0.279 | 0.798 | 0.505 | 0.130 | 0.046 |
| 0.20 | 0.858 | 0.088 | 0.122 | 0.230 | 0.092 | 0.038 |
| 0.50 | 0.128 | 0.605 | 0.387 | 0.092 | 0.042 | 0.036 |

Read across the rows: at Γ̄=0.02 you need a *large* dither (σ_d≥2) to hide; at
Γ̄=0.20 a *small* one (σ_d≈0.25); at Γ̄=0.50 the relationship inverts (small
dither makes it *worse*). **There is no single σ_d that is safe across budgets.**

**Control (N1, Gaussian):** power is at nominal (0.004–0.022) for every σ_d and
budget — there is no serial exposure to dither away in the first place.

**Its place in the paper:** this is the motivating negative result. It proves you
cannot buy correlation-stealth by randomizing, because the safe setting depends on
exactly the noise law you don't know. That is what forces a *structured* memory
correction — the rest of the paper.

---

## F2 — Do "off-the-shelf" memory scores buy stealth for the damage they lose?

**What it asks:** several windowed scores (AR predictor on magnitudes, energy,
magnitude-difference, block-hold) all lose damage versus firing on the raw tail.
In the companion the question flips: in exchange, do they reduce exposure? All
compared at *matched firing rate*.

**Damage (Ψ), N2:**

| scheduler | 0.02 | 0.10 | 0.20 | 0.50 | mean burst len |
|---|---|---|---|---|---|
| MEMLESS (B3) | 0.350 | 0.608 | 0.761 | 0.952 | ~1.2 |
| AR | 0.348 | 0.583 | 0.639 | 0.759 | ~1.3 |
| MAGDIFF | 0.281 | 0.480 | 0.593 | 0.798 | ~1.7 |
| ENERGY | 0.049 | 0.196 | 0.343 | 0.669 | ~18–35 |
| BLOCK (L=50) | 0.050 | 0.202 | 0.354 | 0.688 | ~50 |

**Exposure (max_rhat), N2:**

| scheduler | 0.02 | 0.10 | 0.20 | 0.50 |
|---|---|---|---|---|
| MEMLESS | 0.122 | 0.031 | 0.060 | 0.023 |
| AR | 0.150 | 0.022 | 0.014 | 0.043 |
| MAGDIFF | 0.078 | 0.053 | 0.062 | 0.081 |
| ENERGY | 0.009 | 0.010 | 0.010 | 0.009 |
| BLOCK | 0.008 | 0.009 | 0.009 | 0.009 |

**The finding — the found rules split into two camps, and neither is what we want:**
- **AR and MAGDIFF keep the damage** (AR essentially ties B3 at small budget,
  0.348 vs 0.350) but **stay exposed** — they still fire in short bursts on
  near-tail instants, so their exposure tracks MEMLESS.
- **ENERGY and BLOCK kill the exposure** (max_rhat ≈ 0.01, flat) but **throw away
  most of the damage** — they fire in long diffuse bursts (length 18–50) far off
  the tail.

No off-the-shelf rule sits in the corner we care about — **high damage *and* low
exposure**. Damage tracks tail-concentration: mean |z| at firing is highest for
MEMLESS (0.47 at Γ̄=0.20) and lowest for BLOCK/ENERGY (0.23), and Ψ ranks in the
same order. That is the letter's mechanism confirmed and the gap the corrected
rule must fill.

**Its place in the paper:** existence + motivation. Memory *can* move exposure
(ENERGY/BLOCK prove it) but the naive rules are dominated — they buy stealth only
by surrendering the damage that was the point. Motivates a principled correction.

---

## F3 — The price of memory in calibration

**What it asks:** memory windows overlap, so consecutive firing decisions are
correlated. Does that break the distribution-free budget guarantee, and what does
it cost?

**N_eff / N (effective calibration fraction), N2:**

| L | 0.02 | 0.10 | 0.20 | 0.50 |
|---|---|---|---|---|
| 1 (memoryless) | 0.86 | 0.75 | 0.80 | 0.90 |
| 4 | 0.25 | 0.26 | 0.27 | 0.35 |
| 10 | 0.10 | 0.11 | 0.12 | 0.15 |
| 50 | 0.020 | 0.021 | 0.023 | 0.033 |

**Findings, all confirming the theory:**
- **N_eff ≈ N/L for block-type memory** (L=50 → 0.020 = 1/50, L=10 → 0.10 = 1/10).
  The earlier guess "≈N/20 at L=50" was wrong — that number was a specific score's
  burst length, not a law. The law is N_eff = N/(1+2Σρ_ℓ) ≈ N/(effective burst).
- **Memoryless is ~0.98 under N1 but only ~0.75–0.90 under N2** (heavy-tail
  clustering of large |z| correlates the firing indicator), and it is
  budget-dependent — so even "no memory" is not exactly N under a heavy tail.
- **The guarantee holds; the variance inflates.** Realized-rate std tracks the
  N_eff-corrected prediction √(Γ̄(1−Γ̄)/N_eff), not the i.i.d. one:

  | L | N_eff | realized rate std | i.i.d. prediction | N_eff prediction |
  |---|---|---|---|---|
  | 1 | 4322 | 0.0024 | 0.0020 | 0.0021 |
  | 10 | 499 | 0.0069 | 0.0020 | 0.0063 |
  | 50 | 101 | 0.0189 | 0.0020 | 0.0139 |

  The i.i.d. floor understates the spread by ~10× at L=50.
- **The hard limit is small budgets.** Effective exceedances = N_eff·Γ̄:
  at L=50, Γ̄=0.02 there are only **≈2** — and there the realized rate even drifts
  above target (mean 0.031 vs 0.02) because a 98th percentile can't be estimated
  from 2 points. Conformal-violation fraction climbs from ~0.44 (memoryless, as
  the letter's Rem 3 predicts) to 0.72 at L=50/small budget.

**Design rule that falls out:** keep L small relative to N·Γ̄. At N=5000, Γ̄=0.02,
L≤10 keeps ≥50 effective exceedances and the rate within ~8% of budget; L=50 is
unusable. This dovetails with F4, where L≈4–6 is already enough for stealth.

**Its place in the paper:** the honest cost statement. It converts "memory is
free" into a quantified, bounded trade-off and yields a concrete window-length
ceiling.

---

## F4 — The corrected rule vs memoryless vs dither (the centerpiece)

**What it asks:** does the proposed corrected scheduler (hysteresis: start a flip
on a tail crossing, continue for up to L steps while |z|>f·ε_hi, budget held by
calibrating ε_hi) beat both the memoryless rule and dither — at the *same budget*?

**Budget is held:** across all 1,488 rows, max |realized rate − budget| = 0.012.

**The core result (N2, W=5000, Γ̄=0.02).** MEMLESS: Ψ=0.343 at power **1.000**.
CORR power as a function of (L, continue-gate f):

| L \ f | 0.0 | 0.15 | 0.30 | 0.50 | 0.70 |
|---|---|---|---|---|---|
| 2 | 0.263 | 0.251 | 0.080 | 0.170 | 0.880 |
| 4 | 0.016 | 0.016 | **0.014** | 0.299 | 0.894 |
| 6 | 0.012 | 0.014 | 0.018 | 0.299 | 0.896 |
| 10 | 0.012 | 0.018 | 0.018 | 0.299 | 0.896 |

and the matching Ψ (damage):

| L \ f | 0.0 | 0.15 | 0.30 | 0.50 | 0.70 |
|---|---|---|---|---|---|
| 4 | 0.208 | 0.254 | **0.294** | 0.326 | 0.341 |
| 10 | 0.127 | 0.239 | 0.293 | 0.326 | 0.341 |

Reading the two tables together: at **L=4, f=0.3** the rule sits at power **0.014**
(from 1.000) while Ψ is **0.294** (86% of MEMLESS's 0.343). f=0.7 is too mild
(power stays ~0.89); f=0 (pure block) over-pays in damage; **f≈0.3–0.5 is the
sweet spot**, and **L≈4–6 is enough** — power is flat for L≥6, so larger windows
only cost damage and calibration (F3).

**The sweet-spot summary (N2, W=5000): the lowest-exposure CORR that keeps Ψ within
10% of MEMLESS, at every budget:**

| Γ̄ | MEMLESS Ψ | MEMLESS power | → CORR (L, f) | CORR Ψ | CORR power |
|---|---|---|---|---|---|
| 0.02 | 0.343 | **1.000** | (2, 0.3) | 0.310 | **0.080** |
| 0.05 | 0.480 | 0.369 | (6, 0.5) | 0.455 | 0.020 |
| 0.10 | 0.607 | 0.255 | (6, 0.5) | 0.579 | 0.204 |
| 0.20 | 0.761 | **0.858** | (20, 0.3) | 0.685 | **0.218** |
| 0.30 | 0.853 | 0.745 | (20, 0.3) | 0.785 | 0.190 |
| 0.50 | 0.952 | 0.128 | (6, 0.3) | 0.912 | 0.054 |

Wherever MEMLESS is exposed (Γ̄=0.02: power 1.0; Γ̄=0.20: 0.86), CORR keeps ~90% of
the damage and cuts power by 4–70×.

**The Gaussian control (N1):** MEMLESS power is already nominal (0.008–0.020) at
*every* budget — under a Gaussian innovation there is nothing to hide, so CORR is
unnecessary. This is the theory made visible: **the correction matters exactly
when the innovation is not conditionally sign-symmetric, i.e. off-Gaussian.**

**Its place in the paper:** this is the experiment the paper is judged on, and it
delivers — CORR dominates both baselines at matched budget, the knob behaves as
the theory predicts, and the N1/N2 contrast is the corollary in data.

---

## F5 — The achievable damage/stealth region

**What it asks:** plot every rule on the (exposure, damage) plane. Who is on the
efficient frontier — the corner with low exposure and high damage?

**Using detector power as the exposure axis (the operational metric), N2, W=5000 —
best Ψ achievable at power ≤ 0.10:**

| Γ̄ | CORR | DITHER | (others below) |
|---|---|---|---|
| 0.02 | **0.310** | 0.043 | — |
| 0.10 | **0.539** | 0.129 | — |
| 0.20 | **0.626** | 0.605 | — |
| 0.30 | **0.608** | 0.591 | — |

CORR owns the frontier at every budget. Dither is far behind at small budgets
(0.043 vs 0.310) and only draws level at large budgets — where MEMLESS was barely
exposed anyway, so there is little to win.

**One subtlety worth knowing (and stating in the paper).** If you rank instead by
the *functional* max_rhat at Γ̄=0.10, MEMLESS and AR appear to edge CORR. That is
not a contradiction: Γ̄≈0.10 is near the budget where MEMLESS's lag-1 autocovariance
*crosses zero* (the interior extremum of the letter's Rem 6), so its functional is
incidentally small there even though a windowed detector still accumulates power.
The functional and the detector are two readings; the operational one (power) is
the honest exposure axis, and by it CORR dominates. CORR's advantage is largest
exactly where MEMLESS is most exposed (small and large budgets), smallest at the
zero-crossing — which is itself the Rem 6 story confirmed.

**Its place in the paper:** the single memorable figure. Plot Ψ vs power, mark
blind, MEMLESS (γ*), the dither curve, the found-memory scatter, and the CORR
frontier as the upper-left envelope.

---

## F6 — Real vehicle data: is the constraint active, and does CORR help?

**What it asks:** on real CSU CAN innovations (90 stationary segments, recovered by
VARX), (a) is the received-autocovariance exposure actually a problem for the
memoryless rule, and (b) does the corrected rule fix it?

**(a) The constraint is strongly active — premise confirmed.** Under MEMLESS the
exposure functional is large and highly variable across segments (contiguous split,
the one that preserves lag structure):

| Γ̄ | MEMLESS max_rhat | distribution (interleave) |
|---|---|---|
| 0.02 | 0.208 | median 0.085, 75th pct 0.149, max **0.828** |
| 0.20 | 0.204 | — |
| 0.50 | 0.238 | — |

max_rhat ≈ 0.20 is ~20× a nominal white level. Real vehicle residuals (pooled
excess kurtosis ≈ 28) carry exactly the sign–magnitude dependence the paper is
about. **This is F6's main job and it succeeds.**

**(b) A fixed CORR config does *not* transfer — an honest negative result.** With
the single configuration L=10, f=0.3 (tuned on synthetic N2), exposure does **not**
drop; max_rhat rises slightly (0.21 → 0.24 contiguous), reduced in only 34% of
cells. Diagnosis from the data: CORR *does* reduce lag-1 in a plurality of cases
(interleave r1 reduced in 42% of cells, mean |r1| 0.0095→0.0012), but the
max-over-lags functional goes up — **the block shifts correlation to other lags
when L is not matched to the real data's multi-lag dependence.** On synthetic N2
the dependence is lag-1-dominated, so a block cleans it; the VARX-recovered real
innovations have richer structure, so a fixed single-lag-oriented block is
insufficient.

Two methodology notes this exposes:
- **The interleave split decimates the series** (its lag-1 = the original's lag-2),
  so interleave exposure numbers are not what a monitor sees on the live stream;
  the contiguous numbers are physically meaningful but break calibration
  exchangeability. Neither split is clean — for the paper, measure exposure on the
  contiguous received stream and calibrate on a held-out contiguous block.
- **Small-budget rate control degrades on real data** (realized 0.035 vs 0.02
  target at Γ̄=0.02) — F3's few-effective-exceedances warning biting in practice.

**Its place in the paper:** F6 validates the premise on real data (essential) and
honestly bounds the method: the correction is not plug-and-play; it must be tuned
to the deployment's lag structure. That is a limitation to state, and it directly
motivates (i) constraining *multiple* lags jointly (the |r_ℓ|≤δ, ℓ=1..L machinery,
not a single block length) and (ii) per-deployment tuning of L (or the dual
multiplier), rather than a transplanted configuration.

---

## What to do next (concrete, from these results)

1. **Rerun F6 with tuning, not transplant.** Sweep (L, f) per segment (or per
   pooled record) and report the *best achievable* exposure, the way F4 does for
   synthetic data. Expect the multi-lag structure to need larger L or a per-lag
   correction; report where it can and cannot reach nominal. This turns the
   negative into "the method works on real data once tuned to it."
2. **Report exposure on the contiguous received stream only**, and note the
   decimation artifact of interleaving explicitly.
3. **Fix the paper's frontier axis to detector power** (F5), and use the functional
   only for the theory figure — with the zero-crossing subtlety stated so a referee
   doesn't read the Γ̄≈0.10 functional as a loss.
4. **Quote F4's sweet-spot table as the main result**, the N1 control as the
   corollary, and F3's N_eff ≈ N/L with the L≤~10 ceiling as the cost.
5. **Consider dropping AR/MAGDIFF from the main text** (they don't reduce exposure)
   and keeping ENERGY/BLOCK only as the "diffuse rules kill damage" contrast —
   F2's role is motivation, and two curves make the point.

## One-line verdict per experiment
- **F1** ✓ dither is not a stealth dial (non-monotone, budget/law-dependent).
- **F2** ✓ found memory rules are dominated (keep damage *or* stealth, never both).
- **F3** ✓ guarantee survives, variance ~×√(N/N_eff), N_eff≈N/L, hard floor at
  small budget → L≤~10.
- **F4** ✓✓ corrected rule dominates at matched budget; needed only off-Gaussian.
- **F5** ✓ CORR owns the damage/stealth frontier by the operational metric.
- **F6** ◑ premise confirmed on real data (constraint active); fixed config does
  not transfer — tune per deployment.
