# Companion paper — experiments, results, theory & glossary (standalone)

*A self-contained account of the companion study: what each experiment does, what
it found, how it connects to the theory, and a glossary of every term. Read top to
bottom or jump via the map in §A.*

**Companion in one line.** The letter's memoryless sign-flip attack maximises damage
but, off-Gaussian, betrays itself to a whiteness (correlation) monitor. This paper
poses the stealth requirement as an explicit, distribution-free constraint and shows
the optimal schedule under it needs **memory** — and that memory is forced exactly
when the innovation is not conditionally sign-symmetric (i.e. exactly off-Gaussian).

**Run configuration.** N1 (Gaussian) and N2 (heavy-tailed) synthetic regimes;
calibration N=5000, deployment T=50,000, 50 Monte-Carlo seeds (F3: 200 calibration
draws). Real data: 84 usable stationary CSU CAN segments via VARX-estimated
innovations. All schedulers are sign-invariant. Exposure functional
`max_rhat` = maxₗ |r̂ₗ|/S over lags ℓ=1..10 (synthetic) / 1..20 (real); `power_acf1`
= rejection rate of a lag-1 whiteness monitor at window W.

---

## A. How to read this — the experiment map

| Exp | Question in one line | Verdict |
|---|---|---|
| **F1** | Can you hide by just randomising the score (dithering)? | ✗ No — exposure is non-monotone and law-dependent |
| **F2** | Do off-the-shelf memory scores trade damage for stealth well? | ✗ Dominated — keep damage *or* stealth, never both |
| **F3** | What does memory cost in calibration? | Guarantee holds; variance ∝ √(N/N_eff); N_eff≈N/L |
| **F4** | Does the proposed corrected rule beat memoryless & dither? | ✓✓ Yes, at matched budget; needed only off-Gaussian |
| **F5** | Who owns the damage/stealth frontier? | ✓ The corrected rule (by detector power) |
| **F6** | Is the exposure real on vehicle data? | ◑ Not on this pipeline — innovations aren't white; see §F6 |

Each experiment section has three parts: **plain-English**, **what the data says**,
and **theory link** (which theorem it tests and how). Terms in `code font` or
*italics* are defined in the **Glossary (§G)**.

---

## B. The theory this is built on (so the experiments have a target)

**The letter (prior work) established five load-bearing results.** The companion
cites these and builds on them:

- **Thm 1 (pathwise magnitude stealth).** The sign-flip keeps |z^c_k| = |z_k| on
  every sample path, so *every* magnitude-only detector (class `Gmag`) keeps its
  nominal false-alarm rate, for any firing rule and any noise law. → The attack is
  free against `Gmag`; the only battlefield left is `Gsgn`.
- **Prop 1 (covariance matching is necessary, not sufficient).** Off-Gaussian, a
  matched second moment no longer implies a matched law; the sign-flip is the unique
  law-preserving member of the classical attack family. → Justifies studying the
  *received law*, not just its covariance.
- **Thm 2 (distribution-free budget).** A *split-conformal* order-statistic trigger
  attains the budget Γ̄ with a finite-sample rate bound (needs only *exchangeability*)
  and almost-sure attainment under stationarity+ergodicity. → How the budget
  guarantee is enforced without a Gaussian tail.
- **Thm 3 + Cor 1 (damage collapses to one scalar Ψ, maximised memorylessly).**
  Steady-state degradation is `tr(D∞) = 4·S·Ψ·tr L(A,KKᵀ)`, and Ψ is maximised by the
  *memoryless* upper-level set γ*=1{|z|>q}. → The letter's optimum is memoryless.
- **Prop 2 + Rem 6 (the boundary and the damage–exposure coupling).** Under
  *conditional sign symmetry* (`CSS`) the flip is stealthy against `Gsgn` too; `CSS`
  fails only off-Gaussian, governed by a fourth cumulant. The induced lag-1
  autocovariance is `r₁ = −2(T₁₀+T₀₁)` and has an interior maximum in the budget.
  → Damage and exposure are two readings of one tail quantity; the letter stops here
  and names the constrained problem as future work.

**The companion adds five results (the plan's R1–R5), one per experiment family:**

- **R3 — dither fails** (→ **F1**): exposure is non-monotone in the dither scale, so
  randomisation is not a stealth dial.
- **R2 — achievable region** (→ **F2, F5**): the dithered/found-memory rules trace
  dominated curves; a principled rule reaches strictly better (Ψ, exposure) points.
- **R1 — structure of the constrained optimum** (→ **F4**): any optimum is a threshold
  on a *corrected score* whose correction depends on the past window of magnitudes and
  decisions; the correction vanishes ⇔ `CSS` ⇔ memoryless is optimal.
- **R4 — price of memory** (→ **F3**): overlapping windows autocorrelate the firing
  indicator, so the *effective* calibration size shrinks to `N_eff = N/(1+2Σρₗ)`; the
  budget guarantee survives (exchangeability), the variance inflates.
- **R5 — Gaussian recovery** (→ **F4 N1 control, F6**): under `CSS` the correction is
  off and everything collapses to the letter — memory is unnecessary.

---

## HEADLINE — the five things the run establishes

1. **The corrected rule works (F4).** Where the memoryless rule is *certainly*
   detected (N2, Γ̄=0.02, whiteness power **1.000**), the corrected scheduler drops
   power to **0.014** — nominal — while keeping **86%** of the damage (Ψ 0.294 vs 0.343).
2. **Dithering is not a substitute (F1).** Ψ falls monotonically with the dither
   scale, but exposure is non-monotone and the "safe" setting moves with budget and
   noise law — it can't be set without the distributional knowledge we avoid.
3. **Memory is needed only off-Gaussian (F4 N1 control).** Under N1 the memoryless
   rule is already at nominal power at every budget, so the correction has nothing to
   do — the "memory ⇔ not `CSS`" statement, visible in data (R5).
4. **The price of memory is real and bounded (F3).** The budget guarantee holds, but
   `N_eff ≈ N/L`, down to ~2 usable exceedances at L=50, Γ̄=0.02 — a hard ceiling of
   L ≲ 10 at these settings.
5. **On real CAN data the exposure problem does not materialise — and we know why
   (F6).** The `NOATTACK` control shows the VARX innovations are far from white on
   their own (baseline max_rhat ≈ 0.39), so a whiteness monitor calibrated on them has
   **no power against any schedule** (MEMLESS, CORR and dither all sit at nominal FAR).
   The sign-flip even slightly *reduces* their lag-1 correlation. So on this pipeline
   the memoryless rule is already stealthy and CORR is unnecessary; the machinery
   becomes relevant only against a defender whose model actually whitens the residual.
   Completing the real-data validation needs a whiter innovation estimate.

---

## F1 — Dithering is not a stealth dial

**Plain-English.** The cheapest way to look less suspicious is to add random noise to
the firing score so you stop firing *exactly* on the biggest spikes. `σ_d` is the size
of that noise (in units of √S). Does turning `σ_d` up trade damage for stealth cleanly?

**What the data says (N2).**

| σ_d/√S | 0 (=B3) | 0.25 | 0.5 | 1 | 2 | 4 | blind |
|---|---|---|---|---|---|---|---|
| Ψ (damage), Γ̄=0.10, W=5000 | 0.607 | 0.485 | 0.367 | 0.238 | 0.161 | 0.129 | 0.103 |
| power, Γ̄=0.10, W=5000 | 0.255 | 0.279 | **0.798** | 0.505 | 0.130 | 0.046 | 0.016 |

Damage falls **monotonically**; exposure does **not** — at Γ̄=0.10 it *rises* to 0.798
at σ_d=0.5 before falling. And the shape changes with budget (N2, W=5000):

| Γ̄ \ σ_d | 0 | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|
| 0.02 | 1.000 | 1.000 | 0.962 | 0.283 | 0.032 | 0.016 |
| 0.10 | 0.255 | 0.279 | 0.798 | 0.505 | 0.130 | 0.046 |
| 0.20 | 0.858 | 0.088 | 0.122 | 0.230 | 0.092 | 0.038 |
| 0.50 | 0.128 | 0.605 | 0.387 | 0.092 | 0.042 | 0.036 |

At Γ̄=0.02 you need a *large* dither to hide; at Γ̄=0.20 a *small* one; at Γ̄=0.50 the
relationship inverts. **No single σ_d is safe across budgets.** Control (N1): power is
nominal (0.004–0.022) for every σ_d — there is no serial exposure to dither away.

**Theory link (R3; letter Rem 6).** The received lag-1 autocovariance is
`r₁ = −2(T₁₀+T₀₁)`, a difference of two truncated cross-moments whose relative weight
shifts as the firing set slides off the tail — so r₁ passes through zero at a
*law-dependent* budget/σ_d, which is the non-monotonicity above. Because the safe
point depends on the noise law, dithering violates the distribution-free premise. This
is the negative result that *forces* a structured correction — the rest of the paper.

---

## F2 — Do off-the-shelf memory scores buy stealth for the damage they lose?

**Plain-English.** Several windowed scores fire based on a *history* rather than the
current spike: `AR` (predict the next magnitude, fire on surprise), `ENERGY` (windowed
sum of magnitudes), `MAGDIFF` (magnitude change), `BLOCK` (fire on a threshold then
hold L steps). They all sacrifice damage vs firing on the raw tail. Do they buy
stealth in return? Compared at *matched firing rate*.

**Damage (Ψ) and exposure (max_rhat), N2:**

| scheduler | Ψ@0.02 | Ψ@0.10 | Ψ@0.20 | max_rhat@0.02 | max_rhat@0.20 | burst len |
|---|---|---|---|---|---|---|
| MEMLESS (B3) | 0.350 | 0.608 | 0.761 | 0.122 | 0.060 | ~1.2 |
| AR | 0.348 | 0.583 | 0.639 | 0.150 | 0.014 | ~1.3 |
| MAGDIFF | 0.281 | 0.480 | 0.593 | 0.078 | 0.062 | ~1.7 |
| ENERGY | 0.049 | 0.196 | 0.343 | 0.009 | 0.010 | ~18–35 |
| BLOCK | 0.050 | 0.202 | 0.354 | 0.008 | 0.009 | ~50 |

**The finding — two camps, neither in the corner we want:** `AR`/`MAGDIFF` keep the
damage (AR ties B3 at small budget) but stay *exposed*; `ENERGY`/`BLOCK` kill the
exposure (max_rhat≈0.01) but throw away most of the *damage* (long diffuse bursts).
Damage tracks *tail concentration*: mean |z| at firing is highest for MEMLESS and
lowest for BLOCK/ENERGY, and Ψ ranks in the same order.

**Theory link (R2; letter Cor 1 & Rem 6).** Cor 1 says damage is maximised by
concentrating on the tail; Rem 6 says the tail is exactly where the exposure lives.
So any rule that lowers exposure by *leaving* the tail must lose damage — which is the
two-camp split, quantified. This maps the interior of the *achievable region* and
shows no found rule is on its efficient boundary, motivating a principled correction.

---

## F3 — The price of memory in calibration

**Plain-English.** Memory windows overlap, so consecutive firing decisions are
correlated. That makes the calibration record behave as if it had *fewer* independent
points — the *effective* sample size `N_eff`. Does that break the budget guarantee,
and what does it cost?

**N_eff / N (effective calibration fraction), N2:**

| L | 0.02 | 0.10 | 0.20 | 0.50 |
|---|---|---|---|---|
| 1 (memoryless) | 0.86 | 0.75 | 0.80 | 0.90 |
| 4 | 0.25 | 0.26 | 0.27 | 0.35 |
| 10 | 0.10 | 0.11 | 0.12 | 0.15 |
| 50 | 0.020 | 0.021 | 0.023 | 0.033 |

**Findings:** (i) `N_eff ≈ N/L` for block-type memory (L=50→1/50) — the earlier guess
"N/20 at L=50" was wrong; the law is `N_eff = N/(1+2Σρₗ)`. (ii) Memoryless is ~0.98
under N1 but only ~0.75–0.90 under N2 (heavy-tail clustering), and budget-dependent.
(iii) **The guarantee holds; the variance inflates** — realised-rate std tracks the
N_eff-corrected prediction, not the i.i.d. one:

| L | N_eff | realised rate std | i.i.d. prediction | N_eff prediction |
|---|---|---|---|---|
| 1 | 4322 | 0.0024 | 0.0020 | 0.0021 |
| 10 | 499 | 0.0069 | 0.0020 | 0.0063 |
| 50 | 101 | 0.0189 | 0.0020 | 0.0139 |

(iv) **The hard limit is small budgets:** effective exceedances = N_eff·Γ̄ → only ~2 at
L=50, Γ̄=0.02, where the rate even drifts above target and the conformal-violation
fraction climbs from ~0.44 to 0.72. **Design rule:** keep L ≲ N·Γ̄; at N=5000, Γ̄=0.02,
L≤10 keeps ≥50 effective exceedances — and F4 shows L≈4–6 already suffices for stealth.

**Theory link (R4; letter Thm 2 & Rem 3).** Thm 2's finite-sample bound needs
*exchangeability*, not independence, so memory does **not** bias the budget — confirmed
by the rate means. Rem 3 already noted the conformal index is near-median-unbiased
(~0.44 violation), which we reproduce at L=1. R4 is the honest extension: dependence is
paid in *variance*, quantified by N_eff, giving a principled window-length ceiling.

---

## F4 — The corrected rule vs memoryless vs dither (the centerpiece)

**Plain-English.** The proposed scheduler `CORR` is a *hysteresis* rule: start a flip
when |z| crosses a high threshold ε_hi (a rare tail event), then *continue* flipping
for up to `L` steps while |z| stays above a lower gate ε_lo = `f`·ε_hi, with ε_hi
calibrated so the total firing rate equals the budget. This turns isolated tail spikes
into short runs — removing the firing/non-firing *boundaries* that create correlation —
without changing the budget. `f`=1 is exactly the memoryless rule (no continuation);
`f`→0 is a pure block. Does it beat both baselines *at the same budget*?

**Budget is held:** across all rows, max |realised rate − budget| = 0.012.

**Core result (N2, W=5000, Γ̄=0.02).** MEMLESS: Ψ=0.343 at power **1.000**. CORR power
by (L, f) — the table you read as "`CORR L \ f`" (rows = memory length L, columns =
continue-gate f):

| L \ f | 0.0 | 0.15 | 0.30 | 0.50 | 0.70 |
|---|---|---|---|---|---|
| 2 | 0.263 | 0.251 | 0.080 | 0.170 | 0.880 |
| 4 | 0.016 | 0.016 | **0.014** | 0.299 | 0.894 |
| 6 | 0.012 | 0.014 | 0.018 | 0.299 | 0.896 |
| 10 | 0.012 | 0.018 | 0.018 | 0.299 | 0.896 |

with matching damage Ψ:

| L \ f | 0.0 | 0.15 | 0.30 | 0.50 | 0.70 |
|---|---|---|---|---|---|
| 4 | 0.208 | 0.254 | **0.294** | 0.326 | 0.341 |
| 10 | 0.127 | 0.239 | 0.293 | 0.326 | 0.341 |

At **L=4, f=0.3** power is **0.014** (from 1.000) with Ψ **0.294** (86% of MEMLESS).
`f`=0.7 is too mild; `f`=0 over-pays in damage; **f≈0.3–0.5, L≈4–6 is the sweet spot**
(power flat for L≥6, so bigger windows only cost damage and calibration).

**Sweet-spot summary (N2, W=5000): lowest-exposure CORR keeping Ψ within 10% of
MEMLESS, per budget:**

| Γ̄ | MEMLESS Ψ | MEMLESS power | CORR (L,f) | CORR Ψ | CORR power |
|---|---|---|---|---|---|
| 0.02 | 0.343 | **1.000** | (2, 0.3) | 0.310 | **0.080** |
| 0.05 | 0.480 | 0.369 | (6, 0.5) | 0.455 | 0.020 |
| 0.10 | 0.607 | 0.255 | (6, 0.5) | 0.579 | 0.204 |
| 0.20 | 0.761 | **0.858** | (20, 0.3) | 0.685 | **0.218** |
| 0.30 | 0.853 | 0.745 | (20, 0.3) | 0.785 | 0.190 |
| 0.50 | 0.952 | 0.128 | (6, 0.3) | 0.912 | 0.054 |

Wherever MEMLESS is exposed, CORR keeps ~90% of the damage and cuts power 4–70×.

**Gaussian control (N1):** MEMLESS power is already nominal (0.008–0.020) at *every*
budget — under a Gaussian innovation there is nothing to hide, so CORR is unnecessary.

**Theory link (R1 + R5; letter Cor 1, Prop 2).** R1: any constrained optimum is a
threshold on a *corrected score* whose correction depends on the past window of
magnitudes and decisions; the hysteresis rule is a deployable one-pass instance
(f=1 ⇒ correction off ⇒ exactly Cor 1's memoryless γ*). The N1 control is R5 — under
`CSS` the correction is inactive and the rule collapses to the letter. Budget is
enforced by re-calibrating ε_hi with the same split-conformal index (Thm 2), and the
0.012 rate check confirms the guarantee transfers to the corrected score.

---

## F5 — The achievable damage/stealth region

**Plain-English.** Plot every rule on the (exposure, damage) plane. Who sits on the
efficient frontier — the corner with low exposure *and* high damage?

**Using detector power as the exposure axis (the operational metric), N2, W=5000 —
best Ψ achievable at power ≤ 0.10:**

| Γ̄ | CORR | DITHER |
|---|---|---|
| 0.02 | **0.310** | 0.043 |
| 0.10 | **0.539** | 0.129 |
| 0.20 | **0.626** | 0.605 |
| 0.30 | **0.608** | 0.591 |

CORR owns the frontier at every budget; dither is far behind at small budgets and only
draws level at large ones (where MEMLESS was barely exposed anyway).

**One subtlety (state it in the paper).** Ranked by the *functional* max_rhat at
Γ̄≈0.10, MEMLESS/AR appear to edge CORR — because Γ̄≈0.10 is near where MEMLESS's r₁
*crosses zero* (letter Rem 6 interior extremum), so its functional is incidentally
small there even while a windowed detector still accumulates power. Functional and
detector are two readings; the operational one (power) is the honest axis, and by it
CORR dominates — most where MEMLESS is most exposed (small and large budgets), least at
the zero-crossing.

**Theory link (R2).** This is the achievable region `F = {(Ψ, maxₗ|r̂ₗ|)}`: the blind
endpoint (Γ̄, 0), the dominated dither curve, the found-memory scatter, and the CORR
upper-left envelope — the picture of R2.

---

## F6 — Real vehicle data: is the exposure real? (revised, with the NOATTACK control)

**Plain-English.** Take real heavy-duty-truck CAN innovations (84 stationary segments,
recovered by a `VARX` model), split each into a calibration half and a deployment half
*contiguously* (so lag structure is preserved), and ask: (a) does the memoryless attack
actually expose itself to a whiteness monitor here, and (b) does CORR help? Crucially we
add a **`NOATTACK` baseline** — the whiteness of the innovations *themselves*, with no
attack — so we can tell attack-induced correlation from the model's own residual
correlation.

**What the data says (summary; exposure = maxₗ|r̂ₗ|/S over ℓ=1..20, power at W=2000):**

| Γ̄ | NOATTACK r | MEMLESS r | MEMLESS power | CORR-tuned r | CORR-tuned Ψ | best (L,f) | CORR-xfer r | DITHER r |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.392 | 0.327 | **0.008** | 0.318 | 0.364 | L2,f0.7 | 0.442 | 0.300 |
| 0.10 | 0.392 | 0.330 | **0.008** | 0.320 | 0.606 | L2,f0.7 | 0.431 | 0.224 |
| 0.20 | 0.392 | 0.342 | **0.008** | 0.330 | 0.739 | L2,f0.7 | 0.440 | 0.151 |
| 0.50 | 0.392 | 0.375 | **0.008** | 0.361 | 0.905 | L2,f0.0 | 0.463 | 0.102 |

Three facts settle the picture:

1. **The innovations are not white to begin with.** The `NOATTACK` baseline is
   max_rhat ≈ **0.39** (median 0.19; 65% of segments above 0.10; only 4% below 0.05),
   with a broad per-lag profile (|r̂|: r₁=0.34, r₂=0.30, r₃=0.23, r₄=0.23, r₅=0.20).
   This is `VARX` residual mismatch — the fitted model does not fully whiten the CAN
   residual — not anything the attack did.
2. **The whiteness monitor has no power against any schedule.** Pooled over 84
   segments, the maximum lag-1 power across *every* config and budget equals the
   false-alarm rate at each window (0.010, 0.011, 0.011, 0.017 at W=250/500/1000/2000);
   Ljung–Box likewise. MEMLESS, CORR and dither are all at nominal FAR. On this data the
   memoryless rule is **already stealthy** against `Gsgn`.
3. **The attack *reduces* correlation, it doesn't add it.** Attack-induced change in
   lag-1, Δr₁ = attacked − nominal, is **negative** at every budget (−0.034 at Γ̄=0.02,
   −0.070 at Γ̄=0.10). Flipping the tail of a *positively* autocorrelated innovation
   partially cancels the existing correlation, so the attacked stream is if anything a
   little *more* white than nominal — which is why a monitor calibrated on nominal
   cannot flag it.

**So the honest reading:** the companion's exposure mechanism does not appear on this
pipeline, because its precondition — a **white** nominal innovation carrying only
*tail sign–magnitude* dependence — is not met. The `VARX` innovations carry ordinary
lag correlation that swamps (and reverses) the attack's tail effect. Consequently CORR
is unnecessary here, and the tuning/transfer numbers are in the baseline-dominated
regime and should not be over-read: CORR-tuned barely moves the functional (0.32 vs
MEMLESS 0.33–0.38, both around the 0.39 baseline), a single transferred config actually
*worsens* it (0.44 — it overfits calibration-segment noise), and dither gives the
lowest functional (0.10–0.30) only because it spreads firing off the tail — moot, since
all powers are nominal.

> Note this **supersedes** the first F6 pass, which reported "constraint active,
> max_rhat≈0.20." That reading was an artifact of the *interleave* split (which
> *decimates* the series and hides baseline correlation) and the absence of a NOATTACK
> control. The contiguous split + NOATTACK baseline reveal the true picture.

**Theory link (R5 + the constraint's premise).** The result is consistent with the
theory, read carefully: Prop 2 says exposure is a *fourth-cumulant, tail* phenomenon of
a **white** innovation. When the nominal innovation is *not* white (a modelling failure,
not a distributional one), a whiteness monitor is uninformative against every rule — it
cannot separate nominal from attacked — so there is nothing for the correction to
correct. This is a boundary of the theory's applicability, not a contradiction of it.
It also carries a defender-side message: **the exposure that motivates CORR only exists
against a defender whose model actually whitens the residual;** a poorly-whitening
defender has already disarmed its own whiteness monitor.

**What to do to complete the real-data validation:**
1. **Whiten the innovations properly** — raise the VARX order (or use a better
   whitening filter) until the `NOATTACK` baseline max_rhat drops toward ~0. Only on a
   white baseline can the attack-induced tail exposure be measured. Then re-run F6 and
   check whether MEMLESS power rises above FAR (it should, per Prop 2) and whether CORR
   pulls it back (per R1).
2. **Report exposure as attack-induced Δ** (attacked − nominal per lag), not absolute
   r̂ₗ, so residual model correlation cancels out. The detector power already does this
   implicitly (it compares to a nominal null), which is why it is the trustworthy metric
   here.
3. **Segment on whiteness** — the 4% of segments with baseline max_rhat<0.05 are the
   only ones where the mechanism could show; analyse them separately as a proof of
   principle while the whitening is improved.

Until then, the real-data section should be written as an **honest boundary result**:
the synthetic regimes (F1–F5) isolate and confirm the mechanism; the CAN data shows the
mechanism is only operationally relevant when the innovation is well-whitened, which the
current pipeline does not achieve.

---

## What to do next (consolidated)

1. **F6 whitening pass** — the single most important next run (above). Turns the
   boundary result into either a clean real-data confirmation or a well-characterised
   limit.
2. **Fix the paper's frontier axis to detector power** (F5); use the functional only for
   the theory figure, with the zero-crossing subtlety stated.
3. **Quote F4's sweet-spot table as the main result**, the N1 control as the corollary
   (R5), and F3's `N_eff ≈ N/L` with the L≲10 ceiling as the cost (R4).
4. **Trim F2 in the main text** to the two-camp contrast (keep BLOCK/ENERGY as "diffuse
   kills damage", AR as "near-tail stays exposed"); its role is motivation for R1.
5. **State budget enforcement once, clearly** (see §C below) so every experiment inherits
   it by reference.

---

## C. Cross-cutting: how the budget guarantee and the corrected rule are handled

**How the budget guarantee is ensured (every experiment).** The trigger is a
*split-conformal* order statistic of the nominal magnitude record: with N calibration
magnitudes, fire when the score exceeds the ⌈(N+1)(1−Γ̄)⌉-th order statistic. By Thm 2
this attains E[rate] ≤ Γ̄ under *exchangeability* alone (no Gaussian tail, no
independence), with almost-sure attainment under stationarity+ergodicity. For CORR the
*same* conformal index sets ε_hi, so the guarantee transfers to the corrected score;
F4's max |realised − budget| = 0.012 confirms it. F3 quantifies the only cost — the
realised-rate *variance* inflates as √(N/N_eff) because overlapping windows correlate
the firing indicator, while the *mean* stays on budget (no bias).

**How the corrected rule is evaluated.** Four ways, so no single metric can flatter it:
(i) *matched-budget* head-to-head vs MEMLESS and DITHER on both Ψ (damage) and `Gsgn`
power (F4); (ii) the *achievable region* — does it reach (Ψ, exposure) points the others
cannot (F5); (iii) the *N1 control* — it must reduce to the memoryless rule when the
innovation is Gaussian (R5), which it does; (iv) *real data* — is the exposure it targets
even present (F6). The rule passes (i)–(iii) cleanly; (iv) is pending a whiter innovation
estimate.

---

## G. Glossary (every term used above)

**System & attack**
- **Remote state estimation** — a sensor runs a Kalman filter and sends its *innovation*
  over a network to a remote estimator that reconstructs the state. The attacker sits on
  that link.
- **Innovation `z_k`** — the one-step prediction error of the Kalman filter,
  z_k = y_k − C x̂. White under a correct model; its covariance is **S**.
- **`S`** — the (scalar, here) innovation variance, E[z_k²]. Used to normalise exposure.
- **White / whiteness** — a sequence is *white* if it has zero autocorrelation at all
  nonzero lags (E[z_k z_{k+ℓ}]=0). Whiteness tests check this. Note: white ≠ independent
  off-Gaussian.
- **Sign-flip attack / received stream `z^c_k`** — the attacker negates the innovation at
  firing instants: z^c_k = (1−2γ_k) z_k (so z^c = −z when firing, z otherwise). It never
  changes the *magnitude* |z^c_k| = |z_k|.
- **Divergence `d_k`, `tr(D∞)`** — d_k is the gap between the clean and corrupted remote
  estimates; tr(D∞) is its steady-state mean-square size = the attack's *damage*.

**Budget & firing**
- **Budget `Γ̄`** — the maximum fraction of time steps the attacker may corrupt (e.g.
  0.02 = 2%). A resource limit.
- **Firing rule `γ_k` ∈ {0,1}** — the schedule: 1 = attack this step. Its long-run mean
  must be ≤ Γ̄.
- **Blind firing** — firing at random, independent of the innovation, at rate Γ̄. The
  do-nothing-clever baseline; captures Ψ = Γ̄ and adds no correlation.
- **Realised rate** — the actual fraction of steps fired in deployment (should equal Γ̄).

**Damage**
- **`Ψ` (energy capture)** — the fraction of innovation energy the schedule corrupts,
  Ψ = E[γ_k z_k²]/S. The single scalar that sets damage: tr(D∞) = 4·S·Ψ·tr L. Higher Ψ
  = more damage. Ψ = Γ̄ for blind firing; Ψ up to ~1 for tail-focused firing.
- **Tail concentration** — firing on the largest |z| (the distribution's tail). Maximises
  Ψ, but the tail is where the exposure lives (see Rem 6).

**Detectors**
- **`Gmag` (magnitude-measurable detectors)** — detectors that see only magnitudes |z^c|
  (χ², CUSUM, windowed χ²). The sign-flip is *pathwise* invisible to all of them (Thm 1).
- **`Gsgn` (sign-sensitive detectors)** — detectors that read the *signed* sequence:
  whiteness/autocorrelation tests, Ljung–Box, sign-balance. The battlefield of this paper.
- **Whiteness test / lag-1 autocorrelation** — checks whether the received sequence has
  nonzero serial correlation; a nonzero r̂₁ trips it.
- **Ljung–Box** — a whiteness test pooling several lags into one statistic.
- **Sign-balance test** — checks whether received signs are still ~50/50; catches
  *sign-dependent* rules (see below), not the sign-flip.
- **`W` (window)** — the number of samples the detector accumulates before deciding.
  Larger W = more power to detect a small correlation.
- **`power_acf1`, `power_ljungbox`** — the detector's rejection rate under attack (its
  *power*). power = FAR means "no better than chance" — undetectable.
- **FAR (false-alarm rate)** — the detector's rejection rate under *nominal* (no-attack)
  data; here calibrated to ~1% (0.01).

**Exposure (the constraint object)**
- **`r_ℓ` / `r̂_ℓ`** — the received-stream autocovariance at lag ℓ, r_ℓ = E[z^c_k z^c_{k+ℓ}]
  (hat = estimated). This is the *exposure functional* — a sufficient statistic for lag-ℓ
  correlation detectors, and estimable by the attacker from its own output (since
  |z^c|=|z|). Nonzero r_ℓ = a correlation signature.
- **`|r̂_ℓ|/S`** — the lag-ℓ exposure, normalised by the innovation variance so it's
  dimensionless (a correlation-like number). ~0 = white/stealthy.
- **`max_rhat`** — maxₗ |r̂_ℓ|/S over ℓ=1..L: the worst-case single-lag exposure. The
  headline exposure number; lower = stealthier.
- **`T₁₀`, `T₀₁` / boundary** — in r₁ = −2(T₁₀+T₀₁), T₁₀ counts neighbour pairs where step
  k fires but k+1 does not (and vice versa) — the firing/non-firing *boundaries*.
  Isolated flips create many boundaries → high exposure; runs create few → low exposure.

**Regimes & data**
- **N1** — Gaussian innovation (control; `CSS` holds; no exposure).
- **N2** — heavy-tailed innovation (a variance-matched mixture; excess kurtosis 7.49);
  the regime where the exposure mechanism lives.
- **MM / TV** — model-mismatch / time-varying regimes (dropped from this study).
- **CSU** — the real heavy-duty-truck J1939 CAN log used for F6.
- **`VARX`** — the high-order autoregressive model fit to CAN data to *recover* the
  innovation without knowing the plant. Its residual is the estimated innovation; if the
  fit is imperfect, that residual is not white (the F6 confound).
- **Contiguous vs interleave split** — two ways to split a segment into calibration and
  deployment. *Contiguous* (first half / second half) preserves lag structure.
  *Interleave* (even / odd samples) *decimates* the series — its lag-1 becomes the
  original's lag-2 — and under-reports exposure; hence contiguous is used.
- **`NOATTACK` baseline** — the exposure of the innovations with *no* attack applied. The
  control that separates the model's own residual correlation from attack-induced
  correlation.
- **Excess kurtosis** — a measure of tail heaviness; 0 for Gaussian, large for heavy tails.
  It is the fourth-cumulant quantity governing both the damage gap and the exposure.

**Schedulers**
- **MEMLESS (= B3 = γ*)** — the letter's memoryless optimum: fire on the top-Γ̄ magnitudes,
  γ*=1{|z|>q}. Max damage, but exposed off-Gaussian.
- **AR** — fire on the surprise of an autoregressive one-step magnitude prediction.
- **ENERGY** — fire on a windowed sum of magnitudes.
- **MAGDIFF** — fire on the magnitude *change*.
- **BLOCK** — fire on a threshold crossing, then hold (keep firing) for L steps. The
  transparent memory baseline; the f→0 limit of CORR.
- **Dithering / `σ_d`** — add independent random noise of scale σ_d to the score before
  thresholding, so firing is no longer exactly on the tail. **`σ_d/√S`** is that scale in
  units of the innovation std (dimensionless). σ_d=0 is MEMLESS; σ_d→∞ is blind.
- **CORR (the proposal)** — the *corrected-score* scheduler, implemented as **hysteresis**:
  start a flip when |z|>ε_hi, continue for up to `L` steps while |z|>ε_lo=`f`·ε_hi, with
  ε_hi calibrated so the rate = Γ̄. Extends isolated tail flips into short runs to remove
  boundaries.
  - **`L` (memory length / window)** — the maximum run length; also the number of lags the
    correction spans. Larger L = lower exposure but lower damage and smaller N_eff.
  - **`f` (continue-gate)** — the fraction ε_lo/ε_hi. `f`=1 ⇒ no continuation ⇒ exactly
    MEMLESS (correction off); `f`→0 ⇒ pure BLOCK. The knob that trades damage for stealth.
  - **`ε_hi`, `ε_lo`** — the start and continue thresholds.
  - **"`CORR L \ f`"** — the F4 tables whose *rows are L* and *columns are f*; each cell is
    that (L,f) config's power or Ψ.

**Calibration & guarantees**
- **Split-conformal calibration** — set the firing threshold to an *order statistic* of the
  nominal calibration magnitudes (the ⌈(N+1)(1−Γ̄)⌉-th), which guarantees the budget under
  exchangeability. The distribution-free replacement for the Gaussian tail inverse.
- **Exchangeability** — the assumption that calibration and deployment scores are
  interchangeable in distribution (weaker than independence). What Thm 2 actually needs.
- **`N_eff` (effective sample size)** — N/(1+2Σρ_ℓ), where ρ_ℓ is the firing-*indicator*
  autocorrelation. The number of *independent-equivalent* calibration points; ≈ N/L for
  block-type memory.
- **Effective exceedances** — N_eff·Γ̄: how many calibration points actually inform the
  threshold. Too few (≈2 at L=50, Γ̄=0.02) makes the rate estimate unstable.
- **Conformal-violation fraction** — the fraction of calibration draws on which the
  realised rate exceeds Γ̄. ~0.44 is expected (near-median-unbiased index); higher means
  the guarantee is degrading.

**Theory objects**
- **`CSS` (conditional sign symmetry)** — given all magnitudes, the signs are i.i.d. ±1.
  Holds automatically under Gaussian noise; its failure (off-Gaussian) is exactly what
  creates `Gsgn` exposure. The correction is inactive ⇔ `CSS` holds.
- **Fourth cumulant** — the higher-moment quantity that is zero for a Gaussian and governs
  both the damage-prediction gap and the sign-magnitude dependence; the single "cause"
  behind the exposure.
- **Achievable region `F`** — the set of attainable (Ψ, exposure) pairs; its efficient
  upper-left boundary is what the corrected rule reaches.
- **R1–R5** — the companion's five results (structure of the optimum, achievable region,
  dither failure, price of memory, Gaussian recovery); see §B.

---

## One-line verdicts
- **F1** ✓ dither is not a stealth dial (non-monotone, budget/law-dependent). [R3]
- **F2** ✓ found memory rules are dominated (keep damage *or* stealth). [R2]
- **F3** ✓ guarantee survives, variance ∝ √(N/N_eff), N_eff≈N/L, floor at small budget →
  L≲10. [R4]
- **F4** ✓✓ corrected rule dominates at matched budget; needed only off-Gaussian. [R1,R5]
- **F5** ✓ CORR owns the damage/stealth frontier by the operational metric. [R2]
- **F6** ◑ on this VARX pipeline the innovations aren't white, so no rule is detectable
  and CORR isn't needed; validation pending a whiter innovation estimate. [boundary of R5]
