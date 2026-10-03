# F7 & F8 — Calibration repair and the baseline whiteness comparison
*Companion to `cps_companion_F7.py`, `cps_companion_F8.py` and their result CSVs.*

Both experiments are already run; the CSVs are attached. Nothing below is a
prediction — every number is from the attached files.

---

# PART A — F7: the calibration repair

## A.1 The structural question, answered

You asked me to check this claim:

> *Under hysteresis the firing indicator is not a threshold on an exchangeable
> scalar score — it depends on the run state. So even a correctly-applied
> conformal index on ε_hi would not deliver the ≤ Γ̄ guarantee for the realized
> rate, only for the start events.*

**It is true, and F7 confirms it in both directions.** The precise statement:

| Event | Is it a threshold on an exchangeable scalar? | Guarantee |
|---|---|---|
| **START** `{ \|z_k\| > ε_hi }` | **Yes** — `\|z_k\|` alone | `P(start) ≤ α` **exact** (Lemma 4 verbatim) |
| **FIRE** `γ_k` | **No** — depends on run state `n_{k−1}` | none; two weaker bounds survive |

The two surviving bounds:

- **Deterministic (pathwise):** each start licenses at most `L−1` continuations, so
  `rate ≤ L · start_rate`. **Zero violations in all 684 rows of F7.**
- **Renewal (asymptotic):** `rate ≈ start_rate × E[B]`, `E[B]` = mean run length —
  which is **law-dependent and not known a priori**. This is precisely why the
  guarantee cannot transfer distribution-free.

### The evidence (F7, N2, f = 0.3)

`start_rate_mean` vs the analytic conformal prediction `1 − j_N/(N+1)`:

| L | α | start_rate (measured) | conformal prediction | error |
|---|---|---|---|---|
| 2 | 0.0127 | 0.0124 | 0.0125 | −0.0001 |
| 4 | 0.0103 | 0.0100 | 0.0101 | −0.0001 |
| 6 | 0.0102 | 0.0098 | 0.0100 | −0.0002 |
| 10 | 0.0102 | 0.0098 | 0.0100 | −0.0002 |

Across all 228 `CONF_START` rows: mean bias **+0.00014 (N1)**, **−0.00005 (N2)** —
sub-1% relative. Under N1 the innovation is genuinely i.i.d., so `|z|` is
exchangeable and the guarantee is *exact*; under N2 heavy-tail clustering breaks
strict exchangeability and introduces a small two-sided wobble.

**Conclusion: the start-level guarantee is real and provable. The realized-rate
guarantee is not, and no amount of correct conformal application recovers it.**

## A.2 Three calibrators compared

`CALIBRATORS` in `cps_companion_F7.py`:

| | what it does | finite-sample guarantee |
|---|---|---|
| `BISECT` | your current code: bisect ε_hi so realized rate = Γ̄ | **none** |
| `CONF_NAIVE` | conformal index at level Γ̄ applied to `\|z\|` — Sec. IV read literally | starts only, at the *wrong* level |
| `CONF_START` | **the repair**: ε_hi always an order statistic; bisect the *level* α so realized rate = Γ̄ | **starts, exactly, at level α** |

Results (mean relative bias of the realized rate; `viol_frac` = fraction of the
200 calibration draws exceeding Γ̄):

| regime | calibrator | mean rel. bias | max rel. bias | mean viol. frac |
|---|---|---|---|---|
| N1 | BISECT | +0.016 | +0.415 | 0.523 |
| N1 | **CONF_NAIVE** | **+2.235** | **+30.9** | **0.973** |
| N1 | CONF_START | +0.016 | +0.415 | 0.520 |
| N2 | BISECT | +0.010 | +0.459 | 0.497 |
| N2 | **CONF_NAIVE** | **+2.116** | **+29.3** | **0.973** |
| N2 | CONF_START | +0.009 | +0.459 | 0.493 |

### Three things this settles

1. **`CONF_NAIVE` is catastrophic — and it is what Sec. IV literally claims to do.**
   It over-fires by exactly the mean run length (at L=4, Γ̄=0.02: rate 0.0407 vs
   0.0200, and `mean_run_len` = 2.28), and it violates the budget on **97.3% of
   calibration draws**. If a referee implements your Eq. (13) as written, this is
   what they get. This alone justifies rewriting the sentence.

2. **`CONF_START` costs nothing.** It matches `BISECT` on realized-rate accuracy
   to three decimals (+0.009 vs +0.010) *while adding* an exact, statable,
   distribution-free guarantee on the start events. **There is no trade-off — the
   repair is free.** Use it.

3. **`BISECT` is not wrong, it is just unjustified.** It controls the rate fine;
   it simply has no finite-sample guarantee attached, so the paper cannot claim one.

> ⚠️ **Note on my first attempt.** I initially implemented `CONF_START` as a fixed
> point on `α = Γ̄/B̂`. It **diverges** at L=50, f=0: once runs merge, the observed
> mean block length exceeds L (88 > 50), the map is non-monotone, and it
> oscillates — producing a realized rate of 0.638 against a budget of 0.02. The
> shipped version bisects on α instead, because the realized rate *is* monotone
> in α. If you write the fixed-point form into the paper, it will fail at large L.

## A.3 Can you draw the degradation conclusion?

> *"The finite-sample guarantee applies to the run-start events; for the realized
> rate it degrades as the effective exceedance count falls."*

**The first clause: yes, exactly, and F7 proves it (§A.1).**

**The second clause: yes, but state it as magnitude, not frequency.** Binning all
`CONF_START` rows by effective start-exceedances `α · N_eff`:

| α·N_eff | n | mean \|rel. bias\| | max \|rel. bias\| | viol. frac |
|---|---|---|---|---|
| **< 2** | 20 | **0.083** | **0.459** | 0.527 |
| 2–5 | 8 | 0.009 | 0.026 | 0.489 |
| 5–10 | 15 | 0.016 | 0.035 | 0.504 |
| 10–25 | 30 | 0.009 | 0.022 | 0.500 |
| 25–50 | 33 | 0.009 | 0.026 | 0.516 |
| **> 50** | 122 | **0.003** | 0.020 | 0.504 |

`corr(log α·N_eff, log |rel. bias|) = −0.628`.

**A 28× degradation in bias magnitude** from >50 effective exceedances down to <2.
The claim is supported.

**But the violation fraction stays flat at ≈0.50 across every bin.** That is the
correct behaviour of a median-unbiased conformal index and is *not* degradation.
So the honest wording is:

> *"The finite-sample guarantee applies to the run-start events. For the realized
> rate no such guarantee exists; its bias magnitude degrades as the effective
> start-exceedance count α·N_eff falls — by a factor of 28 between α·N_eff > 50
> and α·N_eff < 2 — while the violation frequency remains at the ≈1/2 expected of
> a median-unbiased index."*

Do **not** write "the guarantee degrades" — a guarantee either holds or does not.
What degrades is the accuracy of a quantity that never had one.

## A.4 What to change in the paper

**Proposition 4, replacement statement:**

> The split-conformal index applied to `|z_k|` at level α sets ε_hi as an order
> statistic of the nominal magnitude record. The start event `{|z_k| > ε_hi}` is a
> threshold on an exchangeable scalar score, so Lemma 4 applies to it verbatim:
> the expected rate of run starts is at most α at every horizon. The firing
> indicator of (18) is not such a threshold — it depends on the run state — and
> the guarantee does not transfer to the realized rate. Two bounds survive: the
> pathwise bound `Γ̂ ≤ L·α`, and the renewal relation `Γ̂ ≈ α·E[B]` with `E[B]` the
> mean run length, which is law-dependent. Choosing α so that (18) meets the
> budget on the calibration record therefore retains the distribution-free
> guarantee on starts while making the realized rate a plug-in quantity whose bias
> grows as the effective start-exceedance count α·N_eff falls (Table III).

Then **delete** "so Lemma 4 transfers verbatim" from Sec. IV, and **delete**
"the mean realized rate stays on target at every window length" from Sec. V.

---

# PART B — F8: the baseline whiteness comparison

## B.1 First — I over-claimed, in two ways

My original sentence was:

> *"A lag-ℓ autocorrelation or Ljung–Box monitor defeats all three of these
> attacks, at any budget, and under Gaussian noise."*

**Defect 1 — "at any budget" is a category error.** Li & Yang, Shang & Chen and
Ren et al. have **no budget**: they corrupt every transmission (rate 1.0). Budget
is *your* decision variable, not theirs. F8 confirms `fire_rate = 1.000` for all
three.

**Defect 2 — it depends on the monitor's lag count m.** Their correlation sits at
different places:

| attack | received correlation lives at | escapes m < τ? |
|---|---|---|
| Ren `z̃ₖ = H z̃ₖ₋τ + T zₖ` | lag τ, 2τ, … (AR(1) at lag τ) | partially |
| Li `z̃ₖ = T zₖ₋τ + L zₖ + bₖ` | lag τ **only** | **yes** |
| Shang `z̃ₖ = Σᵢ Tᵢ zₖ₋ᵢ + bₖ` | **all** lags ≤ τ (MA(τ)) | **no** |

Measured ACF of Ren's stream, lags 1–22 (N1, H = 0.819):
```
+0.005 +0.005 -0.000 +0.003 +0.007 +0.004 +0.000 +0.005 +0.005 +0.818
+0.005 +0.005 -0.001 +0.003 +0.006 +0.005 +0.001 +0.005 +0.005 +0.669  ...
```
Clean spikes at lags 10 and 20; essentially zero elsewhere. Exactly as designed.

**Use the corrected claim instead.** It is stronger *because* it is conditional:

> A Ljung–Box monitor with `m ≥ τ` detects all three with power 1.000 under
> Gaussian noise, where the sign flip is provably invisible at every budget. The
> attacker must know τ to place its correlation outside the χ² window — Ren et
> al.'s own Remark 7 concedes this — whereas the defender may simply choose
> `m ≥ τ`. The asymmetry favours the defender.

## B.2 The headline result (F8, N1 = Gaussian, W = 5000, τ = 10, FAR = 1%)

| attack | rate | tr(D) | acf₁ | acf_τ | χ² | LB m=1 | m=5 | **m=10** | m=20 | m=30 |
|---|---|---|---|---|---|---|---|---|---|---|
| NOMINAL (control) | 0 | 0 | .000 | .001 | .008 | .015 | .014 | .008 | .003 | .009 |
| **Li & Yang 2021** | 1.00 | 0.434 | .001 | **.485** | .027 | .040 | .113 | **1.000** | **1.000** | **1.000** |
| **Shang & Chen 2021** | 1.00 | 1.369 | **.917** | .080 | .197 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| **Ren et al. 2023** | 1.00 | 0.547 | .001 | **.824** | .127 | .297 | .758 | **1.000** | **1.000** | **1.000** |
| OURS corrected Γ̄=0.02 | 0.02 | 0.042 | .000 | .001 | .008 | .009 | .009 | **.009** | .010 | .015 |
| OURS corrected Γ̄=0.10 | 0.10 | 0.127 | .000 | .000 | .008 | .014 | .009 | **.005** | .006 | .015 |
| OURS corrected Γ̄=0.50 | 0.50 | 0.310 | .000 | .000 | .008 | .018 | .010 | **.009** | .009 | .017 |

Read the three things this shows:

1. **All three baselines pass the detector they designed against.** χ² power
   0.027 / 0.197 / 0.127 — low. Their certificates hold. *(Shang's 0.197 is not a
   failure of their theory; it reflects that the empirical χ² null is calibrated
   on windows of W = 5000, not on their assumed window τ = 10.)*
2. **All three fail the detector they did not constrain, at m ≥ τ, with power
   1.000.** Not 0.8 — saturated.
3. **Ours stays at nominal FAR at every budget from 0.02 to 0.50.** Powers
   0.005–0.021 against a 1% target. This is Prop. 1 in operation.

**The m-threshold is exactly as predicted:** Li goes 0.040 → 0.113 → **1.000** as m
crosses τ = 10. Shang is caught at m = 1 already (MA structure). Ren leaks
partially at m = 5 (0.758) — not from genuine lag ≤ 5 correlation, which is
≈0.005, but from the inflated variance of the sample ACF under its long-memory
structure; the clean mechanistic statement remains m ≥ τ.

## B.3 The result that pre-empts the obvious referee objection

*"You chose degenerate parameters to make detection easy."* The `sweep_*` rows
answer this. Sweeping Ren's H and Li's T over their whole admissible set (N1):

| H (Ren) | tr(D) | acf_τ | LB power m=10 |
|---|---|---|---|
| **0.00** | 0.354 | 0.001 | **0.008** ← nominal |
| 0.10 | 0.376 | 0.099 | **0.999** |
| 0.20 | 0.398 | 0.199 | 1.000 |
| 0.50 | 0.472 | 0.499 | 1.000 |
| 0.90 | 0.550 | 0.900 | 1.000 |

Li's sweep is identical in shape (T = 0.10 → 0.999).

> **A 10% historical weight already gives detection power 0.999.** The *only*
> point in either family at nominal power is `H = 0` / `T_τ = 0` — which is
> precisely Guo et al.'s memoryless attack. **To evade a whiteness monitor, these
> attacks must switch the historical data off entirely, i.e. abandon the exact
> contribution the papers are about.**

That is the sentence to put in the paper. It is not rhetoric — it is a measured
frontier with 8 points per family.

And the Pareto comparison is favourable: at `H = 0` (their only stealthy point)
they reach `tr(D) = 0.354` while corrupting **100%** of transmissions; your
corrected rule reaches `tr(D) = 0.310` — **88% of that** — while corrupting **50%**.

## B.4 🔴 A finding in F8 that you need to act on, independent of the baselines

Under **N2**, against **your own corrected rule**, lag-1 ACF and Ljung–Box
disagree sharply:

| Γ̄ | power acf₁ | LB m=5 | LB m=10 | LB m=20 |
|---|---|---|---|---|
| 0.02 | **0.014** | **0.331** | 0.321 | 0.169 |
| 0.05 | 0.028 | 0.144 | 0.141 | 0.071 |
| 0.20 | 0.256 | **0.487** | 0.421 | 0.267 |
| 0.30 | 0.221 | **0.424** | 0.369 | 0.212 |

Two observations:

- **Your `power_acf1 = 0.014` at Γ̄ = 0.02 reproduces Table I's 0.014 exactly**,
  from a fully independent reimplementation. Good — Table I is sound as stated.
- **But a Ljung–Box monitor pooling 5 lags detects the same schedule at 0.331.**
  The nominal control is well calibrated (N2 NOMINAL: LB m=5 = 0.013), so this is
  a real detection, roughly **24× the lag-1 figure**.

**Implication.** The abstract says the corrected rule returns "an
innovation-whiteness monitor to its nominal false-alarm rate." That is true for
**the lag-one monitor**. It is **not** true for Ljung–Box, which is the monitor
your own Sec. II-C names first and which `ljung1978measure` is cited for. A
referee who runs LB will find 0.33 where you claim nominal.

**Recommended fix — and it costs you very little:**
1. Add a Ljung–Box column to Table I alongside π. You already compute
   `power_ljungbox` in `F4_corrected_rule.csv`; it is simply not reported.
2. Change the abstract to *"returning a lag-one innovation-whiteness monitor to
   its nominal false-alarm rate, and reducing the power of a pooled Ljung–Box
   monitor by [factor]."*
3. This is also a natural place to state the defender-side finding: **pooling lags
   is the right defence**, which strengthens your Sec. VI conclusion rather than
   weakening the paper.

---

# PART C — How to run these

```bash
# fast smoke test (~3 s and ~16 s)
python3 cps_companion_F7.py --quick
python3 cps_companion_F8.py --quick

# full runs (~3.5 min and ~2 min); these produced the attached CSVs
python3 cps_companion_F7.py --out F7_conformal_calibration.csv
python3 cps_companion_F8.py --out F8_baseline_whiteness.csv

# sensitivity: does the m >= tau threshold move with tau?
python3 cps_companion_F8.py --tau 5  --out F8_tau5.csv
python3 cps_companion_F8.py --tau 20 --out F8_tau20.csv
```

`cps_companion_F8.py` imports the plant, simulator and schedulers from
`cps_companion_F7.py`, so keep them in the same directory. Both are self-contained
otherwise (no dependency on `cps_companion.py`). `numba` is used if present and
falls back to pure Python if not — install it, the full F7 run is ~40× faster with it.

**Plant validation** (both scripts, against Sec. V of the draft):

| quantity | script | paper |
|---|---|---|
| S | 0.0752 | 0.0752 |
| K | [0.335, 0.029] | [0.335, 0.029] |
| tr ℒ(A,KKᵀ) | 1.182 | 1.182 |
| κ_exc(z) N1 / N2 | 0.00 / 7.32 | 0.00 / 7.49 |

---

# PART D — Suggested new text

**New Prop. 4 / Table III** — see §A.4.

**New paragraph for Sec. V (or a new subsection), from F8:**

> *Comparison with historical-data attacks.* The strictly-stealthy attacks of
> [Li & Yang], [Shang & Chen] and [Ren et al.] impose whiteness on the received
> stream only within an assumed χ²-detector window of length τ, leaving the
> autocovariance free at lag τ and beyond; their transmitted streams are by
> construction MA or AR processes. Table [X] evaluates all three on the plant of
> Sec. V under N1, with each family's parameters chosen to maximize damage subject
> to its own stealth constraint. All three retain low power against the windowed
> χ² detector they target, and all three are detected with power 1.000 by a
> Ljung–Box monitor with m ≥ τ — under Gaussian noise, where the sign flip of this
> paper is invisible at every budget (power ≤ 0.021 for Γ̄ ∈ [0.02, 0.50]).
> Sweeping each family's historical weight shows the exposure is not an artefact
> of the damage-optimal parameter: a historical weight of 0.10 already yields
> power 0.999, and the only admissible point at nominal power is the one that
> switches the historical term off, recovering the memoryless attack of [Guo et
> al.]. The attacker must know τ to place its correlation outside the detection
> window — as [Ren et al., Rem. 7] concedes — whereas the defender need only
> choose m ≥ τ.
