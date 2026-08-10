# VIScore: full specification

Everything needed to recompute the metric exactly, plus the constants' provenance and the
estimator's known failure modes. The implementation in `viscore/` is verified **bit-exact**
against the cached intermediates the paper's tables were computed from (worst relative
difference 4.4e-15 over `d_tol`, `veracity`, `tr Σ_z`, `m_emp`, `influence`, `sobriety` on
PushT / Reacher / Two-Room cells).

```
VIS = erf( (d_tol/2) / (√2 σ_roll) ) · min(m_emp / τ, 1) · (1 − p̂)
```

All three factors are computed from one shared object per checkpoint: the **latent bank**
`z ∈ R^{F×D}`, the encoder's output on a frozen probe, plus the predictor. No environment, no
planner, no success labels.

---

## 0. The probe

A probe is a fixed slice of **training** data at the model's own temporal cadence
(frameskip 5), sampled with `numpy.default_rng(0)` without replacement, 300 episodes:

| array | shape | meaning |
|---|---|---|
| `pixels` | `(F, 224, 224, 3)` uint8 | frames to encode |
| `act_blocks` | `(F, frameskip·A)` f32 | the action **block** executed at each row |
| `ep_ptr` | `(E+1,)` int64 | episode boundaries |
| `<state…>` | `(F, ·)` f32 | ground-truth coordinates named by the `TaskSpec` |

Two things this encodes:

* **A probe is part of the measurement.** Scores computed against different probes are not
  comparable, like accuracies on different test sets. Keep `seed=0`.
* **Frameskip 5 means concatenation, not repetition.** One probe row carries the five distinct
  per-step actions taken between two stored frames, so PushT's `act_blocks` is 10-dimensional
  (2 action dims × 5). This mirrors how the world model was trained.

Actions are normalized by the **probe's own** per-dimension mean/std (deviation from
full-dataset statistics measured below 1%).

---

## 1. Veracity

```
σ_roll² = NMSE_ol(H) · tr(Σ_z)
d_tol   = median ‖Δz‖ over probe pairs already one tolerance apart  (the "shell")
veracity = erf( (d_tol / 2) / (√2 σ_roll) )
```

**`NMSE_ol(H)`** — open-loop rollout error at horizon `H`, normalized by the latent bank's total
variance. Open loop: the model's own predictions are fed back as context; only the actions come
from data, which is the regime a planner rolls out in. Clips of length `history_size + h_max`
are cut at stride 3, NaN-bearing windows dropped, capped at 512 clips with `rng(0)`.

> `h_max` (15, or 8 on Two-Room's short episodes) sets the clip pool and is separate from the
> horizon `H` that is read off the curve. Keep the defaults: a different `h_max` changes which
> clips exist and therefore the value.

> The NMSE normalizer here is `tr(Σ_z)` with `ddof=1` (`numpy.cov`), while the released cache used
> `numpy.var` (`ddof=0`). At `F ≈ 5000` probe rows that is a 1.8e-4 relative difference in the
> variance and 9e-5 in `σ_roll` — the size of the residual seen in §6. Noted so the discrepancy is
> not mistaken for a porting error.

**`H`** is the diagnostic horizon in action blocks. The evaluation protocol's goal offset `d` is
in raw environment steps at frameskip 5, so `d = 25 / 50 / 75` ⇒ `H = 5 / 10 / 15`. Everything
published is `d = 25`, i.e. `H = 5`.

**`d_tol`** — the latent image of the environment's success tolerance. Among within-episode pairs
1–5 rows apart (capped at 40 000 with `rng(0)`), keep those whose **task-scored** physical
displacement lies in `[0.8, 1.2] × tolerance`; where the criterion ignores coordinates, further
keep the quartile whose *nuisance* displacement is smallest; take the median latent distance.
Fewer than 30 surviving pairs ⇒ `nan`, and the score abstains rather than extrapolating.

> Why measured, not extrapolated. The earlier recipe estimated a local latent-per-state slope on
> the nearest 15% of pairs and multiplied by the tolerance. Where the graded object barely moves
> while an arm sweeps, that extrapolates by orders of magnitude: on Cube, nearest-pairs had the
> block moving 0.115 mm (32% exactly static) while the arm swung 160 mm, giving `d_tol ≈ 4034`,
> about 300× the diameter of the entire latent cloud, and veracity pinned at 1.0000 on all 190
> cells — a model planning 56 points worse than a baseline still scored VIS 0.95. The two recipes
> agree within 10% where the extrapolation factor is below 1 (PushT 3.72 vs 3.54, Reacher 1.95 vs
> 1.90). **This is why `nuisance` in a `TaskSpec` is not optional** for arm-plus-object tasks.

**Fixed choices, not fitted constants.**

* The **half tolerance** in the numerator is a deliberately stricter prediction requirement than
  the grading ball, chosen to keep `erf` off its ceiling: at the full tolerance 10.4% of pool
  checkpoints read veracity exactly 1.0, at half 0%. It must **not** propagate into sobriety —
  see §3.
* **`erf`** is a bounded monotone map of the ratio, not a Gaussianity claim. Within-task rank
  statistics are invariant to it (the raw ratio and the Rayleigh form
  `1 − exp(−(d_tol/σ_roll)²/2)` give bit-identical within-task correlations); the **product**
  V·I·S is not, since a product mixes factor values and not only ranks — substituting either
  alternative moves the development-pool pooled correlation between +0.849 and +0.873 against the
  +0.864 reported. So the map is fixed rather than treated as free.
* Because only the **ratio** `d_tol/σ_roll` enters, an invertible rescaling of the latent leaves
  veracity unchanged — asserted in `tests/test_smoke.py`.

---

## 2. Influence

```
Ŝ    = Cov[ z_{t+H}(a* + δ) − z_{t+H}(a*) ]         K = 32 anchors, M = 64 perturbations
Ê    = Cov[ teacher-forced one-step residual ]      diag-shrunk γ = 0.1, + ridge
Ê_H  = σ_roll² · Ê / tr(Ê)
m_emp = ½ logdet(I + Ê_H⁻¹ Ŝ) = ½ Σ log(1 + λᵢ)     nats
influence = min(m_emp / τ, 1),   τ = 82
```

`Ŝ` is how much the *action space* can move the future; `Ê_H` is the noise the predictor itself
contributes at the same horizon. Their log-determinant ratio is the Gaussian channel capacity
from actions to futures — how many distinguishable outcomes the actions command. Perturbations
`δ` are drawn from the empirical action-block covariance (diag-shrunk γ = 0.1), so `Ŝ` measures
the spread of futures under actions the data actually contains, not under isotropic noise.

Under any invertible linear reparameterization `z ↦ Az` both covariances transform congruently,
so `m_emp` is unchanged — asserted in `tests/test_smoke.py`.

**The horizon-consistent floor `Ê_H`** is required for dimensional consistency: `Ŝ` is an
`H`-step quantity while `Ê` is one-step, and using raw `Ê` puts the floor far below the error the
rollout accumulates, inflating every `λᵢ`. Matching the trace fixes the magnitude and leaves the
directions untouched. **Stated assumption, not verified:** that the *directions* the predictor
errs in are approximately horizon-stable. If they rotate with horizon, `Ê_H` misallocates the
floor across eigendirections and `m_emp` is biased in a way these experiments cannot detect.

**The cap `τ = 82` nats.** Raw capacity stops predicting success past a task-dependent knee — 129,
65 and 59 nats on PushT / Reacher / Two-Room, with the correlation falling from +0.82/+0.95/+0.82
below to +0.31/+0.38/+0.44 above. Converting capacity into a *sufficiency* condition is what
makes readings transfer, since nats do not mean the same thing on two tasks. `τ` was selected
once by run-level two-fold cross-validation on the development tasks and then frozen.

* Not delicate: calibration error 8.1 / 8.5 / 8.8 / 8.9 points at τ = 55 / 82 / 110 / 129 while
  the worst within-task correlation stays at +0.69…+0.70.
* Paid for: 86% of PushT development checkpoints sit above τ, so the factor cannot separate them,
  and it reads exactly 1.0 on every Cube checkpoint (median 236 nats) — one third of the product
  carrying no information. `VISFactors.saturated` reports this instead of hiding it.
* What the cap buys beyond scale alignment: immunity to a specific failure mode. On checkpoints
  trained far past their best epoch raw capacity keeps drifting upward (+34% over sixteen further
  epochs) while success declines; an uncapped factor rewards that drift, the capped one does not
  move at all.

A logistic softening `σ((m_emp − t)/s)` was tried: better on development (12.1 → 7.4 points) and
it survived resampling, but its constants had seen every label set, and on the pre-registered
held-out evaluation the original hard cap won (4.0 vs 5.1 points). The hard cap is what ships.

---

## 3. Sobriety

```
p̂ = (1/K) Σ_k 1[ J(a*_k) > J(a^cem_k) ],   K = 64 anchors
sobriety = 1 − p̂
```

At each expert anchor the goal is the future the **recorded** action reached, so the expert's own
action is a known-good plan whose imagined cost the search must beat.
`J(a) = ‖rollout(z_ctx, a)_H − z_goal‖²`, the same last-step cost the deployed planner minimizes.
A mini-CEM — 128 candidates, 4 iterations, top-16 elites, initialized at the expert block, in
normalized action space, no clamp to the action box — searches for something cheaper. Every
anchor it wins is an exploitable hole in the landscape. Total probe cost: 64 × 512 rollouts.

Only the **sign** of each gap enters, which is what makes the probe cheap and stable.

**A rate, deliberately — not a magnitude.** The earlier form
`1/(1 + max(median_k ΔJ_k, 0)·tr(Σ_z)/d_tol²)` is not implemented and should not be reintroduced:

* `median` has a 50% breakdown point, so it read *exactly* 1.0 on 57% of Two-Room and 77% of Cube
  checkpoints (the rate saturates on 0% of all four tasks);
* five repeats of one checkpoint gave it CV 0.50 against the rate's 0.02, because the median sits
  at the sign change of a right-skewed (+3.8) gap distribution;
* it was worse on every pool outside the one it was fitted on (MAZE calibration 24.2 → 11.4
  points, held-out-method ρ +0.55 → +0.66);
* mixing the two forms across a comparison produces a large spurious level shift.

`m_optimism` (the median gap) in any older metrics CSV is an input to that deleted form. **It is
not sobriety.**

**Determinism.** The CEM's sampling is seeded from the cell id via `zlib.crc32` (not `hash()`,
which is salted per process), so the factor is a deterministic function of the checkpoint.

**Planner conditioning.** Sobriety is the one factor that contains a search by construction.
Across six deployed planners (CEM, MPPI, iCEM, predictive sampling, multi-start and single-start
AdamW) VIScore keeps positive within-task association in all 18 planner × task cells (+0.29 to
+0.91, median +0.67) and does **not** decay as the planner weakens (rank correlation between a
planner's success deficit and its within-task ρ: +0.07). Cross-task *numerical* comparability is
what degrades, and only for planners whose weakness is task-selective. A comparison must keep the
probe fixed.

**Resolution floor.** `gap_resolution` = median elite-cost sd / √16, the standard error of the
elite mean: gaps below it are indistinguishable from zero, so a sobriety of exactly 1 would be
unwarranted. Reported, not hidden.

---

## 4. Limitations

Measured, not hypothesized. Quote these when quoting the score.

1. **Sufficiency yes, necessity no.** All-factors-high ⇒ the model is fine (holds). Low factor ⇒
   fixing it raises success (**falsified**): on one validation environment, veracity
   0.58 → 1.00 and VIS 0.50 → 1.00 moved success 52 → 50.
2. **Search-based planning only.** A goal-conditioned inverse-dynamics / amortized policy has no
   sobriety estimand — its success is independent of rollout fidelity, so the prerequisites are
   different ones.
3. **Discrete-mode tasks are out of scope.** Cube succeeds only if the gripper enters a closed
   mode on the block, a discrete transition none of the three continuous factors represents; its
   within-task column is parenthesized in the paper and excluded from pooled and calibration
   numbers.
4. **`Ê_H` assumes horizon-stable error directions** (§2).
5. **Label noise bounds what any metric can score.** Among converged checkpoints one evaluation
   seed ranks another's success at only +0.59 / +0.53 / +0.09 (PushT / Reacher / Two-Room): over
   the last two training epochs the between-checkpoint success spread is 1.1 / 2.9 / 2.7 points
   against a 4–6 point binomial standard error, so nothing there is rankable by any metric. Before
   correlating any metric against success, check that between-checkpoint success sd exceeds the
   binomial standard error of the labels — otherwise the correlation is measuring noise.
6. **Saturated factors carry no information.** Check `VISFactors.saturated` before reading the
   product as a diagnosis.

---

## 5. Verification of this implementation

What was actually checked, with the numbers, so the claim can be audited rather than trusted.

**Algebra — bit-exact.** Recomputing `d_tol`, `veracity`, `tr Σ_z`, `m_emp`, `influence` and
`sobriety` from the released caches (latent banks, `(S, E)` spectra, per-anchor gap dumps) that the
paper's tables were built from reproduces the published values to a **worst relative difference of
4.4e-15** over PushT / Reacher / Two-Room cells — float round-off.

**Full pipeline — 1e-4, for a stated reason.** Re-running the rollout kernels on a real released
checkpoint (a Two-Room VISReg run at epoch 10, on CPU, from the cached latent bank):

| quantity | published | recomputed | rel. diff |
|---|---|---|---|
| `d_tol_grade` | 17.734496 | 17.734497 | 4.2e-08 |
| `σ_roll` | 3.6337103 | 3.6333776 | 9.2e-05 |
| `veracity` | 0.98532387 | 0.98533295 | 9.2e-06 |
| `m_emp` | 128.87666 | 128.88725 | 8.2e-05 |
| `influence` | 1 | 1 | 0 |
| `sobriety` | 0.703125 | 0.640625 | 8.9e-02 |

`σ_roll`'s 9e-5 is the `ddof` difference documented in §1, and `m_emp` inherits it through `Ê_H`.

**Sobriety is reproducible per seed, not across seeds.** That 4/64-anchor difference is the probe's
own sampling variability rather than a porting error: repeating with the same cell id returns
*identically* 0.640625, while four different CEM seeds on the same checkpoint give
0.640625 / 0.750000 / 0.640625 / 0.750000 — a range containing the published 0.703125. On this
checkpoint **25–27 of the 64 anchors have `|ΔJ|` below `gap_resolution`**, so their sign is decided
by numerics rather than by the landscape, and CPU-vs-GPU float and RNG differences flip a few.

Read that as a property of the estimator worth knowing rather than a defect of the port: this
Two-Room checkpoint is a near-degenerate case — a flat landscape is exactly why the earlier
median-gap form saturated on 57% of Two-Room checkpoints — and its per-seed spread of ±0.055 is
larger than the ±0.02-scale variability reported for the rate elsewhere. **When `gap_resolution`
is large relative to the gaps, average sobriety over CEM seeds before comparing two checkpoints.**

**Invariances — asserted in `tests/`.** Latent rescaling leaves veracity unchanged (and scales
`d_tol`, `σ_roll` together); a linear reparameterization leaves `m_emp` unchanged; an unreachable
tolerance makes `d_tol` abstain with `nan` instead of extrapolating. No GPU, dataset or checkpoint
required: `pytest tests/`.

---

## 6. Partial scores, and what they do and do not support

`exclude=` / `--exclude` returns the product over the remaining factors, labelled `VIS` / `VI` /
`IS` / `V` / `I` / `S` in the record. Excluded factors are `nan`, never `1.0`.

**Why this exists.** Not every model has an estimand for every factor. An amortized policy
(goal-conditioned inverse dynamics, behaviour cloning) never runs a search, so there is no
"search beats the expert in imagination" event and sobriety has nothing to measure. Separately, the
factor ablation treats single factors as objects of study, and their per-pool behaviour is
informative in its own right:

| pool | VIS | V | I | S | constant ref. |
|---|---|---|---|---|---|
| development | +0.88 / 10.2 | +0.80 / 12.4 | +0.54 / 21.9 | +0.89 / 9.3 | 24.4 |
| held-out ckpt | +0.91 / 7.0 | +0.81 / 12.7 | +0.65 / 14.0 | +0.89 / 7.3 | 18.2 |
| held-out method | +0.75 / 8.3 | +0.62 / 10.4 | +0.09 / 12.2 | +0.68 / 8.9 | 11.3 |
| held-out dataset | +0.87 / 11.5 | +0.85 / 31.9 | +0.87 / 9.4 | +0.88 / 8.3 | 41.7 |

(pooled ρ / calibration error in success points). No single factor dominates every shift and each
one *fails* somewhere — influence inverts on held-out methods, veracity mis-calibrates by 32 points
on the held-out dataset. The full product never falls below +0.66 on any pool. So a subset is a
legitimate measurement, and also a weaker instrument than the product on the cross-task axis.

**Three constraints that come with a subset.**

1. **It is not VIScore.** A different factor set is a different scale. Comparing `VI` with a
   published `VIS`, or with `VS`, is meaningless. `factors` is carried in the record and the CSV so
   this cannot happen silently.
2. **Computable ≠ validated.** `--exclude sobriety` on an amortized policy produces a number, but
   nothing here validates that number as a predictor of *that policy's* success: measured on a
   23-run fixed-epoch pool, an amortized policy's success was **independent of rollout fidelity**,
   which removes veracity's mechanism as well. The right axis there is representation content
   (`--baselines` probe R²), reported as a description, not a prediction.
3. **Cost is not symmetric.** Sobriety is the 64×512-rollout probe and dominates runtime — dropping
   it took one CPU checkpoint from 52 s to 19 s, so it is roughly two thirds of the cost. Dropping veracity saves only the `d_tol` shell,
   because influence's noise floor `Ê_H` is anchored to `σ_roll` and therefore still needs the
   open-loop rollout. Asserted in `tests/test_smoke.py::test_excluding_veracity_still_measures_sigma_roll`.

---

## 7. Cost

Batched veracity + influence + sobriety: **≈ 7 s per checkpoint on one H100** once the probe is
encoded (encoding is cacheable with `--latents-dir` and is the only stage scaling with pixel
count). A single 50-episode planning evaluation takes at least 20× the wall clock and two orders
of magnitude more predictor rollout steps. Averaging success over more evaluation seeds raises
evaluation cost and leaves diagnostic cost unchanged.
