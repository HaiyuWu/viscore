# Published numbers, and the command that regenerates each one

Numbers are from the current manuscript; the paper is the authority if the two disagree. Every
metric column below is produced by the code in `viscore/`, whose algebra reproduces the cached
intermediates these tables were built from to 4.4e-15 relative — see
[../docs/METRIC.md §5](../docs/METRIC.md#5-verification-of-this-implementation) for what was
verified how, including the one quantity (sobriety) that is reproducible per CEM seed rather than
across seeds.

Two conventions that change what a number means, so they are stated once here:

* **Peak epoch** = maximum evaluation `success_rate` over epochs, **ties to the later epoch**.
* **Significance** — one threshold per comparison. A single `num_eval=50` evaluation has a true
  success sd of ≈6 points, so most single-seed gaps in this literature collapse under 3–4
  evaluation seeds. Add seeds before hypothesizing a mechanism for a gap.

---

## 1. VIScore against the diagnostics in common use

Within-task Spearman ρ with planning success, pooled ρ across tasks, and cross-task calibration
error in success points (lower is better). The first two pools use a leave-one-task-out isotonic
fit; the two transfer pools are scored by the map already fitted on development, with no refit,
because a new task or a new method arrives without labels. Cube's within-task column is
parenthesized where a pool contains it and always excluded from pooled/calibration — its success
turns on a discrete grasp mode none of the three factors represents (see
[../docs/METRIC.md](../docs/METRIC.md#4-limitations)).

**Development pool**

| metric | PushT | Reacher | Two-Room | Cube | pooled | calib. err. |
|---|---|---|---|---|---|---|
| straightness | +0.51 | +0.13 | −0.67 | (+0.01) | +0.33 | 40.4 |
| physical-state probe R² | +0.41 | +0.74 | +0.57 | (−0.05) | +0.77 | 14.5 |
| empowerment `m_emp` | +0.82 | +0.86 | +0.71 | (−0.18) | +0.40 | 23.1 |
| **VIScore** | +0.80 | +0.83 | +0.71 | (−0.29) | **+0.88** | **10.2** |
| *ref: constant predictor* | | | | | | *24.4* |

**Held-out checkpoints** (runs disjoint from development; 103 checkpoints from 33 runs spanning
epochs 1–12, three evaluation seeds each)

| metric | PushT | Reacher | Two-Room | pooled | calib. err. |
|---|---|---|---|---|---|
| straightness | +0.38 | −0.81 | −0.74 | −0.08 | 19.8 |
| physical-state probe R² | +0.38 | +0.65 | +0.71 | +0.61 | 27.6 |
| empowerment `m_emp` | +0.73 | +0.82 | +0.86 | +0.49 | 15.6 |
| **VIScore** | +0.84 | +0.72 | +0.83 | **+0.91** | **7.0** |
| *ref: constant predictor* | | | | | *18.2* |

**Held-out methods** (23 checkpoints from four methods; frozen development calibration, no refit)

| metric | pooled | calib. err. |
|---|---|---|
| straightness | +0.22 | 17.4 |
| physical-state probe R² | +0.32 | 16.1 |
| empowerment `m_emp` | +0.12 | 12.0 |
| **VIScore** | **+0.75** | **8.3** |
| *ref: constant predictor* | | *11.3* |

**Held-out dataset** (20 MAZE checkpoints from two independent runs; frozen calibration)

| metric | pooled | calib. err. |
|---|---|---|
| straightness | +0.66 | 31.2 |
| physical-state probe R² | +0.75 | 45.4 |
| empowerment `m_emp` | +0.87 | **9.2** |
| **VIScore** | **+0.87** | 11.5 |
| *ref: constant predictor* | | *41.7* |

The instruments separate on the **cross-task** axis, not within tasks: raw empowerment matches
VIScore within every task of the held-out pool and is the better ranker on two of the three, but
it is measured in nats whose meaning does not transfer, so its pooled correlation falls to +0.49
and its calibration error to 15.6 points — worse than the 18.2-point constant predictor's margin
allows any use. Restricting the held-out pool to its converged tail adds no further test: over the
last two epochs the between-checkpoint spread of success is 1.1 / 2.9 / 2.7 points against the 4–6
point standard error of one 50-episode evaluation (ratios 0.3 / 0.5 / 0.7), so no metric's
within-task ordering is recoverable there. That is a resolution limit of the evaluation, not a
property of any diagnostic.

```bash
# scores for one run's epoch sweep (the unit every within-task correlation is computed over)
PROJECT_DIR=$PWD DATA_HOME=$VISCORE_HOME SUBDIR=<run> TASK=pusht \
  sbatch --account=<acct> --partition=<part> --gres=gpu:1 reproduce/slurm/score.slurm
# the success labels they are correlated against
EPOCHS=1,2,3,4,5,6,7,8,9,10 sbatch --array=0-9%8 --export=ALL,\
PROJECT_DIR=$PWD,DATA_HOME=$VISCORE_HOME,SUBDIR=<run>,EVAL_CONFIG=pusht reproduce/slurm/eval.slurm
```

## 2. Factor ablation — the product is a summary, the factors are the diagnosis

Pooled ρ / calibration error per pool. `n / runs` in the first column.

| pool (n / runs) | VIS = V·I·S | V veracity | I influence | S sobriety | ref. |
|---|---|---|---|---|---|
| development (137/14) | +0.88 / **10.2** | +0.80 / 12.4 | +0.54 / 21.9 | **+0.89** / **9.3** | *24.4* |
| held-out ckpt (103/33) | **+0.91** / **7.0** | +0.81 / 12.7 | +0.65 / 14.0 | +0.89 / 7.3 | *18.2* |
| held-out method (23/23) | **+0.75** / **8.3** | +0.62 / 10.4 | +0.09 / 12.2 | +0.68 / 8.9 | *11.3* |
| held-out dataset (20/2) | +0.87 / 11.5 | +0.85 / 31.9 | **+0.87** / 9.4 | **+0.88** / **8.3** | *41.7* |

No factor dominates every shift, and **the strongest factor changes with the regime** — which is
the argument for reading the factors individually rather than only the product:

* **sobriety** leads among converged models, where representation and capacity vary little but a
  localized predictor error can still be amplified by search;
* **influence** is competitive where the pool spans collapsed, undertrained and healthy models
  (capacity has not saturated) and goes silent where it has — exactly 1.0 on every converged PushT
  checkpoint and on 13 of the 18 method-shift cells, hence the +0.09 there;
* **veracity** binds when the task's precision requirement is tight;
* each factor also *fails* somewhere (influence carries no signal on held-out methods, +0.09, and
  −0.14 once our own checkpoints are dropped; veracity mis-calibrates to 31.9 points on the
  held-out dataset), and either failure would be invisible in
  the other two.

The product never falls below +0.66 on any pool. It is fixed in advance rather than selected per
pool: less locally optimal than post-hoc factor selection, and interpretable.

## 3. Transfer across planner families *(paper-only appendix — not reproducible from the release)*

> The published artifacts are CEM-only, by design: labels and sobriety probes must come from the
> same search family. The numbers below are reported from the paper; re-measuring them needs the
> per-planner probes and evaluations, which are not published.

Six planners (CEM, MPPI, iCEM, predictive sampling, multi-start AdamW, single-start AdamW) on the
same 36 checkpoints, budgets from 100 to 9000 model evaluations. Veracity and influence are
planner-independent; sobriety is recomputed with a family- and budget-matched probe assigned
before results were seen.

VIScore keeps **positive within-task association in all 18 planner × task cells** (+0.29 to
+0.91, median +0.67), and it does **not** decay as the planner weakens: the rank correlation
between a planner's per-task success deficit against CEM and its within-task ρ is +0.07.
Predictive sampling scores 43.3 on PushT against CEM's 84.7 and VIScore still ranks that task's
checkpoints at +0.76 — a planner that finds much worse plans still finds better plans with a better
model.

Cross-task *numerical* comparability is what degrades, and only for planners whose weakness is
**task-selective**: pooled ρ / calibration stays strong for CEM (+0.81 / 9.4), MPPI (+0.81 / 17.4),
iCEM (+0.87 / 6.9) and multi-start gradient (+0.86 / 10.7), and weakens for predictive sampling
(+0.51 / 26.4) and single-start gradient (+0.12 / 16.2). MPPI is the counter-case that prevents
stating this as a rule: its per-task deficits span 20 points and it still pools at +0.81.

```bash
# swap only the optimizer; keep checkpoint, goals, episode count and planning cost fixed
EXTRA="solver=mppi" ... reproduce/slurm/eval.slurm         # solver ∈ {cem,mppi,icem,ps,adam}
```

## 4. VIS-WM vs LeWM: what the regularizer swap actually buys

> Every number in this section is reproducible from the shipped labels:
> `python reproduce/planning_tables.py --strict` recomputes it and gates it against the published
> value. The ± is the standard error over the cell set (OOD: 6 shapes × 3 training × 3 evaluation
> seeds = 54), and the epoch each arm was reported at is data in `pools/planning_arms.csv`.

**In-distribution planning** — VIS-WM is competitive with the strongest end-to-end latent world
models under the conventional single-evaluation-seed protocol (strongest on PushT, Cube and
Two-Room). Under three fresh evaluation seeds it becomes **indistinguishable from LeWM**. The
single-seed protocol was over-confident, which is a finding about the protocol rather than about
either objective. See the paper's main table for per-task numbers.

**Out-of-distribution PushObj shapes** — success on six unseen object shapes, peak epoch,
3 training × 3 evaluation seeds:

| model | d=25 | d=50 | d=75 |
|---|---|---|---|
| LeWM (SIGReg, B=128) | 79.1 ±0.9 | 61.8 ±1.1 | 44.3 ±1.2 |
| **VIS-WM (VISReg, B=512)** | **82.0** ±0.7 | **62.0** ±1.1 | **45.6** ±1.1 |
| VIS-WM, B=128 (matched control) | 77.4 ±1.0 | 59.7 ±1.3 | 41.8 ±1.3 |
| VIS-WM, B=256 | 82.0 ±0.7 | 62.2 ±1.2 | 44.3 ±1.2 |
| random policy | 10.1 | 7.3 | 3.0 |

The matched-batch control is the point: **at B=128 the two objectives are indistinguishable**, and
the advantage in the top block comes from the batch-size ladder. VISReg's sorted-quantile shape
term has a finite-sample floor that shrinks with batch size, so batch size acts as a hidden
regularization-strength knob for VISReg while SIGReg's normalized Epps–Pulley statistic is
batch-invariant in expectation. Judge the batch effect by the whitened shape residual and shape
SNR, never by raw loss values — a "lower loss" at larger B is largely the floor moving.

Absolute PushObj levels carry the caveats in [ARTIFACTS.md](ARTIFACTS.md#tier-3--datasets--180-gb--7-ood-files)
(degenerate static tasks, rotational symmetry ignored by the criterion, goal marker rendered in the
substituted shape). Relative comparisons on the same file are fair.

## 5. Protocol floors — measure them before quoting any absolute success rate

`eval.py policy=zero` emits the zero action; `policy=random` samples uniformly. They are **not**
interchangeable and neither bounds the other: on one validation environment do-nothing scored 74%
while random scored 0%. Two protocol properties make a nonzero floor possible in any
same-trajectory-goal setup — the start state may already satisfy the tolerance, and `set_state`
restores velocities so a body with momentum coasts toward where it was recorded `d` steps later —
and swm 0.0.6 additionally counts a *transient* pass at any step of the budget as a success.

Measured floors across the environments used here: 0 / 0 / 8 / ~30±5 / 74. Record the floor for
every new environment before reporting a model on it.
