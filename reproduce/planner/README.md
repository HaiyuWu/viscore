# Planner-family transfer

Reproduces `tab:planner-transfer`: the same checkpoints scored under six planners, to test whether
VIScore depends on the search procedure it was developed with.

```bash
python reproduce/planner/planner_tables.py            # the six blocks, as printed in the paper
python reproduce/planner/planner_tables.py --latex    # LaTeX rows
```

Verified against the published table: 6 blocks x 7 rows, 0 mismatches.

## Why this lives in its own folder

Every arm must be evaluated on the **same** checkpoints for the comparison to be paired, so this
experiment uses a fixed 36-checkpoint set rather than the pools of `reproduce/tables.py`. All 25 of
its training runs also appear in those pools, and 20 of the 36 checkpoints are themselves cells of
the development or held-out pool; the remaining 16 are further epochs of the same runs, which widens
the quality range any single pool covers. It is a subset of the paper's **runs**, not of its
reported **checkpoints**, so these numbers are not comparable to `reproduce/tables.py`'s — the
pools differ, and so do the success-rate levels they average over.

A refresh of this experiment on the full 103-checkpoint held-out pool is in progress. It is not what
the paper reports, and it is not published here.

## Files

| File | Contents |
|------|----------|
| `planner_cells.csv` | 36 checkpoints: planner-independent metrics, one sobriety per probe family, one success rate per planner |
| `planner_tables.py` | recomputes the table; no GPU, no checkpoints, no dataset |

## The probe assignment is a rule, not a choice

Sobriety is the one factor that contains a search, so each planner block needs a probe. The rule was
declared before any outcome was seen — **same search family, matched budget fraction**:

| planner | probe |
|---------|-------|
| CEM, MPPI, iCEM, predictive sampling | mini-CEM (`sobriety_mcem`) |
| gradient, 100 starts x 30 steps | mini-AdamW (`sobriety_madam`) |
| single-start gradient, 1 start x 100 steps | single-start mini-AdamW (`sobriety_ss`) |

Selecting per planner whichever probe correlated best would manufacture the result: predictive
sampling's pooled correlation is +0.51 under the declared probe and +0.80 under the one that would
have flattered it. The frozen table carries all three sobriety columns so this is checkable rather
than described.

## Scope

All six planners optimise an action sequence through the learned predictor and replan on a receding
horizon — that is, they are model-predictive controllers. Amortized policies (an inverse dynamics
model, goal-conditioned behaviour cloning, a learned actor) are a different class and are outside
what this table tests: they neither roll the predictor out nor search at deployment, so veracity
loses its causal path and sobriety has no estimand at all rather than merely a missing estimator.
`viscore`'s `exclude` option exists for that case.
