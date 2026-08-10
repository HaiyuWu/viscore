# Design: why the metric is a separate package from the training code

One decision shapes this repository: **`viscore/` does not import the training framework.**

```
viscore/   numpy · scipy · torch                        ← the contribution
viswm/      + stable-worldmodel · stable-pretraining · lightning · hydra   ← one instance of a model
```

## Why

**1. The metric outlives the framework.** The research fork this code comes from is pinned to
`stable-worldmodel==0.0.6`, whose data backend is HDF5; current swm has moved to Lance and to a
different dataset API. If the metric imported swm, upgrading the training side would break every
published number. It does not, so the migration in [SWM_MIGRATION.md](SWM_MIGRATION.md) touches
`viswm/` and `reproduce/` only — and the invariance tests in `tests/` keep passing throughout
without a GPU, a dataset, or a checkpoint.

**2. A diagnostic that only works on its own family is not a diagnostic.** VIScore's central
claim is that its readings transfer — across checkpoints, across tasks, and across *methods*
trained in other codebases. That is testable only if scoring a foreign model requires no adoption
of our training stack. The requirement is three methods (`encode`, `action_embed`,
`predict_next`), stated in `viscore/adapters.py`; everything else is the scorer's problem.

**3. The expensive dependency is data, not code.** All three factors read one shared latent bank
plus the predictor. Nothing needs a dataloader, an optimizer, a logger, or an environment — so
requiring them would be cost with no benefit. `pip install viscore` on a laptop is enough to
re-derive every number from a released probe and latent cache.

## The seams, and what each one is allowed to know

| seam | file | knows about |
|---|---|---|
| model API | `viscore/adapters.py` | one model family's `encode`/`predict` signatures, and the pickle module-name trick that makes LeWM checkpoints loadable |
| task API | `viscore/tasks.py` | environment success criteria: tolerance, graded coordinates, ignored coordinates |
| data API | `viscore/probe.py` | the HDF5 column layout of the released datasets |
| everything else | `veracity/influence/sobriety/score.py` | **only** arrays, a `TaskSpec`, and the three model methods |

The rule that keeps this honest: if a change to the training stack would require editing
`veracity.py`, `influence.py`, `sobriety.py`, or `score.py`, the abstraction has leaked. Adding a
task edits `tasks.py`. Adding a model family edits `adapters.py`. Adding a dataset format edits
`probe.py`.

## Consequences worth knowing about

* **The probe is a build artifact, not a library object.** It is written once with `rng(0)` and
  then frozen; the metric will happily score against a differently-built probe and produce
  numbers that are not comparable to published ones. Treat a probe like a test set.
* **`history_size` is read off the checkpoint** (`predictor.pos_embedding.shape[1]`), not
  configured per task, so a model trained with a different context length scores correctly
  without configuration.
* **Latent caching is explicit** (`--latents-dir`). Encoding is the only stage whose cost scales
  with pixel count, so an epoch sweep re-scores from cache in seconds; but a stale cache is a
  silent correctness bug, so the cache key is the cell id and nothing is invalidated
  automatically. Delete the directory when you change encoders.
* **Factors are returned as a record, not a scalar.** `VISFactors` carries `d_tol`, `σ_roll`,
  `m_emp` in nats, `p̂`, and `saturated`. The product alone cannot tell you that one third of it
  was pinned at 1.0 on this task.
