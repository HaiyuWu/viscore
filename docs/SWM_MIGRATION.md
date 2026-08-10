# Migrating the training side to current `stable-worldmodel`

**Status: planned, not done.** This release is pinned to `stable-worldmodel==0.0.6`, which is what
every published number was produced with. This document is the migration plan and, more
importantly, the list of things that must be **re-verified rather than assumed** — because several
published absolute success rates depend on 0.0.6 protocol behavior, not only on the models.

`viscore/` is unaffected by this migration by construction (see [DESIGN.md](DESIGN.md)): it
imports no swm, and its tests run without a dataset or a checkpoint. Everything below concerns
`viswm/` and `reproduce/`.

## The pinned environment

```
stable-worldmodel==0.0.6      stable-pretraining==0.1.6     lightning==2.6.1
hydra-core==1.3.2             torch==2.11.0+cu128           Python 3.10
```

## The swm/spt API surface `viswm/` actually uses

Small on purpose — this is the whole migration surface.

| symbol | used in | role |
|---|---|---|
| `swm.data.HDF5Dataset` | `train.py`, `eval.py` | dataset backend |
| `swm.data.utils.get_cache_dir` | `train.py`, `eval.py` | resolves `$STABLEWM_HOME` |
| `swm.World` | `eval.py` | environment + episode bookkeeping |
| `swm.PlanConfig` | `eval.py` | planning horizon / action-block config |
| `swm.policy.AutoCostModel` | `eval.py` | loads a checkpoint as a cost model |
| `swm.policy.WorldModelPolicy` | `eval.py` | MPC loop around a solver |
| `swm.policy.RandomPolicy` | `eval.py` | baseline / `ZeroPolicy` base class |
| `spt.Module`, `spt.Manager`, `spt.data.{DataModule,random_split,transforms.Compose}`, `spt.backbone.utils.vit_hf` | `train.py` | training loop |
| `spt.data.dataset_stats.ImageNet` | `eval.py` | normalization constants |

## Step 1 — data backend: HDF5 → Lance

0.1.x replaces the HDF5 backend with Lance. Consequences:

* every `.h5` in the manifest and every `config/train/data/*.yaml` that names one is orphaned; a
  conversion step (`h5 → .lance`) becomes part of the artifact pipeline;
* Lance's Tokio runtime is **not fork-safe**, so DataLoader workers must start via `forkserver`;
* forkserver *pickles* the `Subset` that `random_split` returns. On spt 0.1.6 the base `Dataset`
  holds a live `_trainer` reference and defines no `__getstate__`, which drags
  `trainer → spt.Module → forward=partial(...)` into the pickle and fails with *"functools.partial
  has no `__name__`"*. Under fork (the HDF5 path) workers inherit memory and nothing is pickled,
  which is why this never appears on 0.0.6. Newer swm drops `_trainer` on pickle; **verify that
  before removing any local workaround.**

`viswm/train.py` currently raises `NotImplementedError` for `data.format != "hdf5"` rather than
silently ignoring the key. Replace that branch with the real backend, do not delete it.

## Step 2 — CUDA wheel

The pinned `torch==2.11.0+cu128` wheel is not among current swm's dependency resolutions. Resolve
the torch/CUDA pin **before** touching configs; a mismatched wheel presents as unrelated runtime
errors deep in the encoder.

## Step 3 — re-verify the evaluation protocol (do not skip)

Absolute success rates in the paper depend on these 0.0.6 behaviors. If a new version changes any
of them, published numbers stop being comparable — and the change will be silent.

1. **Transient passes count as success.** 0.0.6 scores `episode_successes |= terminated`
   (`world.py:961`), so touching the tolerance ball at *any* step of the budget is a success. This
   raises every floor.
2. **`world.terminate_at_goal` does not exist in 0.0.6** and is silently swallowed if passed. On
   one validation environment, setting it changed a random-policy score from 68% to 0% — i.e. a
   config key that looks respected and is not.
3. **`set_state` restores velocities.** A body with momentum coasts toward where it was recorded
   `d` steps later, so a do-nothing policy scores above zero on some tasks (measured floors:
   0 / 0 / 8 / ~30±5 / 74 across environments). **Measure the do-nothing floor per environment
   after migrating** — `eval.py policy=zero` exists for exactly this. Random is not a substitute.
4. **Frameskip 5 is concatenation of 5 distinct actions, not action repeat.** The 0.0.6
   `PlanConfig.action_block` docstring says "repeated" and is wrong; verified against code, the
   papers, and the on-disk HDF5. If a new version *implements* what that docstring says, action
   dimensionality and every trained model become incompatible.
5. **Goal sampling.** Goals come from the same trajectory `goal_offset_steps` ahead (d = 25
   default), `num_eval = 50`, `eval_budget = 2d`, CEM `n_steps = 30` on PushT and 10 elsewhere,
   evaluation seed fixed in the config. Confirm each after migration.

## Step 4 — the metric side needs no change, but check two things

* **Probe building** (`viscore/probe.py`) reads HDF5 columns directly with `h5py`, independent of
  swm. If the released datasets are re-published in Lance form, add a `probe` reader for that
  format; the probe *contract* (arrays, dtypes, `rng(0)` sampling) must not change, or previously
  published scores become incomparable.
* **Checkpoint format.** `LeWMAdapter` unpickles a full-object `torch.save` of `jepa.JEPA` and
  needs `jepa` / `module` importable as top-level names (`register_lewm_modules` does this with the
  copies in `viswm/`). If the training side starts saving state dicts instead, add a state-dict
  path to `adapters.py` — keep the object path working so released checkpoints stay loadable.

## Acceptance criteria for calling the migration done

1. `pytest tests/` passes unchanged (it does not depend on swm at all).
2. `viscore score` on one released checkpoint reproduces its published factors to float
   round-off — the metric must be provably untouched by the migration.
3. `eval.py policy=zero` reports a do-nothing floor per environment, recorded in
   `reproduce/RESULTS.md` next to the model numbers.
4. One retrained VIS-WM PushT seed reproduces its planning success within the reported
   multi-seed spread. Below that, the migration changed the model, not just the plumbing.
