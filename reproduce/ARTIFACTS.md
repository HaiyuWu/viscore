# Artifacts: what is published, where, and what you actually need

Everything is on the Hub. Nothing here is a placeholder.

| | repository | size |
|---|---|---|
| checkpoints | [`BooBooWu/viscore`](https://huggingface.co/BooBooWu/viscore) (model) | 37.7 GiB |
| datasets + reproduction bundle | [`BooBooWu/viscore`](https://huggingface.co/datasets/BooBooWu/viscore) (dataset) | 4.9 GiB |
| the four base datasets | [LeWorldModel collection](https://huggingface.co/collections/quentinll/lewm) | 80.5 GiB compressed |

Fetch with `reproduce/download.py`.

---

## What you need for which claim

| you want to | download | size | command |
|---|---|---|---|
| reproduce `tab:viscore` | nothing — the manifest ships with the repo | 0 | `python reproduce/tables.py` |
| reproduce the Section-4 planning tables | nothing — the labels ship with the repo | 0 | `python reproduce/planning_tables.py --strict` |
| recompute the metric values yourself, no GPU | `bundle/` | 3.8 GiB | see [below](#cpu-only-reproduction) |
| score the released checkpoints from pixels | `vis-wm/` or `pools/` + one base dataset | 1–37 GiB + 12–43 GiB | `--tier vis-wm` / `--tier pools` |
| re-evaluate planning success | checkpoints + the base dataset for that task | | `reproduce/slurm/eval.slurm` |
| retrain from scratch | base datasets | 268 GB decompressed | `reproduce/slurm/train.slurm` |

## Model repository — `BooBooWu/viscore`

```
vis-wm/<task>/seed<S>/vis-wm_epoch_<N>.ckpt     0.88 GiB, 39 files (reported epoch only)
baselines-lewm/maze2d/seed729/                       0.07 GiB, 3 files
pools/<pool>/<env>/<run>/<model>_epoch_<N>.ckpt      36.74 GiB, 547 files
pools/pool_manifest.csv, pools/pool_assignment.csv
```

**VIS-WM arms** (VISReg, λ = 4.5, lr 1e-4, 10 epochs, batch 128): PushT / Reacher / Two-Room / Cube
at seeds 403 / 46 / 729, each at the epoch the paper reports — maximum success on the single-seed
d=25 evaluation, ties to the later epoch. Per-epoch checkpoints live under `pools/`.
**MAZE has one seed (729), not three** — it is the
held-out *dataset* validation and that block contains two runs in total, the VISReg run and its
SIGReg counterpart under `baselines-lewm/`.

**Pools** are the checkpoints `tab:viscore` is computed on, laid out by pool:

| pool | checkpoints / runs | n (PushT/Reacher/Two-Room/Cube) | labels |
|---|---|---|---|
| `development` | 137 / 14 | 57 / 50 / 30 / — | d = 25, all available evaluation seeds |
| `heldout` | 103 / 33 | 27 / 44 / 32 / — | fresh seeds 5501 / 60601 / 90210, three per checkpoint |
| `heldout_method` | 23 / 23 | 7 / 8 / 8 / — | four world-modeling methods, nine seeds per checkpoint |
| `heldout_dataset` | 20 / 2 | — (MAZE) | the unseen task, its own seeds |
| `calibration_fit` | 472 / 47 | 152 / 180 / 140 / — | every cell carrying a development label; 137 survive the split into `development` and 103 into `heldout`, and the frozen calibration map is fitted on all of them |

Two columns answer two different questions, and conflating them is the easiest way to misread this
file. `fold` is a property of the **training run**: which side of the run-level split it fell on
(`dev`, `heldout`, or `unsplit` -- the Cube runs, which entered no fold). The `in_*` flags are
properties of the **cell**: which reported number it enters. They are not the same set -- 335 cells
belong to held-out runs, but only 103 of them are the held-out pool the paper reports, because that
pool takes one epoch ladder per run under the three fresh evaluation seeds. Select with `in_*`,
audit independence with `fold`.

The split is at the level of the **training run**, not the evaluation seed: every constant (τ, the
`d_tol` recipe, the calibration maps) is fitted on `development`, and no checkpoint in a test pool
comes from a run that contributed there. `pool_assignment.csv` is the object to audit. A run absent
from it never carried a development label. Three SIGReg Two-Room runs landed in neither fold and are
not released, which is why `heldout` carries no SIGReg checkpoint on Two-Room. Earlier revisions reported a separate `terminal` pool of converged
checkpoints; 14 of its 17 runs were also in the held-out fold, so its cells are folded into
`heldout` here (they contribute the epoch 4–10 cells).

> A run's epochs can **split across pools**: which epochs carry which pool's labels is a property of
> the evaluation, not of the run. `visreg_tworoom_..._seed403_150694` is one example — epoch 2 sits in
> `heldout`, epochs 4-10 in `terminal`. So look a checkpoint up in `pool_manifest.csv` rather than
> assuming a run lives in one directory.

> **Peak-epoch selection**, when you need one checkpoint per run: maximum evaluation `success_rate`
> over epochs, **ties to the later epoch**. State it when reporting — it changes which checkpoint
> "peak" means.

## Dataset repository — `BooBooWu/viscore`

```
data/maze2d_medium.h5.zst          600 MB    MAZE, the held-out dataset
data/pushobj_<shape>.h5.zst        61-121 MB each, 7 shapes (6 unseen + T control)
bundle/latents|spectra|gaps/       3.70 GiB  per-cell derived quantities for 557 cells
bundle/probes/probe_<task>_nopixels.npz      the probe minus pixels (~1 MB, not 1.5 GB)
bundle/pool_manifest.csv, bundle/pool_assignment.csv
bundle/success_labels.csv                    every planning evaluation (2753 rows, 512 KB)
bundle/planning_arms.csv                     the epoch each Section-4 arm was reported at
```

### Success labels

`success_labels.csv` is one row per **(checkpoint, task, goal offset, evaluation seed)** with the
50-episode success rate, covering the main table, the long-horizon rows, the OOD shapes, the batch
ablations and the per-seed labels of the `heldout` and `terminal` pools. It turns Section 4 from a
re-run into a groupby: `reproduce/planning_tables.py` reproduces all 18 published values and fails
loudly if any drifts. `planning_arms.csv` carries the peak epoch each arm was reported at, so the
selection is auditable instead of described.

**One planner throughout.** Every published label, every published sobriety gap dump and the
manifest's sobriety column are CEM — the deployed planner, and the protocol behind every number in
Sections 4 and 5. Nothing in the release can pair a CEM success label with a sobriety probed by a
different search, which is a real failure mode rather than a cosmetic one: mixing probe families
produces a spurious level shift.

Not included, deliberately: the six-planner robustness appendix (it needs per-planner sobriety
probes, which are unpublished), MAZE labels, the superseded 3×3 OOD grid (evaluation seeds
0/123/2024, replaced by the fresh-seed reruns that are included), and the amortized-policy
appendix.

### CPU-only reproduction

All three factors are pure linear algebra once the latents and the `(S, E)` spectra exist, so the
bundle makes the paper checkable without a GPU or the 268 GB of pixels:

```bash
python reproduce/download.py --tier bundle --dest $VISCORE_HOME
python reproduce/tables.py                 # every cell of tab:viscore
python reproduce/tables.py --boot 2000     # + run-clustered bootstrap intervals
```

This is also how the released implementation was verified: recomputing `d_tol`, `veracity`,
`tr Σ_z`, `m_emp`, `influence` and `sobriety` from these files reproduces the published values to
4.4e-15 relative (`docs/METRIC.md` §5).

### PushObj caveats

Action replays of the T-block expert actions on substituted shapes (AdaJEPA App. A.2 protocol).
Block-static degenerate tasks inflate absolute success; the swm 0.0.6 criterion ignores the object's
rotational symmetry, under-counting square / plus / Z / I; and the goal marker renders *in the
substituted shape*, a perception confound. **Between-method comparisons on the same file are fair;
absolute levels are not portable.**

## Base datasets — not re-hosted

LeWorldModel already publishes them, and re-uploading 268 GB of someone else's data helps nobody.

All four live in LeWorldModel's own [LeWM collection](https://huggingface.co/collections/quentinll/lewm); `download.py` resolves the
individual repositories from the table below, which is kept only so the file names are checkable.

| task | repository | file | compressed → decompressed |
|---|---|---|---|
| PushT | `quentinll/lewm-pusht` | `pusht_expert_train.h5.zst` | 12.2 → 46 GB |
| Reacher | `quentinll/lewm-reacher` | `reacher.tar.zst` | 22.1 → 99 GB |
| Two-Room | `quentinll/lewm-tworooms` | `tworoom.tar.zst` | 3.2 → 13 GB |
| Cube | `quentinll/lewm-cube` | `cube_single_expert.tar.zst` | 43.0 → 102 GB |

`download.py --tier datasets` fetches all six sources (these four plus MAZE and PushObj) and
decompresses them into the layout the configs expect: `pusht_expert_train.h5`, `dmc/reacher.h5`,
`tworoom.h5`, `ogbench/cube_single_expert.h5`, `maze2d_medium.h5`, `pushobj_*.h5`.

## Probes are instruments, not data

Probes are rebuilt deterministically (`viscore probe`, `rng(0)`, 300 episodes, frameskip 5) rather
than shipped with pixels. A probe built with a different seed or episode count is a **different
measurement instrument**, and scores taken against it are not comparable to published ones. The
pixel-free probes in `bundle/probes/` carry everything the factors read, which is why they are ~1 MB.

One content difference from the research caches: `viscore probe` writes each task's ground-truth
state into the probe, including Cube's `block_pos` / `effector_pos`, which the original Cube probe
omitted and the scorer re-read from the source HDF5. Episode sampling is unchanged, so latent banks
are identical.

## The planner experiment lives in its own folder

`reproduce/planner/` reproduces `tab:planner-transfer` from `planner_cells.csv`: 36 checkpoints, the
planner-independent metrics, one sobriety per probe family and one success rate per planner. It is a
separate pool by necessity -- every arm must score the SAME checkpoints for the comparison to be
paired -- so its numbers are not comparable to the pools above. All 25 of its runs appear in those
pools and 20 of its 36 checkpoints are cells of `development` or `heldout`; the other 16 are further
epochs of the same runs. A refresh on the full 103-checkpoint held-out pool is in progress and is not
published here, because it is not what the paper reports.
