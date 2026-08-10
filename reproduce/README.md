# Reproducing the paper

Everything version-, artifact-, and site-specific lives here, so the packages above it stay
portable. **All artifacts are published** — see [ARTIFACTS.md](ARTIFACTS.md) for the inventory.

| | |
|---|---|
| [tables.py](tables.py) | **recomputes `tab:viscore`** from the published manifest — no GPU, no checkpoints, no dataset |
| [planning_tables.py](planning_tables.py) | **recomputes the Section-4 planning tables** (`tab:ood`, PushT main + long-horizon) from the published success labels, with a pass/fail gate on every published value |
| [pools/success_labels.csv](pools/success_labels.csv) | every planning evaluation: (checkpoint, task, goal offset, evaluation seed) → success rate, 2753 rows |
| [pools/planning_arms.csv](pools/planning_arms.csv) | the epoch each reported Section-4 arm was taken at |
| [pools/pool_manifest.csv](pools/pool_manifest.csv) | the definition of the paper's pools: per checkpoint, membership + success labels + all seven metric values |
| [pools/pool_assignment.csv](pools/pool_assignment.csv) | the run-level development / test fold split (the leakage audit object) |
| [download.py](download.py) | fetches checkpoints / bundle / datasets from the Hub |
| [download_external.py](download_external.py) | fetches the third-party checkpoints (Qantara, RC-aux, INTACT, DINO-WM) from their own releases — they are not re-hosted here |
| [pools/README_POOLS.md](pools/README_POOLS.md) | what each pool flag means, and which blocks `tables.py` reproduces |
| [ARTIFACTS.md](ARTIFACTS.md) | what is published where, and what you need for which claim |
| [env/ENVIRONMENT.md](env/ENVIRONMENT.md) | the two environments and why `stable-worldmodel` is pinned |
| [env/environment-frozen.txt](env/environment-frozen.txt) | the complete 225-package resolution behind the published numbers |
| [slurm/](slurm/) | `train.slurm`, `eval.slurm` (the expensive labels), `score.slurm` (the cheap metric) |
| [RESULTS.md](RESULTS.md) | the published numbers with the command that regenerates each |

## Three depths of reproduction

The tables reproduce from files in this repository; recomputing what goes into them needs more.

| depth | what it re-derives | what you need | held-out-method coverage |
|---|---|---|---|
| 1 | every published table, from frozen metric values and success labels | this repository only | **23/23** |
| 2 | the metric itself, from cached latents / spectra / sobriety gaps | `download.py --tier bundle` (3.8 GiB) | 15/23 |
| 3 | the metric from the weights, end to end | `download.py --tier pools` (35.9 GiB) and, for the method pool, `download_external.py` | 20/23 are other groups' releases |

Depth 1 is `python reproduce/tables.py`, and it is complete: the held-out-method block is computed
from `pools/heldout_method_cells.csv` (metrics, frozen because scoring them needs each source's own
code) and `pools/heldout_method_labels.csv` (one row per checkpoint and evaluation seed, so our
averages are checkable rather than asserted).

Depth 3 exists for anyone who wants to re-score the weights instead of trusting our numbers:

```bash
python reproduce/download_external.py --dest $VISCORE_HOME/external --with-code
```

That fetches Qantara, RC-aux and INTACT from their own releases, unpacks INTACT's tarballs,
verifies every release that ships a `SHA256SUMS`, and clones the matching code repository next to
each download. The code checkout is not optional: each release ships its own fork of the LeWM tree
with the same class names, so unpickling one against our classes binds the weights to different
code and produces wrong numbers without raising. DINO-WM is listed but hosted on OSF, which has no
API client here, so that one is a manual download.

The three DINO-CLS-WM checkpoints in the same pool are ours and are published under
`pools/heldout-method/` in the model repository.

## If you are building a better metric

Everything a competing diagnostic needs is published, so a comparison can be made on identical
data rather than on re-run evaluations:

| what you need | where it is |
|---|---|
| checkpoints with success labels, to score your own metric on | 472 calibration-fit + 103 held-out + 20 MAZE + 16 off-split, under `pools/` in the model repository |
| the labels themselves, per evaluation seed | `pools/success_labels.csv` (3104 evaluations, every one CEM at d = 25, 50 episodes) |
| the pool definitions and the run-level split | `pools/pool_manifest.csv`, `pools/pool_assignment.csv` |
| the third-party methods of the held-out-method pool | `download_external.py --with-code` |
| the two calibration protocols, implemented | `tables.py`: leave-one-task-out for the first two pools, the frozen development map for the transfer pools |

The scoring pool is deliberately larger than the pools the paper reports. `in_calibration_fit`
marks 472 cells that carry a development success label; only 137 of them survive the run-level
split into the reported development pool, and 103 more into held-out. The rest exist because the
frozen calibration map is fitted on all 472, and they are published because they double the
labelled pool a new metric can be measured on.

Two conventions worth matching if you want numbers comparable to ours. Success labels average
three evaluation seeds, and a single 50-episode evaluation has a success sd of about 6 points, so
check that the between-checkpoint spread of your target pool exceeds that before reading any
correlation. And calibration error is only interpretable next to its own pool's constant-predictor
reference, since the pools span different success ranges.

To turn a score into a predicted success rate under the published calibration:

```python
from viscore import predict_success, fit_pool_size
predict_success(0.72)     # success points, from the map fitted on fit_pool_size() == 472 cells
```

## The 60-second path

```bash
git clone https://github.com/HaiyuWu/viscore && cd viscore && pip install -e .
python reproduce/tables.py
```

That prints every cell of `tab:viscore` — three pools × seven metrics, within-task ρ, pooled ρ,
leave-one-task-out calibration, and each pool's constant-predictor reference — from
`pools/pool_manifest.csv`, which ships with the repo. Add `--boot 2000` for run-clustered bootstrap
intervals, `--latex` for the paper's rows.

```bash
python reproduce/planning_tables.py --strict
```

That does the same for Section 4 — `tab:ood` and the PushT main / long-horizon rows — from
`pools/success_labels.csv`, and **gates every published value**: all 18 currently reproduce
(12 OOD cells, 6 PushT cells). Without those labels the same numbers would cost O(100–500)
GPU-hours of 50-episode MPC to re-measure.

## Paths from an empty machine

**Check the metric without a GPU** (3.8 GiB): the bundle carries the latents, the `(S, E)` spectra,
the per-anchor sobriety gaps and pixel-free probes for all 264 pool checkpoints, so all three factors
can be recomputed as pure algebra.

```bash
python reproduce/download.py --tier bundle --dest $VISCORE_HOME
```

**Score the released checkpoints from pixels** (0.9–35 GiB of checkpoints + one base dataset):

```bash
python reproduce/download.py --tier vis-wm --dest $VISCORE_HOME     # or --tier pools
python reproduce/download.py --tier datasets --dest $VISCORE_HOME   # 268 GB decompressed
viscore probe --tasks pusht --data-home $VISCORE_HOME --out-dir $VISCORE_HOME/probes
viscore score --task pusht --probe $VISCORE_HOME/probes/probe_pusht.npz \
                --run-dir $VISCORE_HOME/published/vis-wm/pusht/seed403 --epochs 1-10 --csv out.csv
```

**Reproduce a correlation number from scratch**: the metric side is `slurm/score.slurm`, the success
labels are `slurm/eval.slurm`. The labels are the expensive half by two orders of magnitude, which is
the reason this metric exists.

**Retrain from scratch**: base datasets + `slurm/train.slurm`. Budget ~1 h/epoch/GPU on PushT,
dominated by dataset I/O rather than compute; read the host-RAM warning in
[env/ENVIRONMENT.md](env/ENVIRONMENT.md#storage) before raising `num_workers`.

## Reproducibility notes that are easy to get wrong

* **The pools are split by training run, not by evaluation seed.** Every constant is fitted on the
  `development` pool; `terminal` and `heldout` contain no checkpoint from a development run.
  Re-deriving the pools yourself means honouring `pool_assignment.csv` — a run absent from it never
  carried a development label and is test-side by construction.
* **Probes are frozen instruments** (`rng(0)`, 300 episodes, frameskip 5). A probe rebuilt with a
  different seed or episode count gives scores that are not comparable to published ones. If you
  change it, say so.
* **Cube is parenthesized and excluded from pooled/calibration** — its label spread does not exceed
  its own binomial standard error, so nothing there is rankable by any metric.
* **A partial score (`--exclude`) is not VIScore.** Different factor set, different scale.
* **`EPOCHS` cannot go through `sbatch --export`** — sbatch splits that option on commas, so
  `--export=ALL,EPOCHS=1,2,3` silently sets `EPOCHS=1`. Export it as a shell variable; both array
  scripts document this and have an index-based fallback.
* **`EVAL_CONFIG` has no default** in `slurm/eval.slurm`, on purpose: the original had one and it
  silently evaluated the wrong task.
* **Don't re-run a `(checkpoint, config, seed)` triple that already has a log.** Duplicate
  evaluations of the same triple are the easiest way to fabricate a "replication".
* **Quote a do-nothing floor with any absolute success rate** (`eval.py policy=zero`). Measured
  floors span 0–74% across the environments used here; random is not a substitute.
