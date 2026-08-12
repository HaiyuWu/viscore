<div align="center">

# VIScore: Diagnosing Planning-Relevant Quality in Latent World Models

[Haiyu Wu](https://haiyuwu.github.io/)<sup>1</sup> &emsp; [Randall Balestriero](https://randallbalestriero.github.io/)<sup>2</sup> &emsp; [Morgan Levine](https://www.altoslabs.com/team/morgan-levine)<sup>1</sup>

<sup>1</sup>Altos Labs &emsp; <sup>2</sup>Brown University

<a href='https://arxiv.org/abs/2608.11174'><img src='https://img.shields.io/badge/arXiv-2608.11174-b31b1b.svg'></a>
<a href='https://haiyuwu.github.io/viscore/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://huggingface.co/BooBooWu/viscore'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow'></a>
<a href='https://huggingface.co/datasets/BooBooWu/viscore'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-yellow'></a>
<a href='https://opensource.org/licenses/MIT'><img src='https://img.shields.io/badge/License-MIT-lightgrey'></a>

</div>

This is the official implementation of **VIScore**, an interpretable diagnostic of latent world-model quality for planning. Four stages decide whether a model plans successfully — the **vision encoder**, the **predictor**, the **planner**, and the **task tolerance** that grades the result — and existing metrics stop at the first two. VIScore covers all four, offline, from a single checkpoint.

```
                    d_tol / 2                  m_emp
VIS  =  erf( ───────────────────── )  ·  min( ─────── , 1 )  ·  ( 1 − p̂ )
                  √2 · σ_roll                    τ
        └────── veracity ──────┘        └── influence ──┘      └ sobriety ┘

  veracity   does an open-loop rollout stay inside the ball the task grades on?
  influence  do actions command enough distinguishable futures — for THIS task?
  sobriety   how often does the model's preferred plan hold up against a known-good one?
```

&emsp;🔥 **High correlation with success**: pooled ρ +0.90, where raw empowerment reaches +0.82 and reverses sign on converged checkpoints<br>
&emsp;💪 **Best calibration across planners and datasets**: the only metric below the constant-predictor reference on every pool<br>
&emsp;🚀 **20× faster than the real evaluation**: ~7 s per checkpoint, two orders of magnitude fewer rollout steps<br>
&emsp;🔍 **A diagnosis, not just a ranking**: three bounded factors, and the smallest one names the defect<br>
&emsp;🧩 **Any world model**: three methods (`encode`, `action_embed`, `predict_next`) are the whole interface<br>
&emsp;📦 **No framework lock-in**: the metric needs only numpy, scipy and torch — no training stack<br>
&emsp;♻️ **Reproducible on a CPU**: every table in the paper recomputes in seconds, with no GPU, dataset, or checkpoint<br>

**If you find VIScore useful for your research, please consider citing us 😄**
```bibtex
@article{wu2026viscore,
  title         = {VIScore: Diagnosing Planning-Relevant Quality in Latent World Models},
  author        = {Wu, Haiyu and Balestriero, Randall and Levine, Morgan},
  journal       = {arXiv preprint arXiv:2608.11174},
  year          = {2026},
  eprint        = {2608.11174},
  archivePrefix = {arXiv}
}
```

# News/Updates

- [2026/8] Success labels released — the planning tables are now replayable without re-running MPC.
- [2026/8] Checkpoints, datasets and the CPU-reproduction bundle added to [HuggingFace](https://huggingface.co/BooBooWu/viscore).
- [2026/8] Code created.

# Table of Contents

| | Section | | Section |
|---|---------|---|---------|
| :wrench: | [Installation](#wrench-installation) | :rocket: | [Score a checkpoint](#rocket-score-a-checkpoint) |
| :arrow_down: | [Download checkpoints](#arrow_down-download-checkpoints) | :scissors: | [Partial scores](#scissors-partial-scores-any-subset-of-v-i-s) |
| :floppy_disk: | [Download datasets](#floppy_disk-download-datasets) | :jigsaw: | [Score your own world model](#jigsaw-score-your-own-world-model) |
| :bar_chart: | [Reproduce the paper](#bar_chart-reproduce-the-paper) | :trophy: | [Results](#trophy-results) |
| :repeat: | [Train VIS-WM](#repeat-train-vis-wm) | :warning: | [What it does not claim](#warning-what-it-does-not-claim) |
| :file_folder: | [Repository layout](#file_folder-repository-layout) | :page_facing_up: | [License](#page_facing_up-license) |

# :wrench: Installation

**Requirements**: Python 3.10+, and a GPU only if you are scoring checkpoints from pixels.

```bash
git clone https://github.com/HaiyuWu/viscore.git
cd viscore

pip install -e .              # the metric: numpy, scipy, torch. Nothing else.
pip install -e ".[data]"      # + h5py/hdf5plugin, needed only to BUILD a probe cache
pip install -e ".[dev]"       # + pytest, to run tests/
```
The import name is `viscore`. The distribution is named `viscore-metric` because `viscore` on PyPI belongs to an unrelated project.


The metric deliberately does **not** depend on `stable-worldmodel`, `stable-pretraining`, lightning, or hydra. Those are training-side and pinned separately — see [reproduce/env/](reproduce/env/) and [docs/DESIGN.md](docs/DESIGN.md).

# :arrow_down: Download checkpoints

All checkpoints are on [HuggingFace](https://huggingface.co/BooBooWu/viscore). Each is a full-object `torch.save` pickle of `jepa.JEPA` (~15M parameters, 72 MB).

| Path | What | Size |
|------|------|------|
| `vis-wm/<task>/seed<S>/` | VIS-WM arms: PushT / Reacher / Two-Room / Cube × seeds 403/46/729, at the reported epoch | 0.9 GiB |
| `baselines-lewm/maze2d/seed729/` | the SIGReg (LeWM) counterpart on MAZE | 0.07 GiB |
| `pools/<pool>/<env>/<run>/` | every checkpoint `tab:viscore` is computed on, **organized by pool** | 37.8 GiB |

```bash
python reproduce/download.py --tier vis-wm --dest $VISCORE_HOME    # or --tier pools
```

**MAZE has one training seed (729), not three** — it is the held-out *dataset* validation, and that block of the paper contains two runs in total.

# :floppy_disk: Download datasets

The four base training datasets are LeWorldModel's and are **not** re-hosted here; the two this work created are:

| Dataset | Source | Size |
|---------|--------|------|
| PushT / Reacher / Two-Room / Cube | [LeWorldModel collection](https://huggingface.co/collections/quentinll/lewm) | 80.5 GiB compressed |
| **MAZE** (maze2d-medium) | [`BooBooWu/viscore`](https://huggingface.co/datasets/BooBooWu/viscore) | 600 MB |
| **PushObj** (7 shapes, OOD) | [`BooBooWu/viscore`](https://huggingface.co/datasets/BooBooWu/viscore) | 61–121 MB each |

```bash
export VISCORE_HOME=/path/with/room               # 268 GB decompressed for all six
python reproduce/download.py --tier datasets --dest $VISCORE_HOME
```

# :rocket: Score a checkpoint

A **probe** is one frozen slice of training data (`rng(0)`, 300 episodes, frameskip 5) shared by every checkpoint you compare. Build it once:

```bash
viscore probe --tasks pusht --data-home $VISCORE_HOME --out-dir cache/probes

viscore score --task pusht --probe cache/probes/probe_pusht.npz \
                --run-dir $VISCORE_HOME/my_run --epochs 1-10 \
                --latents-dir cache/latents --csv out.csv
```

```
seed403__vis-wm_epoch_9 (pusht)
  VIS       0.7213
  veracity  0.9014   d_tol 4.37 (pred req 2.19)  sigma_roll 1.79
  influence 1.0000   m_emp 147.3 / tau 82 nats     [saturated: influence]
  sobriety  0.8003   p_hat 0.200 (gap resolution 0.0031)
```

That `[saturated: influence]` line is the point: on this task the capacity factor is at its ceiling and carries no information, so only veracity and sobriety are separating checkpoints.

From Python:

```python
from viscore import score_checkpoint

f = score_checkpoint("vis-wm/pusht/seed403/vis-wm_epoch_9.ckpt",
                     "cache/probes/probe_pusht.npz", task="pusht")
print(f.vis, f.veracity, f.influence, f.sobriety, f.factors, f.status, f.saturated)
```

`viscore tasks` prints the task registry (tolerances, required probe keys); `--json` emits one machine-readable record per checkpoint. Exit codes: `0` scored, `2` usage error, `3` a cell abstained. Agents driving this repo should read [CLAUDE.md](CLAUDE.md).

# :scissors: Partial scores: any subset of V, I, S

Some models have no estimand for some factors. An amortized policy — goal-conditioned inverse dynamics, behaviour cloning — never runs a search, so **sobriety has nothing to measure**:

```bash
viscore score --task pusht --probe cache/probes/probe_pusht.npz --ckpt C --exclude sobriety
#   VI        0.9853  [saturated: influence]
#   excluded: sobriety -> this is VI, not VIS; do not compare it to VIS numbers
```

Full names or initials (`--exclude s`, `--exclude v s`); excluded factors come back as `nan`, never as a silent `1.0`, and the record carries `factors` (`VIS`/`VI`/`I`…) so a partial number cannot be mistaken for a full one. Two things to know:

* **A partial score is not VIScore** — different factor set, different scale. Never compare `VI` against a published `VIS`, or against `VS`.
* **Excluding sobriety saves real time** (it is the 64×512-rollout probe: one checkpoint on CPU went 52 s → 19 s). **Excluding veracity does not**, because influence's noise floor `Ê_H = σ_roll²·Ê/tr(Ê)` is anchored to `σ_roll` — only the `d_tol` shell is skipped.

Mechanically computable is not the same as validated: for amortized policies success was measured to be **independent of rollout fidelity** too, so treat any subset there as a description of the model rather than a predictor of that policy's success.

# :jigsaw: Score your own world model

VIScore needs exactly three operations, and nothing else:

```python
class MyWorldModel:
    device: torch.device
    history_size: int                              # predictor context length
    def encode(self, pixels_u8, chunk=128): ...    # (T,3,H,W) uint8 -> (T,D)
    def action_embed(self, actions): ...           # (B,T,A)      -> (B,T,D_a)
    def predict_next(self, z_ctx, a_ctx): ...      # (B,HS,D)     -> (B,D)
```

Implement those and pass the object to `viscore.score_latents`. Worked example: [examples/custom_adapter.py](examples/custom_adapter.py). Adding a **task** means declaring its success criterion (tolerance, the coordinates it grades, the coordinates it ignores) — see [viscore/tasks.py](viscore/tasks.py); every tolerance there is read off the environment source, never tuned against success labels.

# :bar_chart: Reproduce the paper

Both of the paper's main tables recompute from files that ship with this repo — **no GPU, no dataset, no checkpoint download**:

```bash
python reproduce/tables.py                     # tab:viscore, four pools x seven metrics
python reproduce/planning_tables.py --strict   # tab:ood + PushT main/long-horizon, 18/18 gated
python reproduce/planner/planner_tables.py     # tab:planner-transfer, six planners (its own pool)
```

The second one *gates* every published value and exits non-zero if any drifts. Everything else — the artifact inventory, what you need for which claim, the environment pins, the SLURM launchers — is in [reproduce/](reproduce/).

# :repeat: Train VIS-WM

`viswm/` is the world model the score was developed on: the LeWM architecture (ViT-Tiny encoder + action-conditioned AdaLN transformer predictor, ~15M parameters) trained end-to-end from pixels with a next-embedding prediction loss plus a Gaussian-latent regularizer — **VISReg** in place of LeWM's SIGReg.

```bash
cd viswm
export VISCORE_HOME=/path/to/datasets
python -u train.py loss=visreg data=pusht optimizer.lr=1e-4 seed=403     # VIS-WM
python -u train.py loss=sigreg data=pusht                                # LeWM baseline
python -u eval.py --config-name pusht policy=<run_subdir>/lewm_epoch_7_object.ckpt
```

Published recipe: 10 epochs, AdamW, linear warmup + cosine decay, batch 128, lr 1e-4, VISReg λ = 4.5 (the SIGReg baseline keeps the default LeWM configuration), three training seeds. **Learning rate is not portable across regularizers or datasets** — copying lr 1e-4 onto a new dataset made validation prediction loss diverge (0.076 → 409) in a way that reads as "the model cannot learn this task" but is purely an optimizer artifact.

This side is pinned to `stable-worldmodel==0.0.6` (HDF5 data backend); migrating to current swm is planned and the seams are documented in [docs/SWM_MIGRATION.md](docs/SWM_MIGRATION.md).

# :trophy: Results

## Association with planning success

Within-task Spearman ρ, pooled ρ, and cross-task calibration error in success points (leave-one-task-out isotonic fit; lower is better). The split is at the level of the **training run**: every constant is fitted on the development pool, and no checkpoint in a test pool comes from a run used to choose one. Cube is parenthesized and excluded from pooled/calibration — its label spread does not exceed its own binomial standard error.

| Metric | PushT | Reacher | Two-Room | Cube | Pooled | Calib. err. |
|--------|-------|---------|----------|------|--------|-------------|
| *development pool (137 ckpts, 14 runs)* | | | | | | |
| Straightness | +0.51 | +0.13 | −0.67 | (+0.01) | +0.33 | 40.4 |
| Physical-state probe (R²) | +0.41 | +0.74 | +0.57 | (−0.05) | +0.77 | 14.5 |
| Empowerment | +0.82 | +0.86 | +0.71 | (−0.18) | +0.40 | 23.1 |
| **VIScore** | +0.80 | +0.83 | +0.71 | (−0.29) | **+0.88** | **10.2** |
| *Ref.: constant predictor* | | | | | | *24.4* |
| *held-out checkpoints (103 ckpts, 33 runs)* | | | | | | |
| Straightness | +0.38 | −0.81 | −0.74 | — | −0.08 | 19.8 |
| Physical-state probe (R²) | +0.38 | +0.65 | +0.71 | — | +0.61 | 27.6 |
| Empowerment | +0.73 | +0.82 | +0.86 | — | +0.49 | 15.6 |
| **VIScore** | +0.84 | +0.72 | +0.83 | — | **+0.91** | **7.0** |
| *Ref.: constant predictor* | | | | | | *18.2* |
| *held-out methods (23 ckpts, 4 methods)* | | | | | | |
| Straightness | — | — | — | — | +0.22 | 17.4 |
| Physical-state probe (R²) | — | — | — | — | +0.32 | 16.1 |
| Empowerment | — | — | — | — | +0.12 | 12.0 |
| **VIScore** | — | — | — | — | **+0.75** | **8.3** |
| *Ref.: constant predictor* | | | | | | *11.3* |
| *held-out dataset — unseen MAZE task (20 ckpts, 2 runs)* | | | | | | |
| Straightness | — | — | — | — | +0.66 | 31.2 |
| Physical-state probe (R²) | — | — | — | — | +0.75 | 45.4 |
| Empowerment | — | — | — | — | +0.87 | **9.2** |
| **VIScore** | — | — | — | — | **+0.87** | 11.5 |
| *Ref.: constant predictor* | | | | | | *41.7* |

The instruments separate on the **cross-task** axis, not within tasks: raw empowerment matches VIScore inside every task of the held-out pool and is the better ranker on two of the three, but it is measured in nats whose meaning does not transfer — pooled +0.49 at 15.6 points, against +0.91 at 7.0. The last two pools are scored by the calibration map already fitted on development, with no refit, because a new task or a new method arrives without labels; they carry too few checkpoints per task for a within-task column.

## Which factor binds, and where

Pooled ρ / calibration error per pool. No single factor dominates every shift, and each one *fails* somewhere — which is the argument for reading the factors individually rather than only their product.

| Pool (n / runs) | VIS = V·I·S | V veracity | I influence | S sobriety | *Ref.* |
|---|---|---|---|---|---|
| development (137/14) | +0.88 / **10.2** | +0.80 / 12.4 | +0.54 / 21.9 | **+0.89** / **9.3** | *24.4* |
| held-out ckpt (103/33) | **+0.91** / **7.0** | +0.81 / 12.7 | +0.65 / 14.0 | +0.89 / 7.3 | *18.2* |
| held-out method (23/23) | **+0.75** / **8.3** | +0.62 / 10.4 | +0.09 / 12.2 | +0.68 / 8.9 | *11.3* |
| held-out dataset (20/2) | +0.87 / 11.5 | +0.85 / 31.9 | **+0.87** / 9.4 | **+0.88** / **8.3** | *41.7* |

## Not an artifact of CEM

Keeping the checkpoint, goals, episode count and planning cost fixed and replacing only the optimizer, VIScore keeps **positive within-task association in all 18 planner × task cells** (+0.29 to +0.91, median +0.67), and it does *not* decay as the planner weakens (rank correlation between a planner's success deficit against CEM and its within-task ρ: +0.07).

## Out-of-distribution planning (PushObj)

Success on six unseen object shapes, peak epoch, 3 training × 3 evaluation seeds. The matched-batch control is the point: **at B=128 the two objectives are indistinguishable**, and the top-block advantage comes from the batch ladder.

| Model | d=25 | d=50 | d=75 |
|-------|------|------|------|
| LeWM (SIGReg, B=128) | 79.1 ±0.9 | 61.8 ±1.1 | 44.3 ±1.2 |
| **VIS-WM (VISReg, B=512)** | **82.0** ±0.7 | **62.0** ±1.1 | **45.6** ±1.1 |
| VIS-WM, B=128 (matched control) | 77.4 ±1.0 | 59.7 ±1.3 | 41.8 ±1.3 |
| VIS-WM, B=256 | 82.0 ±0.7 | 62.2 ±1.2 | 44.3 ±1.2 |
| Random policy | 10.1 | 7.3 | 3.0 |

# :warning: What it does not claim

Measured boundaries, not caveats added for safety:

* **Sufficiency yes, necessity no.** All factors high implies the model is fine; a low factor does *not* imply that fixing it raises success (on one validation environment, veracity 0.58 → 1.00 and VIS 0.50 → 1.00 moved success 52 → 50).
* **Search-based planning only.** An amortized policy has no sobriety estimand, and its success was measured to be independent of rollout fidelity.
* **Discrete-mode success is out of scope.** Cube succeeds only if the gripper enters a closed mode on the block; influence reads exactly 1 on every Cube checkpoint.
* **Single-seed evaluations cannot resolve the differences this literature reports.** True success sd ≈ 6 points at `num_eval=50`.

Each of these is stated with its evidence in [docs/METRIC.md](docs/METRIC.md#4-limitations).

# :file_folder: Repository layout

```
viscore/            the metric. Pure numpy/scipy/torch, no training framework.
  tasks.py           success criteria -> tolerances, scored + nuisance coordinates
  probe.py           build the frozen probe cache from an HDF5 dataset
  veracity.py        rollout NMSE, d_tol on the tolerance shell, erf
  influence.py       action-induced S / residual E spectra, m_emp, the tau cap
  sobriety.py        mini-CEM optimism probe -> p_hat
  score.py           the product, the factor record, CSV/JSON output
  baselines.py       straightness + physical-state probe R^2 (what VIScore is measured against)
  adapters.py        the model API seam; LeWMAdapter for this family
viswm/               VIS-WM training: LeWM architecture + VISReg, Hydra configs, planning eval
reproduce/           tables.py, planning_tables.py, artifacts, environment pins, launchers
docs/                METRIC.md (full spec), DESIGN.md, SWM_MIGRATION.md
tests/               invariance tests: latent-rescaling, linear-reparameterization, factor subsets
CLAUDE.md            operating contract for agents: how to call it, what it does not support
```

# :link: Referenced GitHub repositories

- [LeWorldModel](https://github.com/lucas-maes/le-wm)
- [VISReg](https://github.com/HaiyuWu/visreg)
- [SIGReg/LeJEPA](https://github.com/galilai-group/lejepa)

# :page_facing_up: License

[LICENSE](LICENSE)
