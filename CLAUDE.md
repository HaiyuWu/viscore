# CLAUDE.md — operating instructions for agents working in this repo

This repo computes **VIScore**, an offline diagnostic of a latent world model's fitness for
search-based planning:

```
VIS = erf((d_tol/2)/(√2·σ_roll)) · min(m_emp/τ, 1) · (1 − p̂)
       veracity                    influence          sobriety      τ = 82 nats
```

No environment, no planner rollouts, no success labels. ~7 s/checkpoint on an H100 against ≥20×
that for one 50-episode planning evaluation.

Two independent halves. **Do not mix their dependencies.**

| | what | dependencies |
|---|---|---|
| `viscore/` | the metric | numpy, scipy, torch. **Never** import `stable_worldmodel` here |
| `viswm/` | VIS-WM / LeWM training + planning eval | pinned `stable-worldmodel==0.0.6` stack |

---

## 1. Invoke

```bash
pip install -e .                      # metric only
pip install -e ".[data]"              # + h5py/hdf5plugin, needed to BUILD a probe
pip install -e ".[dev]" && pytest tests/     # 9 checks, no GPU/data/checkpoint needed
```

```bash
viscore tasks                       # task registry: tolerances + required probe keys
viscore probe --tasks pusht --data-home $VISCORE_HOME --out-dir cache/probes
viscore score --task pusht --probe cache/probes/probe_pusht.npz \
                --run-dir $RUNS/my_run --epochs 1-10 \
                --latents-dir cache/latents --csv out.csv
```

```python
from viscore import score_checkpoint
f = score_checkpoint("vis-wm/pusht/seed403/vis-wm_epoch_9.ckpt", "probes/probe_pusht.npz", "pusht")
f.vis, f.veracity, f.influence, f.sobriety, f.factors, f.status, f.saturated
```

Machine-readable: `--json` (one record per line, same fields as the CSV) and `--json-out FILE`.
Exit codes: **0** all scored · **2** usage error · **3** at least one cell abstained.

`--probe` also accepts a directory (uses `probe_<task>.npz`) and will build the probe itself if
you additionally pass `--data-home`.

## 1b. Published artifacts

| what | where | size |
|---|---|---|
| checkpoints (VIS-WM arms + `tab:viscore` pools, by pool) | `BooBooWu/viscore` (HF model) | 38.8 GiB |
| MAZE + PushObj datasets, CPU-reproduction bundle | `BooBooWu/viscore` (HF dataset) | 4.9 GiB |
| the four base datasets | `quentinll/lewm-{pusht,reacher,tworooms,cube}` — **not re-hosted here** | 80.5 GiB compressed |

```bash
python reproduce/tables.py                                          # tab:viscore, no GPU/data needed
python reproduce/download.py --tier bundle --dest $VISCORE_HOME    # 3.8 GiB, recompute factors on CPU
python reproduce/download.py --tier vis-wm|pools|datasets --dest $VISCORE_HOME
```

Pool membership lives in `reproduce/pools/pool_manifest.csv`; the run-level fold split in
`pool_assignment.csv`. The split is by **training run**, not evaluation seed — never re-derive pools
by seed, and never move a run between folds.

## 2. Decision table

| user asks | do |
|---|---|
| "score this checkpoint" | `viscore score --task T --probe P --ckpt C` |
| "which epoch is best?" | `--run-dir R --epochs 1-10 --csv out.csv`, then rank by `vis`; **ties go to the LATER epoch** |
| "why is this model bad?" | report the three factors and `saturated`, not `vis` alone. The smallest non-saturated factor names the defect |
| "compare two checkpoints" | same probe, same task, same `--horizon`, same `--tau`. Otherwise the numbers are not comparable |
| "score a non-LeWM model" | write an adapter (3 methods) — `examples/custom_adapter.py`; do **not** modify `veracity.py`/`influence.py`/`sobriety.py` |
| "add a task" | add a `TaskSpec` in `viscore/tasks.py` with the tolerance **read from the environment source** |
| "it's an IDM / behaviour-cloning / amortized policy" | see §3. `--exclude sobriety` computes; it does **not** validate |
| "d = 50 / 75 instead of 25" | `--horizon 10` / `--horizon 15` (goal offset ÷ frameskip 5) |
| "make the score higher" | **refuse**. Constants are frozen; see §4 |
| "train a model" | `viswm/`, see README "Training VIS-WM"; check `squeue` before submitting anything |
| "reproduce the paper's table" | metric table: `python reproduce/tables.py`; planning tables: `python reproduce/planning_tables.py` — do **not** recompute pools or re-run evals by hand |
| "get the checkpoints / data" | `reproduce/download.py`; MAZE has **one** training seed, not three |

## 3. Factor subsets (`--exclude`)

Any subset of veracity / influence / sobriety. Accepts full names or initials:
`--exclude sobriety`, `--exclude s`, `--exclude v s`.

* The result is the **product of the remaining factors**, labelled in `factors` (`VI`, `IS`, `I`…).
  Excluded factors are `nan`, never a silent 1.0.
* **A partial score is not VIScore.** Never compare a `VI` number with a published `VIS`
  number, or with a `VS` number. Always report which factors were included.
* Excluding all three is an error, not an empty product.

**When to exclude, and what is actually known:**

| case | do | why |
|---|---|---|
| amortized / GC-IDM / BC policy | `--exclude sobriety` | sobriety has **no estimand** — the policy never runs a search, so there is no "search beats the expert" event to measure. Measured further: amortized success was **independent of rollout fidelity**, so veracity does not track it either. The honest reading is representation content (`--baselines` probe R²), and *any* subset here is a description of the model, **not** a validated predictor of that policy's success |
| discrete-mode task (e.g. Cube's grasp) | keep all three, report `saturated` | influence reads exactly 1.0 (median 236 nats vs τ=82), so ⅓ of the product carries no information. Report it as a **missing axis**, not a verdict |
| cheap sweep / no GPU time | `--exclude sobriety` | sobriety is the 64×512-rollout probe and dominates runtime (measured on one checkpoint, CPU: 52 s → 19 s, i.e. ~⅔ of it) |
| you only trust the capacity axis | `--exclude v s` | fine, but `I` alone **fails the cross-task axis** (calibration 21.9 pts vs 10.2 for the product) and inverts on held-out methods (−0.16) |

Excluding **veracity does not save the rollout** when influence is kept: the noise floor
`Ê_H = σ_roll²·Ê/tr(Ê)` is anchored to `σ_roll`. Only the `d_tol` shell is skipped. This is
asserted in `tests/test_smoke.py`.

## 4. Hard rules

1. **Never re-tune the frozen constants to improve a result.** `τ = 82`, the `erf` map, the
   half-tolerance in veracity's numerator, the `(0.8, 1.2)·tol` shell band, the mini-CEM budget
   (128×4, top-16), `rng(0)` probe sampling. τ was selected once by two-fold cross-validation and
   frozen *before* the reported pools existed. Re-selecting it fits it to those pools.
2. **Never fill in an abstention.** `d_tol` is `nan` when no probe pair sits on the tolerance
   shell; `status == "abstained"` and exit code 3 are the results. Do not substitute a value, a
   mean, or a nearby checkpoint's number.
3. **Never compare across different probes.** A probe is a measurement instrument (frozen
   `rng(0)`, 300 episodes, frameskip 5). Different seed or episode count ⇒ different instrument.
4. **Never claim necessity.** Sufficiency holds (all factors high ⇒ the model is fine). Necessity
   is **falsified**: on one validation environment, veracity 0.58 → 1.00 and VIS 0.50 → 1.00
   moved success 52 → 50.
   So *do not* advise "fix the lowest factor to raise success".
5. **Never correlate a metric against success labels without checking label resolvability first.**
   If between-checkpoint success sd is below the labels' binomial SE (≈6 pts at `num_eval=50`),
   the correlation measures noise. This check invalidated two previously published correlations.
6. **Never upgrade `stable-worldmodel`.** 0.0.6 is pinned; 0.1.x changes the data backend and
   several evaluation-protocol behaviours that published absolute numbers depend on. See
   `docs/SWM_MIGRATION.md` before touching it.
7. **Never invent a tolerance.** Read it from the environment's success criterion and cite the
   source in the `TaskSpec`.
8. **Never quote an absolute success rate without its do-nothing floor.** `eval.py policy=zero`.
   Measured floors span 0–74% across environments; random is not a substitute.
9. **Never report a single evaluation seed as evidence of a gap.** True σ ≈ 6 pts at
   `num_eval=50`; most single-seed gaps in this literature vanish under 3–4 seeds.
10. **Never commit or push** unless the user explicitly asks.

## 5. Reading the output

| field | meaning | action |
|---|---|---|
| `vis` | product over the **included** factors | rank checkpoints with it |
| `factors` | `VIS` / `VI` / `I` … | quote it whenever it is not `VIS` |
| `status` | `ok` / `abstained` | `abstained` ⇒ an included factor is `nan`; report, do not patch |
| `saturated` | included factors ≥ 1 − 1e-6 | those factors are not separating anything on this task |
| `m_emp` (nats) | raw capacity, pre-cap | above τ it is uninformative for ranking |
| `d_tol_grade` | latent image of the grading tolerance | `nan` ⇒ tolerance unreachable on this probe |
| `gap_resolution` | smallest gap the sobriety probe resolves | if comparable to the gaps, **average sobriety over CEM seeds** before ranking close checkpoints |
| `p_hat` | fraction of anchors the search beat | `sobriety = 1 − p̂` |

Sobriety is deterministic per `cell_id` but not across CEM seeds: on a flat-landscape Two-Room
checkpoint (25–27 of 64 anchors below `gap_resolution`), four seeds gave 0.64/0.75/0.64/0.75.

## 6. Out of scope — say so rather than improvising

* **Amortized / inverse-dynamics / behaviour-cloning policies** — §3.
* **Discrete-mode success** (grasp, contact-mode switches): no factor represents a discrete
  transition.
* **Non-pixel observation spaces**: probes store `pixels`; a state-input model needs an adapter
  whose `encode` ignores them, and the tolerance-shell logic has not been validated there.
* **Ranking two *architectures* by absolute VIS across different latent dimensions**: the factors
  are scale-invariant by construction (asserted in tests) but this has only been validated across
  checkpoints, batch sizes, regularizers, planners, one held-out dataset and a held-out method
  family — not across arbitrary architectures.
* **Predicting anything other than planning success.** It is not a loss, not a reward, not a
  training signal, and optimizing it directly is unvalidated.

## 7. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `KeyError: 'state'` etc. | probe lacks a key the `TaskSpec` reads | `viscore tasks` lists required keys; rebuild the probe |
| `d_tol` is `nan` (exit 3) | <30 probe pairs within `[0.8,1.2]×tol` | tolerance too small/large for this probe, or episodes too short. Do not widen the band to make it pass |
| `no .encode found in <type>` | checkpoint is not a full-object pickle | it may be a state dict; add a loader in `adapters.py` |
| `ModuleNotFoundError: jepa` | LeWM pickle needs top-level module names | `viscore.register_lewm_modules()` (automatic in `LeWMAdapter.load`) |
| wrong `history_size` | auto-detect read `predictor.pos_embedding` | pass `--history-size` |
| CUDA OOM in sobriety | `B·n` candidates in one batch | lower `viscore.sobriety.CEM_N` and **say so** — it changes the probe |
| `NotImplementedError: data.format='lance'` | training side is HDF5-only on swm 0.0.6 | `docs/SWM_MIGRATION.md` |
| training loss diverges (`validate/pred_loss` jumps orders of magnitude) | learning rate is **not portable** across regularizers/datasets | see README; do not conclude "the model cannot learn this task" |

## 8. Authority

`docs/METRIC.md` is the specification, including §5 (what was verified: algebra reproduces the
published cache to 4.4e-15) and §4 (limitations, each with its evidence). `docs/DESIGN.md` explains
the package split, `docs/SWM_MIGRATION.md` the pinned stack, `reproduce/` the published numbers and
artifacts. If this file and the paper disagree, the paper wins — then fix this file.
