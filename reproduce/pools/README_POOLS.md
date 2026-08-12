# Pools in `pool_manifest.csv`

Select cells with the `in_*` flags; each has its own label column, because a cell can carry
labels from more than one protocol.

| Flag | Label column | Cells / runs | What it produces |
|---|---|---|---|
| `in_development` | `sr_development` | 137 / 14 | block 1 of the main table; every constant is fitted here |
| `in_heldout` | `sr_heldout` | 103 / 33 | block 2 -- the headline test, runs disjoint from development |
| `in_heldout_method` | `sr_heldout_method` | 30 / 30 | block 3 -- one checkpoint per world-modeling method |
| `in_heldout_dataset` | `sr_heldout_dataset` | 20 / 2 | block 4 -- the unseen MAZE task |
| `in_cube_reference` | `sr_heldout` | 50 / 5 | the parenthesised Cube column only |
| `in_calibration_fit` | `sr_development` | 472 / 47 | no reported number: fits the frozen map blocks 3-4 apply |

All 572 cells carry a label, and `tables.py` reproduces all four printed blocks.

`fold` is a property of the training **run** -- which side of the run-level split it fell on
(`dev`, `heldout`, or `unsplit` for the Cube runs, which entered no fold). The `in_*` flags are
properties of the **cell**. They are deliberately different sets: 335 cells belong to held-out runs,
but only 103 are the reported held-out pool, because that pool takes one epoch ladder per run under
the three fresh evaluation seeds. Select with `in_*`; audit independence with `fold`.

## Cube

`in_cube_reference` marks all 50 Cube cells. Cube entered no fold: its label spread does not exceed
its own binomial standard error, so nothing there is rankable by any metric. The paper prints its
correlation in parentheses and excludes it from every pooled and calibration number. The cells are
published because that exclusion argument is measured on them.

## Calibration

`in_calibration_fit` marks 472 cells on the three fitted tasks, and `in_development` (137) and
`in_heldout` (103) are subsets of it. The first two blocks are calibrated leave-one-task-out within
themselves; blocks 3 and 4 apply an isotonic map fitted once on all 472 and never refitted, because
a new method or a new task arrives without labels. `viscore.predict_success` is that map.

## Held-out method

Third-party checkpoints (Qantara, RC-aux, INTACT) cannot be redistributed here; fetch them from
their own releases with `reproduce/download_external.py`. The DINO-CLS variant in that pool is ours
and is published under `pools/heldout-method/` on the Hub.
