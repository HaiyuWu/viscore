# Pools in `pool_manifest.csv`

| Flag | Label column | Cells / runs | Reproduced by |
|---|---|---|---|
| `in_development` | `sr_development` | 137 / 14 | `tables.py` |
| `in_terminal` | `sr_terminal` | 76 / 17 | `tables.py` |
| `in_heldout` | `sr_heldout` | 56 / 38 | `tables.py` |
| `in_heldout_dataset` | `sr_heldout_dataset` | 20 / 2 | **not yet** — see below |

The first three are the blocks `tables.py` prints, and every published value in them is reproduced.

## Held-out dataset (MAZE): cells shipped, table not yet reproduced

The 20 MAZE cells are in the manifest and their checkpoints are published under
`pools/heldout-dataset/` on the Hub, but `tables.py` does **not** emit this block yet: recomputing
it from these columns gives the paper's ranking and calibration for veracity, influence and raw
empowerment, and disagrees for VIScore and sobriety (ρ +0.85 / +0.82 here against +0.88 / +0.88
reported). Since VIS is the product of the three factors, the discrepancy is localised to the
sobriety column: the paper's MAZE analysis computes sobriety with its own probe, which is not the
value stored here.

Rather than publish a block that contradicts the paper, the sobriety column has to be recomputed
for these cells before the block is enabled. The checkpoints, the pixel-free MAZE probe and the
metric code needed for that are all published; `viscore.sobriety.measure` is the entry point.

## Held-out method

Third-party checkpoints (Qantara, RC-aux, INTACT) cannot be redistributed here; fetch them from
their own releases with `reproduce/download_external.py`. The DINO-CLS variant in that pool is
ours and is published under `pools/heldout-method/`.
