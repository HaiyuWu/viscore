# Provenance of `viswm/`

`jepa.py`, `module.py`, `utils.py`, `train.py`, `eval.py` and the Hydra configs derive from the
**LeWorldModel** release (Maes et al., arXiv:2603.19312), MIT licensed, and are modified here. If
you want the unmodified upstream, use the LeWorldModel repository — not this directory.

## What this work adds

* **`module.VISReg`** — the Variance-Invariance-Sketching regularizer, ported to the SIGReg call
  signature `(T, B, D)` so it drops into LeWM's training loop unchanged. Numerics are synced with
  the official VISReg implementation (`github.com/HaiyuWu/visreg`): `clamp_min` on the std rather
  than an additive epsilon, the Gaussian quantile target kept in float32 (casting it under bf16
  quantizes the quantiles and biases the shape term), and the official summation order, since bf16
  addition is not associative. Verified numerically identical to the official module in fp32 and
  bf16 at batch 128/256/512. The local `scale_detach` flag defaults to the official behavior.
* **`config/train/loss/visreg.yaml`** — VISReg with independently weighted center / scale / shape
  terms. The default weight λ = 4.5 is chosen to match SIGReg's *effective* regularizer-to-
  prediction contribution (SIGReg λ=0.09 × ⟨SIGReg⟩≈2.3 ≈ VISReg λ=4.5 × ⟨VISReg⟩≈0.044), so the
  comparison is not confounded by loss scale.
* **CLS-reconstruction ablation hooks** in `jepa.py` / `train.py` / `masking.py` (a masked-CLS or
  patch-token invariance loss). Off by default, and checkpoints from this arm are **excluded from
  every reported metric pool**.
* **`eval.ZeroPolicy`** (`policy=zero`) — the do-nothing protocol floor. Not interchangeable with
  `policy=random` and not bounded by it; see the class docstring and
  [../reproduce/RESULTS.md §5](../reproduce/RESULTS.md).

## What was removed for this release

Present in the research fork, dropped here because the supporting files are not part of this
release and half-wired code is worse than none:

* dispatch to externally released third-party checkpoints (a per-source loader for other groups'
  world models, used for the cross-method pool);
* environment registrations for validation environments outside the four published tasks;
* the HDF5→Lance dataloading experiment. `train.py` now raises `NotImplementedError` for
  `data.format != "hdf5"` rather than silently ignoring the key — see
  [../docs/SWM_MIGRATION.md](../docs/SWM_MIGRATION.md).

## Layout note

The flat module layout (top-level `jepa`, `module`, …) is upstream's and is kept deliberately: a
released checkpoint is a `torch.save` pickle that names `jepa.JEPA` and `module.ARPredictor` as
**top-level** modules, so packaging them differently would break loading every published
checkpoint. `viscore.adapters.register_lewm_modules` binds these files under those names when
loading a checkpoint from outside this directory.