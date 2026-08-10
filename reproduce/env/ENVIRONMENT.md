# Environment

Two environments, on purpose. The metric needs almost nothing; the training and planning-eval
side needs a pinned stack.

## A. Metric only (`viscore/`)

```bash
pip install -e .                # numpy, scipy, torch
pip install -e ".[data]"        # + h5py, hdf5plugin — only to BUILD a probe from HDF5
```

Any recent Python ≥ 3.10 and torch ≥ 2.0 works. There is no swm/lightning/hydra dependency, so a
laptop can recompute every score from a released probe + latent cache. `pytest tests/` needs
neither GPU nor data.

## B. Training + planning evaluation (`viswm/`)

Load-bearing pins — **do not float these**:

```
python==3.10
stable-worldmodel==0.0.6      # HDF5 data backend. 0.1.x switches to Lance: see docs/SWM_MIGRATION.md
stable-pretraining==0.1.6     # spt.Module / spt.Manager training loop
torch==2.11.0+cu128
torchvision==0.26.0+cu128
lightning==2.6.1
hydra-core==1.3.2
transformers==5.4.0           # ViT backbone via spt.backbone.utils.vit_hf
einops==0.8.2
h5py==3.16.0
hdf5plugin==6.0.0             # registers the pixel compression filters; import is required
numpy==2.2.6
scipy==1.15.3
scikit-learn==1.7.2
wandb==0.25.1                 # optional: wandb.enabled=false to skip
```

Environment/task extras, per task:

```
pymunk==7.2.0  shapely==2.1.2          # PushT
dm_control==1.0.38  dm-env==1.6        # Reacher (DeepMind Control)
ogbench==1.2.1                          # Cube
mujoco==3.6.0  glfw==2.10.0             # rendering (eval.py sets MUJOCO_GL=egl)
gymnasium==1.2.3  gymnasium-robotics==1.4.2
```

`environment-frozen.txt` in this directory is the complete 225-package resolution of the
environment the published numbers were produced in. Regenerate it from any environment with:

```bash
python -c "import importlib.metadata as m; \
print('\n'.join(sorted(f\"{(d.metadata['Name'] or '?').lower()}=={d.version}\" \
for d in m.distributions())))" > environment-frozen.txt
```

### Why `stable-worldmodel` is pinned

0.1.x is not a drop-in: it replaces the HDF5 data backend with Lance, which orphans every `.h5`
artifact and every `config/train/data/*.yaml` that names one, and it does not resolve the pinned
CUDA wheel. Beyond packaging, several **evaluation-protocol behaviors** of 0.0.6 affect absolute
success rates and must be re-measured before numbers from two versions are compared — the list,
with the specific traps, is in [../../docs/SWM_MIGRATION.md](../../docs/SWM_MIGRATION.md).

### Storage

Set `VISCORE_HOME` explicitly, always. It is the artifact root: datasets, checkpoints and run
directories all live under it. The scripts export it as `STABLEWM_HOME` before calling into
stable-worldmodel, which is the name that library resolves its cache from; its own default is
`~/.stable_worldmodel`, which is not where anything here lives.

```bash
export VISCORE_HOME=/path/with/room/for/datasets_and_checkpoints
```

Datasets are large (25–84 GB each, see [../ARTIFACTS.md](../ARTIFACTS.md)) and training is
**I/O-bound, not compute-bound** when they sit on network storage: GPU utilization near zero,
~1 h/epoch on PushT. Two operational consequences, both learned the hard way:

* **Host RAM scales super-linearly with `loader.num_workers`.** 16 workers at batch 512 peaked at
  188 GB and was killed by the host cgroup with no CUDA traceback. Budget ≥ 320 GB for 16 workers,
  or use 4 workers / `prefetch_factor=1`. GPU memory is a separate axis.
* Concurrent training jobs share read bandwidth to the same dataset.
