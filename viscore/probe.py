"""Build a frozen probe: the fixed slice of data every checkpoint is scored on.

A probe is one .npz holding, for a fixed sample of episodes at the model's temporal cadence:

    pixels      (F, 224, 224, 3) uint8   observations to encode
    act_blocks  (F, frameskip * A) f32   the action block executed at each row
    ep_ptr      (E+1,) int64             episode boundaries
    ep_ids      (E,)   int64             which dataset episodes were sampled
    <state...>                           ground-truth coordinates named by the TaskSpec

Episodes are drawn with `numpy.default_rng(0)` without replacement, so every checkpoint is scored
on the same frames; scores taken against differently built probes are not comparable. At frameskip
5 one probe row carries the concatenation of five per-step actions, so `act_blocks` has 5*A
columns. Probes are built from training data -- the metric never reads success labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FRAMESKIP = 5

# Per-task dataset columns to store, as {probe_key: h5_column}. The probe keys are what
# `viscore.tasks.TaskSpec.scored` / `.nuisance` read.
PRESETS: dict[str, dict] = {
    "pusht":   dict(h5="pusht_expert_train.h5",           n_eps=300,
                    columns={"state": "state"}),
    "reacher": dict(h5="dmc/reacher.h5",                  n_eps=300,
                    columns={"qpos": "qpos", "finger_pos": "finger_pos",
                             "target_pos": "target_pos"}),
    "tworoom": dict(h5="tworoom.h5",                      n_eps=300,
                    columns={"pos_agent": "pos_agent"}),
    "cube":    dict(h5="ogbench/cube_single_expert.h5",   n_eps=300,
                    columns={"block_pos": "privileged_block_0_pos",
                             "effector_pos": "proprio_effector_pos"}),
    "maze2d":  dict(h5="maze2d_medium.h5",                n_eps=300,
                    columns={"state": "agent_pos", "qvel": "qvel"}),
}


def _slice_column(f, name: str, off: int, n: int, frameskip: int):
    """Read `name` for n frameskipped rows starting at raw offset `off`."""
    return f[name][off:off + frameskip * n:frameskip]


def build_probe(h5_path: Path | str, out: Path | str, columns: dict[str, str],
                n_eps: int = 300, seed: int = 0, frameskip: int = FRAMESKIP) -> Path:
    """Write a probe npz. Idempotent: an existing output is left alone."""
    import h5py
    import hdf5plugin  # noqa: F401  -- registers the pixel compression filters

    out = Path(out)
    if out.exists():
        print(f"skip (exists) {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(h5_path), "r") as f:
        ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
        rng = np.random.default_rng(seed)
        ep_ids = np.sort(rng.choice(len(ep_len), size=min(n_eps, len(ep_len)), replace=False))
        parts: dict[str, list] = {k: [] for k in ("pixels", "act_blocks", *columns)}
        ptr = [0]
        for ei in ep_ids:
            off, L = int(ep_off[ei]), int(ep_len[ei])
            n = L // frameskip
            if n < 2:
                continue
            parts["pixels"].append(_slice_column(f, "pixels", off, n, frameskip))
            for key, col in columns.items():
                parts[key].append(_slice_column(f, col, off, n, frameskip))
            act = f["action"][off:off + frameskip * n].astype(np.float32)
            parts["act_blocks"].append(act.reshape(n, -1))     # (n, frameskip * A)
            ptr.append(ptr[-1] + n)

    data = {k: np.concatenate(v, 0) for k, v in parts.items()}
    data["pixels"] = data["pixels"].astype(np.uint8)
    for k in columns:
        data[k] = data[k].astype(np.float32)
    data["act_blocks"] = data["act_blocks"].astype(np.float32)
    data["ep_ptr"] = np.asarray(ptr, dtype=np.int64)
    data["ep_ids"] = np.asarray(ep_ids, dtype=np.int64)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **data)
    tmp.rename(out)
    print(f"wrote {out}  frames={len(data['pixels'])}  episodes={len(ptr) - 1}  "
          f"act_blocks={data['act_blocks'].shape[1]}d")
    return out


def build_preset(task: str, data_home: Path | str, out_dir: Path | str, **kw) -> Path:
    if task not in PRESETS:
        raise KeyError(f"no preset for {task!r}; known: {sorted(PRESETS)}")
    p = PRESETS[task]
    return build_probe(Path(data_home) / p["h5"], Path(out_dir) / f"probe_{task}.npz",
                       columns=p["columns"], n_eps=p.get("n_eps", 300), **kw)


def load_probe(path: Path | str, with_pixels: bool = True) -> dict:
    """Load a probe npz into memory. `with_pixels=False` skips the ~1 GB pixel array."""
    with np.load(str(path)) as z:
        return {k: z[k] for k in z.files if with_pixels or k != "pixels"}
