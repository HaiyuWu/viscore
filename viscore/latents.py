"""Encode a probe's pixels with a frozen checkpoint into the latent bank every factor reads.

All three factors are functions of the same (F, D) bank plus the predictor, so encoding happens
once per (checkpoint, probe) and is cacheable with `--latents-dir`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def encode_probe(model, probe: dict, chunk: int = 128) -> np.ndarray:
    """(F, D) float32 latents for a probe's pixels, in probe row order."""
    px = torch.from_numpy(probe["pixels"]).permute(0, 3, 1, 2).contiguous()   # (F,3,H,W) uint8
    z = model.encode(px, chunk=chunk)
    return z.numpy().astype(np.float32)


def cached_latents(model_fn, probe, cache: Path | str | None, key: str,
                   chunk: int = 128) -> np.ndarray:
    """Latents for `key`, from `cache` if present, else encoded via `model_fn()` and stored.

    Both `model_fn` and `probe` may be thunks; on a cache hit neither is called, so a re-score loads
    neither the checkpoint nor the probe's pixel array.
    """
    if cache is not None:
        p = Path(cache) / f"{key}.npz"
        if p.exists():
            with np.load(p) as f:
                return f["z"].astype(np.float32)
    z = encode_probe(model_fn(), probe() if callable(probe) else probe, chunk=chunk)
    if cache is not None:
        p = Path(cache)
        p.mkdir(parents=True, exist_ok=True)
        tmp = p / f"{key}.tmp.npz"
        np.savez(tmp, z=z)
        tmp.rename(p / f"{key}.npz")
    return z


def episode_pairs(ep_ptr: np.ndarray, max_offset: int = 5, cap: int = 40000,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Within-episode row pairs 1..max_offset apart, subsampled to `cap` with a fixed seed.
    These are the pairs d_tol is measured on.
    """
    rng = np.random.default_rng(seed)
    pi, pj = [], []
    for a, b in zip(ep_ptr[:-1], ep_ptr[1:]):
        n = int(b) - int(a)
        if n < 3:
            continue
        for off in range(1, min(max_offset + 1, n)):
            i = np.arange(int(a), int(b) - off)
            pi.append(i)
            pj.append(i + off)
    i, j = np.concatenate(pi), np.concatenate(pj)
    if len(i) > cap:
        s = rng.choice(len(i), cap, replace=False)
        i, j = i[s], j[s]
    return i, j
