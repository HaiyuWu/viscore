"""The two diagnostics VIScore is compared against. Not part of the score.

    probe_r2      ridge regression from latents to the coordinates the task is scored on, held out
                  by episode: can the state be read off the latent at all?
    straightness  mean cos(z_t - z_{t-1}, z_{t+1} - z_t) over episodes.
"""

from __future__ import annotations

import numpy as np


def probe_r2(z: np.ndarray, y: np.ndarray, ep_ptr: np.ndarray, seed: int = 1,
             train_frac: float = 0.8, ridge: float = 1e-2) -> float:
    """Mean per-dimension R^2 of a ridge fit latents -> y, held out by episode.

    Episode-level holdout, not frame-level: consecutive frames are near-duplicates, so a random frame
    split leaks the target and reads ~1.0 for any encoder.
    """
    rng = np.random.default_rng(seed)
    nep = len(ep_ptr) - 1
    perm = rng.permutation(nep)
    cut = int(train_frac * nep)
    idx = lambda eps: np.concatenate([np.arange(ep_ptr[e], ep_ptr[e + 1]) for e in eps])  # noqa: E731
    tr, te = idx(perm[:cut]), idx(perm[cut:])
    X = np.column_stack([z, np.ones(len(z))])
    mu, sd = y[tr].mean(0), y[tr].std(0) + 1e-8
    A = X[tr].T @ X[tr] + ridge * len(tr) * np.eye(X.shape[1])
    W = np.linalg.solve(A, X[tr].T @ ((y[tr] - mu) / sd))
    P, Yv = X[te] @ W, (y[te] - mu) / sd
    ss = 1.0 - ((Yv - P) ** 2).sum(0) / np.maximum(((Yv - Yv.mean(0)) ** 2).sum(0), 1e-12)
    return float(np.mean(ss))


def straightness(z: np.ndarray, ep_ptr: np.ndarray) -> float:
    """Mean cosine between consecutive latent steps, over all episodes."""
    cos_all = []
    for a, b in zip(ep_ptr[:-1], ep_ptr[1:]):
        a, b = int(a), int(b)
        if b - a < 3:
            continue
        v = np.diff(z[a:b], axis=0)
        nv = np.linalg.norm(v, axis=1)
        ok = (nv[:-1] > 1e-8) & (nv[1:] > 1e-8)
        if ok.any():
            cos_all.append(((v[:-1] * v[1:]).sum(1) / (nv[:-1] * nv[1:] + 1e-30))[ok])
    return float(np.concatenate(cos_all).mean()) if cos_all else float("nan")


def measure(z: np.ndarray, probe: dict, task) -> dict:
    """Both baselines for one cell. probe_r2 is nan when the task has no probe target."""
    r2 = float("nan")
    if task.probe_target is not None:
        y = np.asarray(task.probe_target(probe), dtype=np.float64)
        r2 = probe_r2(z.astype(np.float64), y, probe["ep_ptr"])
    return dict(probe_r2=r2, straightness=straightness(z.astype(np.float64), probe["ep_ptr"]))
