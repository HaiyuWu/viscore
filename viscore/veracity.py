"""Veracity: the probability that an H-step open-loop rollout stays inside the task tolerance.

    veracity     = erf( (d_tol / 2) / (sqrt(2) * sigma_roll) )
    sigma_roll^2 = NMSE_ol(H) * tr(Sigma_z)

`d_tol` is the latent image of the environment's success tolerance, measured on probe pairs whose
task-scored displacement already equals one tolerance. Only the ratio d_tol / sigma_roll enters,
so the factor is invariant to a rescaling of the latent.

The half tolerance in the numerator and the `erf` map are fixed constants, not fitted; see
docs/METRIC.md for their derivation.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import erf

from .latents import episode_pairs


def _act_stats(act_blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension action normalization, from the probe's own action blocks."""
    return np.nanmean(act_blocks, 0), np.nanstd(act_blocks, 0) + 1e-8


def build_clips(z: np.ndarray, act: np.ndarray, ep_ptr: np.ndarray, hs: int, h: int,
                stride: int = 3, max_clips: int = 512, seed: int = 0):
    """(clips, normalized actions) of length hs+h. NaN-bearing windows are dropped and the pool is
    capped at `max_clips` with a fixed seed.
    """
    mu, sd = _act_stats(act)
    L = hs + h
    zs, as_ = [], []
    for a, b in zip(ep_ptr[:-1], ep_ptr[1:]):
        a, b = int(a), int(b)
        for t in range(0, b - a - L + 1, stride):
            if np.isnan(act[a + t:a + t + L]).any():
                continue
            zs.append(z[a + t:a + t + L])
            as_.append(act[a + t:a + t + L])
    if not zs:
        return None, None
    zs, as_ = np.stack(zs), (np.stack(as_) - mu) / sd
    if len(zs) > max_clips:
        sel = np.random.default_rng(seed).choice(len(zs), max_clips, replace=False)
        zs, as_ = zs[sel], as_[sel]
    return torch.from_numpy(zs).float(), torch.from_numpy(as_).float()


@torch.no_grad()
def _rollout_sq(model, z: torch.Tensor, act: torch.Tensor, hs: int, h: int) -> np.ndarray:
    """Squared open-loop error at horizons 1..h, averaged over the batch.

    Open loop: the model's own predictions are fed back as context; only the actions come from data.
    """
    a_emb = model.action_embed(act)
    emb = z[:, :hs].clone()
    out = []
    for step in range(1, h + 1):
        j = hs - 1 + step
        pred = model.predict_next(emb[:, -hs:], a_emb[:, j - hs:j])
        out.append((pred - z[:, j]).pow(2).sum(-1).mean().item())
        emb = torch.cat([emb, pred.unsqueeze(1)], dim=1)
    return np.asarray(out)


def rollout_nmse(model, z: np.ndarray, probe: dict, hs: int, h_max: int,
                 batch: int = 128) -> np.ndarray:
    """Open-loop NMSE curve for horizons 1..h_max, normalized by the latent bank's total variance.
    Returns a nan curve when no clip of length hs+h_max exists.
    """
    zc, ac = build_clips(z, probe["act_blocks"].astype(np.float32), probe["ep_ptr"], hs, h_max)
    if zc is None:
        return np.full(h_max, np.nan)
    dev = model.device
    tot = np.zeros(h_max)
    for i in range(0, len(zc), batch):
        w = len(zc[i:i + batch]) / len(zc)
        tot += _rollout_sq(model, zc[i:i + batch].to(dev), ac[i:i + batch].to(dev), hs, h_max) * w
    return tot / (trace_cov(z) + 1e-12)


def trace_cov(z: np.ndarray) -> float:
    """tr(Sigma_z): the latent bank's total variance, the unit of every latent-space quantity."""
    return float(np.trace(np.cov(z.astype(np.float64), rowvar=False)))


def d_tol_shell(z: np.ndarray, probe: dict, task, i: np.ndarray, j: np.ndarray,
                band: tuple[float, float] = (0.8, 1.2), nuisance_q: float = 0.25,
                min_pairs: int = 30) -> float:
    """Median latent distance over probe pairs whose task-scored displacement is one tolerance.

    Pairs are kept when their scored displacement lies in `band * tolerance`; where the criterion
    ignores coordinates, the quartile with the least nuisance motion is kept as well. Returns nan
    when fewer than `min_pairs` pairs qualify, which the score reports as an abstention.
    """
    dz = np.linalg.norm(z[i] - z[j], axis=1)
    s = np.asarray(task.scored(probe), dtype=np.float64)
    ds = task.state_distance(s, i, j)
    m = (ds > band[0] * task.tolerance) & (ds < band[1] * task.tolerance)
    if m.sum() < min_pairs:
        return float("nan")
    nu = task.nuisance(probe)
    if nu is not None:
        nu = np.asarray(nu, dtype=np.float64)
        dn = np.linalg.norm(nu[i] - nu[j], axis=1)
        quiet = m & (dn < np.quantile(dn[m], nuisance_q))
        if quiet.sum() >= min_pairs:
            m = quiet
    return float(np.median(dz[m]))


def veracity(d_tol_grade: float, sigma_roll: float, strictness: float = 2.0) -> float:
    """erf( (d_tol / strictness) / (sqrt(2) sigma_roll) ); strictness=2 is the published value."""
    if not np.isfinite(d_tol_grade) or not np.isfinite(sigma_roll):
        return float("nan")
    return float(erf((d_tol_grade / strictness) / (np.sqrt(2) * sigma_roll + 1e-12)))


def measure(model, z: np.ndarray, probe: dict, task, hs: int,
            with_d_tol: bool = True) -> dict:
    """Veracity-side quantities for one (checkpoint, probe) cell.

    `with_d_tol=False` returns sigma_roll only. The influence factor needs sigma_roll too, so the
    open-loop rollout still runs when veracity is excluded; only the d_tol shell is skipped.
    """
    nmse = rollout_nmse(model, z, probe, hs, task.rollout_h_max)
    trS = trace_cov(z)
    h = min(task.horizon, len(nmse))
    nmse_h = float(nmse[h - 1])
    sigma = float(np.sqrt(max(nmse_h, 1e-9) * trS))
    out = dict(veracity=float("nan"), d_tol_grade=float("nan"), d_tol_veracity=float("nan"),
               sigma_roll=sigma, nmse_ol=nmse_h, nmse_curve=nmse, tr_sigma_z=trS)
    if with_d_tol:
        i, j = episode_pairs(probe["ep_ptr"])
        dt_grade = d_tol_shell(z, probe, task, i, j)
        out.update(veracity=veracity(dt_grade, sigma), d_tol_grade=dt_grade,
                   d_tol_veracity=dt_grade / 2.0)
    return out
