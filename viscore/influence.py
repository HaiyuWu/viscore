"""Influence: whether actions command enough distinguishable futures for the task.

    m_emp     = 1/2 logdet(I + E_H^-1 S)          [nats]
    influence = min(m_emp / tau, 1),  tau = 82

`S` is the covariance of H-step terminal-latent displacements under action perturbations drawn
from the empirical action-block covariance; `E_H` is the predictor's teacher-forced residual
covariance rescaled to the error the rollout accumulates at horizon H. Their log-determinant
ratio is the Gaussian channel capacity from actions to futures, invariant to an invertible
linear reparameterization of the latent.

The cap converts a capacity in nats, whose scale is task-specific, into a bounded sufficiency
condition. tau = 82 is a frozen constant; docs/METRIC.md gives its selection and its cost.
"""

from __future__ import annotations

import numpy as np
import torch

TAU = 82.0                      # sufficiency cap, nats. Frozen; see module docstring.
K_ANCHORS, M_PERTURB = 32, 64
SHRINK_E, SHRINK_A = 0.1, 0.1   # diagonal shrinkage on the residual / action covariances
TF_BATCH = 512
TF_MAX_WINDOWS = 4000


def expert_anchors(z: np.ndarray, act: np.ndarray, ep_ptr: np.ndarray, hs: int, h: int,
                   rng: np.random.Generator, k: int = K_ANCHORS, stride: int = 4):
    """K expert anchors: (context latents, normalized action block) at sampled timesteps."""
    mu, sd = np.nanmean(act, 0), np.nanstd(act, 0) + 1e-8
    cand = []
    for a, b in zip(ep_ptr[:-1], ep_ptr[1:]):
        a, b = int(a), int(b)
        if b - a < hs + h:
            continue
        for t in range(a + hs - 1, b - h, stride):
            if np.isnan(act[t:t + h]).any():
                continue
            cand.append(t)
    if not cand:
        return None, None
    sel = rng.choice(len(cand), size=min(k, len(cand)), replace=False)
    ctx = np.stack([z[cand[i] - hs + 1:cand[i] + 1] for i in sel])
    blk = np.stack([(act[cand[i]:cand[i] + h] - mu) / sd for i in sel])
    return torch.tensor(ctx).float(), torch.tensor(blk).float()


@torch.no_grad()
def roll_last(model, ctx: torch.Tensor, blocks: torch.Tensor, hs: int, h: int) -> torch.Tensor:
    """Terminal latent of an h-step open-loop rollout from `ctx` under `blocks`."""
    emb = ctx
    for step in range(h):
        a_ctx = model.action_embed(blocks[:, max(0, step - hs + 1):step + 1])
        z_ctx = emb[:, -min(hs, step + 1):]
        n = min(z_ctx.size(1), a_ctx.size(1))
        pred = model.predict_next(z_ctx[:, -n:], a_ctx[:, -n:])
        emb = torch.cat([emb, pred.unsqueeze(1)], 1)
    return emb[:, -1]


@torch.no_grad()
def tf_residuals(model, z: np.ndarray, act_n: np.ndarray, ep_ptr: np.ndarray,
                 hs: int) -> np.ndarray:
    """Teacher-forced one-step residuals: the predictor's error given true context.
    This is the channel's noise floor, rescaled to horizon H by `noise_floor_at_horizon`.
    """
    dev = model.device
    idx = []
    for a, b in zip(ep_ptr[:-1], ep_ptr[1:]):
        a, b = int(a), int(b)
        if b - a < hs + 1:
            continue
        for j in range(hs, b - a):
            t = a + j
            if np.isnan(act_n[t - hs:t]).any():
                continue
            idx.append(t)
        if len(idx) > TF_MAX_WINDOWS:
            break
    idx = np.asarray(idx)
    zc = torch.from_numpy(np.stack([z[t - hs:t] for t in idx])).float()
    ac = torch.from_numpy(np.stack([act_n[t - hs:t] for t in idx])).float()
    tgt = torch.from_numpy(z[idx]).float()
    out = []
    for s in range(0, len(idx), TF_BATCH):
        pred = model.predict_next(zc[s:s + TF_BATCH].to(dev),
                                  model.action_embed(ac[s:s + TF_BATCH].to(dev)))
        out.append((pred.cpu() - tgt[s:s + TF_BATCH]).numpy())
    return np.concatenate(out, 0)


def spectra(model, z: np.ndarray, probe: dict, hs: int, h: int,
            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(S, E): action-induced terminal-displacement covariance and one-step residual covariance."""
    act = probe["act_blocks"].astype(np.float32)
    mu, sd = np.nanmean(act, 0), np.nanstd(act, 0) + 1e-8
    act_n = (act - mu) / sd
    ok = ~np.isnan(act_n).any(1)
    Sa = np.cov(act_n[ok], rowvar=False)
    Sa = (1 - SHRINK_A) * Sa + SHRINK_A * np.diag(np.diag(Sa))
    La = np.linalg.cholesky(Sa + 1e-6 * np.eye(len(Sa)))

    rng = np.random.default_rng(seed)
    ctx, blk = expert_anchors(z, act, probe["ep_ptr"], hs, h, rng)
    if ctx is None:
        raise RuntimeError("no anchor window survives; probe episodes shorter than hs + horizon")
    ctx, blk = ctx.to(model.device), blk.to(model.device)

    # Perturbations are drawn from the empirical action-block covariance, so S measures the
    # spread of futures under ACTIONS THE DATA CONTAINS rather than under isotropic noise.
    base = roll_last(model, ctx, blk, hs, h)
    deltas = rng.standard_normal((M_PERTURB, h, blk.size(-1))) @ La.T
    diffs = []
    for m in range(M_PERTURB):
        d = torch.tensor(deltas[m]).float().to(model.device)
        diffs.append((roll_last(model, ctx, blk + d.unsqueeze(0), hs, h) - base).cpu().numpy())
    S = np.cov(np.concatenate(diffs, 0), rowvar=False)

    E = np.cov(tf_residuals(model, z, act_n, probe["ep_ptr"], hs), rowvar=False)
    E = (1 - SHRINK_E) * E + SHRINK_E * np.diag(np.diag(E))
    E += 1e-8 * np.trace(E) / len(E) * np.eye(len(E))
    return S.astype(np.float64), E.astype(np.float64)


def noise_floor_at_horizon(E: np.ndarray, sigma_roll: float) -> np.ndarray:
    """E_H = sigma_roll^2 * E / tr(E): keep E's directions, match its trace to the H-step error.

    Required because S is an H-step quantity while E is one-step. Assumes the predictor's error
    directions are approximately horizon-stable, which is not verified here.
    """
    return sigma_roll ** 2 * E / max(np.trace(E), 1e-12)


def capacity(S: np.ndarray, N: np.ndarray) -> float:
    """1/2 logdet(I + N^-1 S), via the symmetric whitening N^-1/2 S N^-1/2, in nats."""
    w, V = np.linalg.eigh(N)
    Nih = V @ np.diag(np.clip(w, 1e-12, None) ** -0.5) @ V.T
    return float(0.5 * np.log1p(np.clip(np.linalg.eigvalsh(Nih @ S @ Nih), 0, None)).sum())


def influence(m_emp: float, tau: float = TAU) -> float:
    return float(min(m_emp / tau, 1.0)) if np.isfinite(m_emp) else float("nan")


def measure(model, z: np.ndarray, probe: dict, task, hs: int, sigma_roll: float,
            tau: float = TAU) -> dict:
    """Influence-side quantities for one cell. `sigma_roll` comes from `veracity.measure`."""
    S, E = spectra(model, z, probe, hs, task.horizon)
    m_emp_1step = capacity(S, E)                       # diagnostic only: NOT the factor's input
    m_emp = capacity(S, noise_floor_at_horizon(E, sigma_roll))
    return dict(influence=influence(m_emp, tau), m_emp=m_emp, m_emp_1step=m_emp_1step,
                tau=tau, S=S, E=E)
