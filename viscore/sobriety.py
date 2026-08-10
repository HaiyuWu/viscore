"""Sobriety: the fraction of anchors where a search CANNOT beat the expert in imagination.

    p_hat    = (1/K) sum_k 1[ J(a*_k) > J(a_search_k) ]
    sobriety = 1 - p_hat

At each of K expert anchors the goal is the future the recorded action reached, so the expert's
action is a known-good plan. A mini-CEM searches for a lower imagined terminal cost; every anchor
it wins is an exploitable hole in the latent landscape. `p_hat` counts those wins, so sobriety --
one minus it -- is high for a landscape the search cannot exploit. Only the sign of each gap
enters, never its size.

Sobriety is the one factor that depends on a search, so a comparison must keep the probe fixed.
It is deterministic given `cell_id`; when many anchors sit near zero gap (see `gap_resolution`),
average over several `cell_id` seeds before ranking two close checkpoints.
"""

from __future__ import annotations

import zlib

import numpy as np
import torch

CEM_N, CEM_ITERS, CEM_ELITE = 128, 4, 16
K_ANCHORS = 64


def anchors_with_goals(z: np.ndarray, act: np.ndarray, ep_ptr: np.ndarray, hs: int, h: int,
                       rng: np.random.Generator, k: int = K_ANCHORS, stride: int = 4):
    """(ctx, expert action block, goal) at sampled timesteps, where goal = z[t+h]."""
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
        return None, None, None
    sel = rng.choice(len(cand), size=min(k, len(cand)), replace=False)
    ctx = np.stack([z[cand[i] - hs + 1:cand[i] + 1] for i in sel])
    blk = np.stack([(act[cand[i]:cand[i] + h] - mu) / sd for i in sel])
    goal = np.stack([z[cand[i] + h] for i in sel])
    return (torch.tensor(ctx).float(), torch.tensor(blk).float(), torch.tensor(goal).float())


@torch.no_grad()
def rollout_cost(model, ctx: torch.Tensor, blocks: torch.Tensor, goal: torch.Tensor,
                 hs: int, h: int) -> torch.Tensor:
    """J(a) = ||rollout(ctx, a)_h - goal||^2, the last-step cost the planner minimizes."""
    emb = ctx
    for step in range(h):
        a_ctx = model.action_embed(blocks[:, max(0, step - hs + 1):step + 1])
        z_ctx = emb[:, -min(hs, step + 1):]
        n = min(z_ctx.size(1), a_ctx.size(1))
        emb = torch.cat([emb, model.predict_next(z_ctx[:, -n:], a_ctx[:, -n:]).unsqueeze(1)], 1)
    return (emb[:, -1] - goal).pow(2).sum(-1)


@torch.no_grad()
def mini_cem(model, ctx, blk, goal, hs: int, h: int, n: int = CEM_N, iters: int = CEM_ITERS,
             elite: int = CEM_ELITE):
    """CEM in normalized action space, initialized at the expert block.

    Candidates fold into the batch dimension and are chunked so B*n cannot exhaust GPU memory. No
    clamp to the action box and no forced mean candidate: the probe measures the landscape.
    """
    B, adim = ctx.size(0), blk.size(-1)
    mu, sd = blk.clone(), torch.ones_like(blk)
    chunk = max(1, 8192 // max(B, 1))
    J = None
    for _ in range(iters):
        samp = mu.unsqueeze(1) + sd.unsqueeze(1) * torch.randn(B, n, h, adim, device=ctx.device)
        parts = []
        for s in range(0, n, chunk):
            k = samp[:, s:s + chunk].shape[1]
            parts.append(rollout_cost(model, ctx.repeat_interleave(k, 0),
                                      samp[:, s:s + chunk].reshape(B * k, h, adim),
                                      goal.repeat_interleave(k, 0), hs, h).view(B, k))
        J = torch.cat(parts, 1)
        idx = J.topk(elite, dim=1, largest=False).indices
        el = torch.gather(samp, 1, idx.view(B, elite, 1, 1).expand(-1, -1, h, adim))
        mu, sd = el.mean(1), el.std(1) + 1e-4
    elite_sd = torch.gather(J, 1, J.topk(elite, dim=1, largest=False).indices).std(1)
    return mu, elite_sd


def measure(model, z: np.ndarray, probe: dict, task, hs: int, cell_id: str = "cell") -> dict:
    """p_hat and sobriety for one cell, with the per-anchor gaps and the probe's resolution floor.

    The CEM's sampling is seeded from `cell_id` (crc32, which is stable across processes).
    """
    act = probe["act_blocks"].astype(np.float32)
    ctx, blk, goal = anchors_with_goals(z, act, probe["ep_ptr"], hs, task.horizon,
                                        np.random.default_rng(0))
    if ctx is None:
        return dict(sobriety=float("nan"), p_hat=float("nan"), gaps=np.array([]))
    dev = model.device
    ctx, blk, goal = ctx.to(dev), blk.to(dev), goal.to(dev)
    torch.manual_seed(zlib.crc32(cell_id.encode()))

    j_expert = rollout_cost(model, ctx, blk, goal, hs, task.horizon)
    a_opt, elite_sd = mini_cem(model, ctx, blk, goal, hs, task.horizon)
    j_search = rollout_cost(model, ctx, a_opt, goal, hs, task.horizon)
    gaps = (j_expert - j_search).cpu().numpy().astype(np.float64)   # > 0 == anchor exploitable
    finite = gaps[np.isfinite(gaps)]
    p_hat = float((finite > 0).mean()) if finite.size else float("nan")
    return dict(sobriety=1.0 - p_hat if np.isfinite(p_hat) else float("nan"), p_hat=p_hat,
                gaps=gaps,
                # Smallest gap this probe can resolve: the standard error of the elite mean.
                # Gaps below it are indistinguishable from zero, so a sobriety of exactly 1
                # would be unwarranted -- report it rather than hide it.
                gap_resolution=float(np.median(elite_sd.cpu().numpy()) / np.sqrt(CEM_ELITE)))
