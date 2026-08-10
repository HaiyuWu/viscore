"""Smoke and invariance tests. No checkpoint and no dataset: a toy linear world model.

The two invariance tests carry the claim that factors are comparable across models of different
latent scale and dimension: veracity depends only on the ratio d_tol / sigma_roll, and m_emp is a
log-determinant ratio of two covariances that transform congruently.
"""

import numpy as np
import pytest
import torch

from viscore import score_latents
from viscore.influence import capacity
from viscore.tasks import TaskSpec
from viscore.veracity import d_tol_shell
from viscore.latents import episode_pairs

D, ADIM, HS = 16, 10, 3
N_EP, EP_LEN = 40, 25


class ToyWM:
    """z_{t+1} = z_t A + emb(a_t), plus a controllable prediction error.

    `scale` multiplies BOTH the latents it is fed and its action embedding, so the whole model
    lives in a rescaled latent space -- the setting the invariance tests need.
    """

    def __init__(self, scale=1.0, err=0.0, seed=0):
        g = np.random.default_rng(seed)
        self.A = torch.tensor(0.9 * np.eye(D) + 0.02 * g.standard_normal((D, D))).float()
        self.W = torch.tensor(g.standard_normal((ADIM, D)) / np.sqrt(ADIM)).float()
        self.scale, self.err = scale, err
        self.device = torch.device("cpu")
        self.history_size = HS

    def action_embed(self, a):
        return (a @ self.W) * self.scale

    def predict_next(self, z_ctx, a_ctx):
        out = z_ctx[:, -1] @ self.A + a_ctx[:, -1]
        if self.err:
            # deterministic, state-dependent error: a fixed rotation of the state, so the
            # residual covariance is non-degenerate without needing an RNG in the forward pass
            out = out + self.err * torch.roll(z_ctx[:, -1], 1, dims=-1)
        return out


def toy_probe(seed=0, scale=1.0):
    """Latents consistent with ToyWM's dynamics, plus a 2-D ground-truth state to grade."""
    g = np.random.default_rng(seed)
    m = ToyWM(scale=1.0)
    A, W = m.A.numpy(), m.W.numpy()
    z, state, acts, ptr = [], [], [], [0]
    for _ in range(N_EP):
        zt = g.standard_normal(D)
        st = g.standard_normal(2)
        for _ in range(EP_LEN):
            a = g.standard_normal(ADIM)
            z.append(zt)
            state.append(st)
            acts.append(a)
            zt = zt @ A + a @ W
            st = st + 0.4 * a[:2]                 # physical state moves with the action
        ptr.append(len(z))
    probe = dict(act_blocks=np.asarray(acts, dtype=np.float32),
                 ep_ptr=np.asarray(ptr, dtype=np.int64),
                 state=np.asarray(state, dtype=np.float32))
    return np.asarray(z, dtype=np.float32) * scale, probe


TOY_TASK = TaskSpec(name="toy", tolerance=0.5, metric="l2",
                    scored=lambda p: p["state"], horizon=5, rollout_h_max=15,
                    probe_target=lambda p: p["state"])


def test_factors_bounded_and_product_consistent():
    z, probe = toy_probe()
    f = score_latents(ToyWM(err=0.05), z, probe, TOY_TASK, cell_id="toy")
    for name in ("veracity", "influence", "sobriety"):
        v = getattr(f, name)
        assert np.isfinite(v), f"{name} is not finite"
        assert 0.0 <= v <= 1.0, f"{name}={v} outside [0,1]"
    assert f.vis == pytest.approx(f.veracity * f.influence * f.sobriety)
    assert 0.0 < f.d_tol_grade < np.inf
    assert f.d_tol_veracity == pytest.approx(f.d_tol_grade / 2.0)


@pytest.mark.parametrize("scale", [0.1, 10.0])
def test_veracity_invariant_to_latent_rescaling(scale):
    """d_tol and sigma_roll both scale with the latent, so their ratio -- and veracity -- must not."""
    z1, probe = toy_probe()
    f1 = score_latents(ToyWM(err=0.05), z1, probe, TOY_TASK, cell_id="toy")
    z2, _ = toy_probe(scale=scale)
    f2 = score_latents(ToyWM(err=0.05, scale=scale), z2, probe, TOY_TASK, cell_id="toy")
    assert f2.d_tol_grade == pytest.approx(f1.d_tol_grade * scale, rel=1e-4)
    assert f2.sigma_roll == pytest.approx(f1.sigma_roll * scale, rel=1e-4)
    assert f2.veracity == pytest.approx(f1.veracity, rel=1e-4)


def test_capacity_invariant_to_linear_reparameterization():
    g = np.random.default_rng(0)
    X = g.standard_normal((D, 200))
    S = X @ X.T / 200 + np.eye(D)
    Y = g.standard_normal((D, 200))
    E = Y @ Y.T / 200 + np.eye(D)
    M = g.standard_normal((D, D)) + 3 * np.eye(D)          # invertible reparameterization
    assert capacity(M @ S @ M.T, M @ E @ M.T) == pytest.approx(capacity(S, E), rel=1e-6)


def test_d_tol_abstains_when_no_pair_sits_on_the_shell():
    """An unreachable tolerance must produce nan, not a silently extrapolated number."""
    z, probe = toy_probe()
    i, j = episode_pairs(probe["ep_ptr"])
    huge = TaskSpec(name="toy_huge", tolerance=1e6, metric="l2", scored=lambda p: p["state"])
    assert np.isnan(d_tol_shell(z, probe, huge, i, j))


def test_excluded_factor_is_nan_and_not_a_silent_one():
    """A dropped factor must be unmeasured (nan), never quietly folded in as 1.0."""
    z, probe = toy_probe()
    m = ToyWM(err=0.05)
    full = score_latents(m, z, probe, TOY_TASK, cell_id="toy")
    part = score_latents(m, z, probe, TOY_TASK, cell_id="toy", exclude=["sobriety"])
    assert part.factors == "VI"
    assert part.excluded == ("sobriety",)
    assert np.isnan(part.sobriety) and np.isnan(part.p_hat)
    assert part.vis == pytest.approx(full.veracity * full.influence)
    assert part.status == "ok"                       # abstention is about MEASURED factors only
    assert "sobriety" not in part.included


def test_excluding_veracity_still_measures_sigma_roll():
    """influence's noise floor E_H is anchored to sigma_roll, so the rollout is not optional."""
    z, probe = toy_probe()
    f = score_latents(ToyWM(err=0.05), z, probe, TOY_TASK, cell_id="toy", exclude="v")
    assert f.factors == "IS"
    assert np.isnan(f.veracity) and np.isnan(f.d_tol_grade)
    assert np.isfinite(f.sigma_roll) and np.isfinite(f.m_emp)


def test_single_factor_and_bad_selections():
    z, probe = toy_probe()
    m = ToyWM(err=0.05)
    only_i = score_latents(m, z, probe, TOY_TASK, cell_id="toy", exclude=["v", "s"])
    assert only_i.factors == "I"
    assert only_i.vis == pytest.approx(only_i.influence)
    for bad in (["veracity", "influence", "sobriety"], ["vis"], ["nonsense"]):
        try:
            score_latents(m, z, probe, TOY_TASK, cell_id="toy", exclude=bad)
        except ValueError:
            continue
        raise AssertionError(f"exclude={bad!r} should have raised")


def test_saturated_factors_are_reported():
    z, probe = toy_probe()
    # tau -> 0 forces influence to its ceiling; the cell must SAY that the factor is uninformative
    f = score_latents(ToyWM(err=0.05), z, probe, TOY_TASK, cell_id="toy", tau=1e-6)
    assert f.influence == pytest.approx(1.0)
    assert "influence" in f.saturated
