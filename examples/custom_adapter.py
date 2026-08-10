"""Score a world model that is not from the LeWM family.

Runnable as-is (`python examples/custom_adapter.py`) against a toy model and a synthetic probe, so
the plumbing can be checked before pointing it at real weights.

Adapting a real model means answering three questions: how pixels become latents (including that
model's preprocessing), how an action block becomes what the predictor conditions on, and how one
predictor step works given a context window. Everything else is model-independent.
"""

import numpy as np
import torch

from viscore import score_latents
from viscore.tasks import TaskSpec

D, ADIM, HISTORY = 32, 10, 3


class MyForeignModel(torch.nn.Module):
    """Stand-in for someone else's world model, with a deliberately different API."""

    def __init__(self):
        super().__init__()
        self.enc = torch.nn.Conv2d(3, D, kernel_size=16, stride=16)
        self.act = torch.nn.Linear(ADIM, D)
        self.dyn = torch.nn.GRU(D, D, batch_first=True)

    def forward_encoder(self, images):                  # (B,3,H,W) float, THEIR normalization
        return self.enc(images).mean((-2, -1))          # global-pool the patch grid

    def forward_dynamics(self, z_seq, a_seq):           # (B,T,D),(B,T,D) -> (B,T,D)
        out, _ = self.dyn(z_seq + a_seq)
        return out


class MyAdapter:
    """The three methods viscore requires. Nothing else is consulted."""

    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.history_size = HISTORY          # the predictor's context length

    @torch.no_grad()
    def encode(self, pixels_u8, chunk=128):
        """(T,3,H,W) uint8 -> (T,D) float32 on CPU. Put YOUR preprocessing here.

        viscore hands over raw uint8 frames precisely so that each model can apply its own
        normalization; the LeWM adapter applies ImageNet mean/std, yours may differ.
        """
        out = []
        for i in range(0, pixels_u8.size(0), chunk):
            x = pixels_u8[i:i + chunk].float().div(255.0).to(self.device)
            out.append(self.model.forward_encoder(x).float().cpu())
        return torch.cat(out, 0)

    def action_embed(self, actions):
        """(B,T,A) -> (B,T,D_a). An identity is fine if your predictor takes raw actions."""
        return self.model.act(actions)

    def predict_next(self, z_ctx, a_ctx):
        """(B,HS,D),(B,HS,D_a) -> (B,D): the NEXT latent, i.e. the last output position."""
        return self.model.forward_dynamics(z_ctx, a_ctx)[:, -1]


def synthetic_probe(n_ep=40, ep_len=25, seed=0):
    """A probe with the arrays viscore reads. Real probes come from `viscore probe`."""
    g = np.random.default_rng(seed)
    F = n_ep * ep_len
    return dict(
        act_blocks=g.standard_normal((F, ADIM)).astype(np.float32),
        ep_ptr=np.arange(0, F + 1, ep_len, dtype=np.int64),
        # the coordinates this task's success criterion grades, in physical units
        state=np.cumsum(0.1 * g.standard_normal((F, 2)), axis=0).astype(np.float32),
    )


# A task is a statement about the ENVIRONMENT: the tolerance its `terminated` uses, the coordinates
# it grades, and (if any) the coordinates it ignores but which still move the latent.
MY_TASK = TaskSpec(
    name="my_task",
    tolerance=0.5,                        # read off the env source, never tuned on success labels
    metric="l2",                          # or "linf" if the criterion is per-coordinate
    scored=lambda p: p["state"],
    nuisance=lambda p: None,              # coordinates the criterion ignores but that move the latent
    horizon=5,                            # action blocks; = goal offset / frameskip
    rollout_h_max=15,
    probe_target=lambda p: p["state"],    # optional, for the probe-R2 baseline
)

if __name__ == "__main__":
    torch.manual_seed(0)
    adapter = MyAdapter(MyForeignModel())
    probe = synthetic_probe()

    # Real use: z = viscore.latents.encode_probe(adapter, probe) with probe["pixels"] present.
    z = np.random.default_rng(1).standard_normal((len(probe["act_blocks"]), D)).astype(np.float32)

    f = score_latents(adapter, z, probe, MY_TASK, cell_id="my_model_ep1", with_baselines=True)
    print(f)
    print(f"\nbaselines: probe_r2={f.extras['probe_r2']:+.3f} "
          f"straightness={f.extras['straightness']:+.3f}")
    if f.saturated:
        print(f"NOTE: {', '.join(f.saturated)} at ceiling -- not separating checkpoints here")