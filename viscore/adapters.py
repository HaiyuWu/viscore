"""Model adapters -- the only place VIScore touches a world model's own API.

VIScore needs exactly three operations from a latent world model, and nothing else:

    encode(pixels)              (T, 3, H, W) uint8   -> (T, D) latents
    action_embed(actions)       (B, T, A)            -> (B, T, D_a) action embeddings
    predict_next(z_ctx, a_ctx)  (B, HS, D), (B, HS, D_a) -> (B, D) next latent

Anything satisfying that protocol can be scored: the metric never imports the training
framework, the dataloader, or the planner. `LeWMAdapter` below wraps the LeWM / VIS-WM
family (a `torch.save` pickle of a `jepa.JEPA`); `examples/custom_adapter.py` shows a
from-scratch adapter for a foreign model.

Two conventions are baked in because the metric's numbers depend on them:

* **Preprocessing** is ImageNet mean/std on [0, 1] floats, matching how the LeWM family is
  trained. A model trained with different preprocessing must apply its own inside `encode`.
* **`predict_next` returns the LAST position** of the predictor's output sequence, which is
  the next-step prediction for an autoregressive predictor over a context window.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Shipped copy of the LeWM model definition. Full-object checkpoints are pickles that name
# `jepa` and `module` as TOP-LEVEL modules, so they can only be unpickled if those names
# resolve -- see register_lewm_modules().
_VISWM_DIR = Path(__file__).resolve().parents[1] / "viswm"


@runtime_checkable
class WorldModel(Protocol):
    """What VIScore requires of a model. See module docstring."""

    device: torch.device
    history_size: int

    def encode(self, pixels_u8: torch.Tensor, chunk: int = 128) -> torch.Tensor: ...

    def action_embed(self, actions: torch.Tensor) -> torch.Tensor: ...

    def predict_next(self, z_ctx: torch.Tensor, a_ctx: torch.Tensor) -> torch.Tensor: ...


def register_lewm_modules(repo_dir: Path | str | None = None) -> None:
    """Bind `jepa` / `module` as top-level modules so LeWM object checkpoints can be unpickled.

    A LeWM checkpoint is `torch.save(JEPA(...))`, so the pickle references those names at top level.
    This registers the copies shipped in `viswm/` (or another checkout via `repo_dir`). Idempotent,
    and it never overwrites an already-imported `jepa` / `module`.
    """
    d = Path(repo_dir) if repo_dir is not None else _VISWM_DIR
    for name in ("module", "jepa"):
        if name in sys.modules:
            continue
        path = d / f"{name}.py"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; pass repo_dir=<LeWM checkout>")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod          # register BEFORE exec: jepa.py may import `module`
        spec.loader.exec_module(mod)


def find_jepa(obj):
    """The submodule of a loaded checkpoint that owns `.encode` (the JEPA itself)."""
    if hasattr(obj, "encode") and callable(obj.encode):
        return obj
    if isinstance(obj, torch.nn.Module):
        for m in obj.modules():
            if m is not obj and hasattr(m, "encode") and callable(m.encode):
                return m
    raise RuntimeError(f"no .encode found in {type(obj)}")


class LeWMAdapter:
    """Adapter for the LeWM / VIS-WM family (ViT encoder + autoregressive AdaLN predictor)."""

    def __init__(self, jepa: torch.nn.Module, history_size: int | None = None):
        self.jepa = jepa.eval().requires_grad_(False)
        self.device = next(jepa.parameters()).device
        # history_size is a property of the trained predictor, not of the task: the predictor's
        # positional embedding has exactly `history_size` slots.
        if history_size is None:
            pos = getattr(jepa.predictor, "pos_embedding", None)
            history_size = int(pos.shape[1]) if pos is not None else 3
        self.history_size = history_size

    @classmethod
    def load(cls, ckpt: Path | str, device: str = "cuda", repo_dir: Path | str | None = None,
             history_size: int | None = None) -> "LeWMAdapter":
        """Load a full-object checkpoint pickle (`<model>_epoch_<N>.ckpt`)."""
        register_lewm_modules(repo_dir)
        obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        return cls(find_jepa(obj).to(device), history_size=history_size)

    @torch.no_grad()
    def encode(self, pixels_u8: torch.Tensor, chunk: int = 128) -> torch.Tensor:
        """(T, 3, H, W) uint8 (any device) -> (T, D) float32 on CPU."""
        x = pixels_u8.float().div_(255.0)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        out = []
        for i in range(0, x.size(0), chunk):
            xb = x[i:i + chunk].unsqueeze(0).to(self.device)      # (1, t, 3, H, W)
            out.append(self.jepa.encode({"pixels": xb})["emb"][0].float().cpu())
        return torch.cat(out, 0)

    def action_embed(self, actions: torch.Tensor) -> torch.Tensor:
        return self.jepa.action_encoder(actions)

    def predict_next(self, z_ctx: torch.Tensor, a_ctx: torch.Tensor) -> torch.Tensor:
        return self.jepa.predict(z_ctx, a_ctx)[:, -1]


def load_model(ckpt: Path | str, device: str = "cuda", family: str = "lewm",
               **kw) -> WorldModel:
    """Front door for the CLI. `family` selects the adapter; add your own here."""
    if family != "lewm":
        raise ValueError(f"unknown model family {family!r}; see examples/custom_adapter.py")
    return LeWMAdapter.load(ckpt, device=device, **kw)
