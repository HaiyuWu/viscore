"""VIScore: an interpretable, single-checkpoint diagnostic of latent world-model quality for
search-based planning.

    from viscore import score_checkpoint, load_probe

    probe = load_probe("cache/probes/probe_pusht.npz")
    f = score_checkpoint("vis-wm/pusht/seed403/vis-wm_epoch_9.ckpt", probe, task="pusht")
    print(f)

VIS = veracity * influence * sobriety, all measured offline from cached latents and the predictor
alone -- no environment, no planner, no success labels. See docs/METRIC.md for the specification.
"""

from .adapters import LeWMAdapter, load_model, register_lewm_modules
from .influence import TAU
from .probe import build_preset, build_probe, load_probe
from .calibration import fit_pool_size, predict_success  # noqa: F401
from .score import (FACTORS, VISFactors, combine, resolve_factors, score_checkpoint,
                    score_latents, write_csv, write_json)
from .tasks import TASKS, TaskSpec, get_task

__version__ = "0.1.0"
__all__ = [
    "predict_success",
    "fit_pool_size",
    "FACTORS", "LeWMAdapter", "TASKS", "TAU", "TaskSpec", "VISFactors", "build_preset",
    "build_probe", "combine", "get_task", "load_model", "load_probe", "register_lewm_modules",
    "resolve_factors", "score_checkpoint", "score_latents", "write_csv", "write_json",
]
