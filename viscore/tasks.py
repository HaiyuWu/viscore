"""Task specifications: how an environment's success criterion maps onto probe arrays.

VIScore is tolerance-anchored, so a `TaskSpec` describes the ENVIRONMENT, not the model. Every
tolerance below is read off the environment's success criterion:

    pusht     ||d state[:4]||_2 < 20 px  (agent_xy + block_xy; the conjunctive pi/9 angle gate
                                          was measured never to bind first)
    reacher   max_j |d qpos_j| < 0.05 rad
    tworoom   ||agent_pos - target_pos|| < 16 px
    cube      ||block_pos - target_pos|| <= 0.04 m
    maze2d    ||agent_pos - goal|| <= 0.5

`scored` returns the coordinates the criterion grades. `nuisance` returns coordinates it ignores
but which still move the latent -- an arm that sweeps while the graded object is still. d_tol
needs them: without the filter, latent motion caused by the arm is credited to the object.

`horizon` is the diagnostic horizon in action blocks; at frameskip 5 the evaluation protocol's
goal offset d = 25 / 50 / 75 corresponds to horizon 5 / 10 / 15. `rollout_h_max` is the length of
the open-loop curve the veracity clip pool is built from; keep the defaults to match published
numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Probe = dict


@dataclass(frozen=True)
class TaskSpec:
    name: str
    tolerance: float
    metric: str                                  # "l2" or "linf" -- how the env compares states
    scored: Callable[[Probe], np.ndarray]        # coordinates the success criterion grades
    nuisance: Callable[[Probe], np.ndarray | None] = lambda p: None
    horizon: int = 5                             # action blocks; d=25 at frameskip 5
    rollout_h_max: int = 15                      # open-loop curve length for the clip pool
    probe_target: Callable[[Probe], np.ndarray] | None = None   # for the probe-R2 baseline

    def state_distance(self, s: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Distance between probe rows i and j in the environment's own metric."""
        return (np.max(np.abs(s[i] - s[j]), axis=1) if self.metric == "linf"
                else np.linalg.norm(s[i] - s[j], axis=1))


def _pusht_probe_target(p: Probe) -> np.ndarray:
    """agent_xy, block_xy, cos/sin of block angle -- the angle needs the circular encoding."""
    st = p["state"]
    return np.column_stack([st[:, 0], st[:, 1], st[:, 2], st[:, 3],
                            np.cos(st[:, 4]), np.sin(st[:, 4])])


TASKS: dict[str, TaskSpec] = {
    "pusht": TaskSpec(
        name="pusht", tolerance=20.0, metric="l2",
        scored=lambda p: p["state"][:, 0:4],
        nuisance=lambda p: p["state"][:, 5:7],       # agent velocity: unscored
        probe_target=_pusht_probe_target,
    ),
    "reacher": TaskSpec(
        name="reacher", tolerance=0.05, metric="linf",
        scored=lambda p: p["qpos"],
        probe_target=lambda p: p["qpos"],
    ),
    "tworoom": TaskSpec(
        name="tworoom", tolerance=16.0, metric="l2",
        scored=lambda p: p["pos_agent"],
        probe_target=lambda p: p["pos_agent"],
        # Two-Room episodes are ~12 frameskip-5 rows long, so the open-loop curve stops at 8.
        rollout_h_max=8,
    ),
    "cube": TaskSpec(
        name="cube", tolerance=0.04, metric="l2",
        scored=lambda p: p["block_pos"],
        nuisance=lambda p: p["effector_pos"],        # the arm; criterion is block position only
        probe_target=lambda p: p["block_pos"],
    ),
    "maze2d": TaskSpec(
        name="maze2d", tolerance=0.5, metric="l2",
        scored=lambda p: p["state"],
        probe_target=lambda p: p["state"],
    ),
}


def get_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}. "
                       "Register your own with viscore.tasks.TASKS[name] = TaskSpec(...)")
    return TASKS[name]
