"""The score: VIS = veracity * influence * sobriety, and the entry points.

    VIS = erf( (d_tol/2) / (sqrt2 sigma_roll) ) * min(m_emp / 82, 1) * (1 - p_hat)

The three factors are multiplied because they are prerequisites: the weakest unmet one dominates,
which is also what makes the decomposition readable -- the small factor names the defect. Report
the factors, not only the product; a saturated factor (`VISFactors.saturated`) carries no
information on that task.

`exclude=` scores any subset and labels it in `VISFactors.factors` ("VIS", "VI", "I", ...).
A partial score is on a different scale and must not be compared with a full VIS number.
Excluding veracity still runs the open-loop rollout when influence is kept, because the noise
floor is anchored to sigma_roll; excluding sobriety is what saves time.

Scope and limitations (all measured) are listed in docs/METRIC.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import influence as _inf
from . import sobriety as _sob
from . import veracity as _ver
from .tasks import TaskSpec, get_task

SATURATION_EPS = 1e-6
FACTORS = ("veracity", "influence", "sobriety")
_INITIAL = {"v": "veracity", "i": "influence", "s": "sobriety"}


def resolve_factors(exclude=(), include=None) -> tuple[str, ...]:
    """Normalize a factor selection to an ordered tuple of full factor names.

    Accepts full names, initials, and compact forms: `exclude="s"`, `exclude=["sobriety"]`,
    `include="VI"`. Raises on an unknown name or an empty selection.
    """
    def _norm(spec) -> list[str]:
        if spec is None:
            return []
        items = [spec] if isinstance(spec, str) else list(spec)
        # "VI" / "vis" as a single string is a set of initials, not one name
        if len(items) == 1 and isinstance(items[0], str):
            s = items[0].strip().lower()
            if s not in FACTORS and set(s) <= set(_INITIAL):
                items = list(s)
        out = []
        for it in items:
            key = str(it).strip().lower()
            name = _INITIAL.get(key, key)
            if name not in FACTORS:
                raise ValueError(f"unknown factor {it!r}; use any of {FACTORS} or v/i/s")
            out.append(name)
        return out

    inc = _norm(include) or list(FACTORS)
    keep = tuple(f for f in FACTORS if f in inc and f not in _norm(exclude))
    if not keep:
        raise ValueError("every factor was excluded; nothing left to score")
    return keep


def label(factors) -> str:
    """"VIS" / "VI" / "S" -- the identity of a (possibly partial) score."""
    return "".join(f[0].upper() for f in FACTORS if f in factors)


@dataclass
class VISFactors:
    """One (checkpoint, probe) cell: the score, its factors, and their inputs.

    An excluded factor is nan, never 1.0; `vis` is the product over the included factors and
    `factors` records which those were.
    """

    task: str
    cell_id: str
    vis: float
    veracity: float
    influence: float
    sobriety: float
    factors: str = "VIS"                  # identity of the score in `vis`
    excluded: tuple[str, ...] = ()
    # inputs, kept because every factor is only interpretable next to its units
    d_tol_grade: float = float("nan")     # latent image of the GRADING tolerance
    d_tol_veracity: float = float("nan")  # the stricter prediction requirement (grade / 2)
    sigma_roll: float = float("nan")
    nmse_ol: float = float("nan")
    tr_sigma_z: float = float("nan")
    m_emp: float = float("nan")           # nats, horizon-consistent noise floor
    m_emp_1step: float = float("nan")     # nats against the raw one-step floor (diagnostic)
    tau: float = _inf.TAU
    p_hat: float = float("nan")
    gap_resolution: float = float("nan")
    horizon: int = 5
    history_size: int = 3
    extras: dict = field(default_factory=dict)

    @property
    def included(self) -> tuple[str, ...]:
        return tuple(f for f in FACTORS if f not in self.excluded)

    @property
    def saturated(self) -> list[str]:
        """Included factors at their ceiling: they cannot separate checkpoints on this task."""
        return [f for f in self.included
                if np.isfinite(getattr(self, f)) and getattr(self, f) >= 1.0 - SATURATION_EPS]

    @property
    def status(self) -> str:
        """"ok", or "abstained" when an included factor could not be measured.

        Abstention is a result, not an error: d_tol is nan when no probe pair sits on the tolerance
        shell. Report it; do not substitute a value.
        """
        return "ok" if all(np.isfinite(getattr(self, f)) for f in self.included) else "abstained"

    def to_dict(self) -> dict:
        """Machine-readable record. Excludes the large arrays in `extras`."""
        d = {k: v for k, v in self.__dict__.items() if k != "extras"}
        d["excluded"] = ";".join(self.excluded)
        d["saturated"] = ";".join(self.saturated)
        d["status"] = self.status
        d.update({k: v for k, v in self.extras.items() if np.isscalar(v)})
        return d

    as_row = to_dict          # kept as an alias: CSV rows and JSON records are the same thing

    def __str__(self) -> str:
        sat = f"  [saturated: {', '.join(self.saturated)}]" if self.saturated else ""
        ab = "  [ABSTAINED: an included factor is nan]" if self.status != "ok" else ""
        lines = [f"{self.cell_id} ({self.task})",
                 f"  {self.factors:9s} {self.vis:.4f}{sat}{ab}"]
        if "veracity" in self.included:
            lines.append(f"  veracity  {self.veracity:.4f}   d_tol {self.d_tol_grade:.3g} "
                         f"(pred req {self.d_tol_veracity:.3g})  sigma_roll {self.sigma_roll:.3g}")
        if "influence" in self.included:
            lines.append(f"  influence {self.influence:.4f}   m_emp {self.m_emp:.1f} / "
                         f"tau {self.tau:.0f} nats")
        if "sobriety" in self.included:
            lines.append(f"  sobriety  {self.sobriety:.4f}   p_hat {self.p_hat:.3f} "
                         f"(gap resolution {self.gap_resolution:.3g})")
        if self.excluded:
            lines.append(f"  excluded: {', '.join(self.excluded)} "
                         f"-> this is {self.factors}, not VIS; do not compare it to VIS numbers")
        return "\n".join(lines)


def combine(veracity: float, influence: float, sobriety: float,
            factors=FACTORS) -> float:
    """Product over the selected factors. Excluded factors do not enter, not even as 1.0."""
    vals = {"veracity": veracity, "influence": influence, "sobriety": sobriety}
    out = 1.0
    for f in factors:
        out *= float(vals[f])
    return float(out)


def score_latents(model, z: np.ndarray, probe: dict, task: TaskSpec | str,
                  cell_id: str = "cell", history_size: int | None = None,
                  tau: float = _inf.TAU, horizon: int | None = None,
                  with_baselines: bool = False, exclude=(), include=None) -> VISFactors:
    """Score an already-encoded latent bank. All included factors share this one bank."""
    task = get_task(task) if isinstance(task, str) else task
    if horizon is not None and horizon != task.horizon:
        task = replace(task, horizon=horizon)
    hs = history_size if history_size is not None else getattr(model, "history_size", 3)
    keep = resolve_factors(exclude, include)

    # sigma_roll is needed by influence as well as by veracity (E_H is anchored to it), so the
    # rollout runs unless BOTH are excluded. Only the d_tol shell is veracity-specific.
    v = dict(veracity=float("nan"), d_tol_grade=float("nan"), d_tol_veracity=float("nan"),
             sigma_roll=float("nan"), nmse_ol=float("nan"), nmse_curve=np.array([]),
             tr_sigma_z=_ver.trace_cov(z))
    if {"veracity", "influence"} & set(keep):
        v = _ver.measure(model, z, probe, task, hs, with_d_tol="veracity" in keep)
    i = dict(influence=float("nan"), m_emp=float("nan"), m_emp_1step=float("nan"),
             S=None, E=None)
    if "influence" in keep:
        i = _inf.measure(model, z, probe, task, hs, v["sigma_roll"], tau=tau)
    s = dict(sobriety=float("nan"), p_hat=float("nan"), gaps=np.array([]),
             gap_resolution=float("nan"))
    if "sobriety" in keep:
        s = _sob.measure(model, z, probe, task, hs, cell_id=cell_id)

    extras = dict(nmse_curve=v["nmse_curve"], gaps=s["gaps"], S=i["S"], E=i["E"])
    if with_baselines:
        from . import baselines
        extras.update(baselines.measure(z, probe, task))
    return VISFactors(
        task=task.name, cell_id=cell_id,
        vis=combine(v["veracity"], i["influence"], s["sobriety"], factors=keep),
        veracity=v["veracity"], influence=i["influence"], sobriety=s["sobriety"],
        factors=label(keep), excluded=tuple(f for f in FACTORS if f not in keep),
        d_tol_grade=v["d_tol_grade"], d_tol_veracity=v["d_tol_veracity"],
        sigma_roll=v["sigma_roll"], nmse_ol=v["nmse_ol"], tr_sigma_z=v["tr_sigma_z"],
        m_emp=i["m_emp"], m_emp_1step=i["m_emp_1step"], tau=tau,
        p_hat=s["p_hat"], gap_resolution=s.get("gap_resolution", float("nan")),
        horizon=task.horizon, history_size=hs, extras=extras,
    )


def score_checkpoint(ckpt: Path | str, probe: dict | str | Path, task: TaskSpec | str,
                     device: str = "cuda", family: str = "lewm", cell_id: str | None = None,
                     latents_dir: Path | str | None = None, **kw) -> VISFactors:
    """Load a checkpoint, encode the probe (or reuse a cached bank), and score it.

    `probe` may be a loaded dict or a path to a probe npz:

        score_checkpoint("vis-wm/pusht/seed403/vis-wm_epoch_9.ckpt", "probes/probe_pusht.npz", "pusht")
    """
    from .adapters import load_model
    from .latents import cached_latents
    from .probe import load_probe

    # A path is loaded WITHOUT pixels and re-read with them only if the latents have to be
    # encoded: on a cache hit that avoids holding a 1-1.8 GB pixel array the metric never reads.
    probe_path = Path(probe) if isinstance(probe, (str, Path)) else None
    if probe_path is not None:
        probe = load_probe(probe_path, with_pixels=False)
    ckpt = Path(ckpt)
    cell_id = cell_id or f"{ckpt.parent.name}__{ckpt.stem}"
    model_holder: list = []

    def get_model():
        if not model_holder:
            model_holder.append(load_model(ckpt, device=device, family=family))
        return model_holder[0]

    def get_probe_with_pixels():
        if "pixels" in probe:
            return probe
        if probe_path is None:
            raise KeyError("probe has no 'pixels' and no path to reload it from; either pass a "
                           "probe path, a probe including pixels, or cached latents")
        return load_probe(probe_path)

    z = cached_latents(get_model, get_probe_with_pixels, latents_dir, cell_id)
    return score_latents(get_model(), z, probe, task, cell_id=cell_id, **kw)


def write_csv(rows: list[VISFactors], out: Path | str) -> Path:
    import csv

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dicts = [r.to_dict() for r in rows]
    keys = list(dict.fromkeys(k for d in dicts for k in d))     # union, first-seen order
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(dicts)
    return out


def write_json(rows: list[VISFactors], out: Path | str) -> Path:
    import json

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.to_dict() for r in rows], indent=2, default=float))
    return out
