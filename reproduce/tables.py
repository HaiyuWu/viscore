#!/usr/bin/env python3
"""Recompute tab:viscore from the published pool manifest. No GPU, no checkpoints, no dataset.

    python reproduce/tables.py            # the three blocks, as printed in the paper
    python reproduce/tables.py --latex    # LaTeX rows
    python reproduce/tables.py --boot 2000    # add run-clustered bootstrap intervals

Everything it needs is in `reproduce/pools/pool_manifest.csv`: per checkpoint, the seven metric
values, the success labels of each pool, and pool membership. That file is the *definition* of the
pools. `pool_assignment.csv` records the split it is derived from, which is at the level of the
training run.

Three estimator choices, all fixed:

* **within-task ρ** is Spearman, so it is invariant to any monotone rescaling of a metric.
* **pooled ρ** is Spearman over the three graded tasks jointly. This is the cross-task axis, and
  it is where a metric measured in task-specific units (raw empowerment, in nats) fails while a
  tolerance-anchored one does not.
* **calibration error** is leave-one-task-out isotonic regression, in success points: fit
  metric → success on two tasks, predict the third, average |error|. Each block prints its own
  **constant reference** (predict a task's success by the median of the other tasks) because
  absolute MAEs are not comparable across pools -- the pools span different success ranges, so a
  calibration number only means something next to that pool's reference.

Cube is reported in parentheses and excluded from pooled and calibration: its label spread does not
exceed its own binomial standard error, so nothing there is rankable by any metric.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

GRADED = ("pusht", "dmc", "tworoom")          # tasks entering pooled / calibration
ALL = GRADED + ("cube",)
NAME = {"pusht": "PushT", "dmc": "Reacher", "tworoom": "Two-Room", "cube": "Cube"}
ROWS = [("vis", "VIScore"), ("m_emp", "empowerment (raw)"), ("straightness", "straightness"),
        ("probe_r2", "state probe $R^2$"), ("veracity", "\\;\\; veracity"),
        ("influence", "\\;\\; influence"), ("sobriety", "\\;\\; sobriety")]
POOLS = [("development (constants fitted here)", "in_development", "sr_development"),
         ("held-out checkpoints (runs disjoint from dev)", "in_heldout", "sr_heldout")]
# The two transfer pools are scored differently: one metric-to-success map, fitted on development
# and applied without refitting, because a new task or a new method arrives without labels. They
# also carry too few checkpoints per task for a within-task correlation, so only the
# cross-checkpoint columns are defined.
FROZEN = [("held-out world-modeling methods (frozen development calibration)",
           "in_heldout_method", "sr_heldout_method"),
          ("held-out dataset: unseen MAZE task (frozen development calibration)",
           "in_heldout_dataset", "sr_heldout_dataset")]
RNG = np.random.default_rng(0)


def load(path: Path) -> list[dict]:
    out = []
    for r in csv.DictReader(open(path)):
        d = {k: (float(v) if v not in ("", "nan") else np.nan) for k, v in r.items()
             if k not in ("run", "env", "fold", "method")}
        d.update(run=r["run"], env=r["env"], method=r.get("method", ""))
        out.append(d)
    return out


def rho(cells, key, label) -> float:
    x = np.array([c[key] for c in cells], float)
    y = np.array([c[label] for c in cells], float)
    k = ~(np.isnan(x) | np.isnan(y))
    if k.sum() < 4 or len(set(np.round(x[k], 9))) < 2:
        return np.nan                          # constant metric: correlation undefined, not 0
    return float(spearmanr(x[k], y[k]).statistic)


def loto(cells, key, label) -> float:
    """Leave-one-task-out isotonic calibration error, in success points."""
    errs = []
    for e in GRADED:
        tr = [c for c in cells if c["env"] != e and c["env"] in GRADED and np.isfinite(c[key])]
        te = [c for c in cells if c["env"] == e and np.isfinite(c[key])]
        if len(te) < 4 or len(tr) < 8:
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit([c[key] for c in tr],
                                                           [c[label] for c in tr])
        errs.append(np.mean(np.abs(iso.predict([c[key] for c in te])
                                   - np.array([c[label] for c in te]))))
    return float(np.mean(errs)) if errs else np.nan


def const_ref(cells, label) -> float:
    errs = []
    for e in GRADED:
        te = [c[label] for c in cells if c["env"] == e]
        ot = [c[label] for c in cells if c["env"] != e and c["env"] in GRADED]
        if len(te) >= 4 and len(ot) >= 8:
            errs.append(np.mean(np.abs(np.array(te) - np.median(ot))))
    return float(np.mean(errs)) if errs else np.nan


def cluster_boot(cells, key, label, stat, n_boot: int):
    """Resample TRAINING RUNS with replacement, carrying all of a run's checkpoints.

    Epoch checkpoints of one run are not independent samples -- development has 137 cells from 14
    runs -- so resampling checkpoints would understate every interval.
    """
    by = defaultdict(list)
    for c in cells:
        by[c["run"]].append(c)
    runs = list(by)
    vals = []
    for _ in range(n_boot):
        pick = RNG.choice(len(runs), size=len(runs), replace=True)
        samp = [c for i in pick for c in by[runs[i]]]
        if len({c["env"] for c in samp} & set(GRADED)) < len(GRADED):
            continue
        v = stat(samp, key, label)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < n_boot // 4:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(Path(__file__).parent / "pools/pool_manifest.csv"))
    ap.add_argument("--boot", type=int, default=0, help="run-clustered bootstrap draws (0 = off)")
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    cells_all = load(Path(a.manifest))

    for tag, flag, label in POOLS:
        cs = [c for c in cells_all if c[flag] == 1 and np.isfinite(c[label])]
        per = defaultdict(list)
        for c in cs:
            per[c["env"]].append(c)
        # Cube is in no pool -- it never entered the run-level split -- but its column is shown in
        # parentheses for reference, so it is read from its own flag and excluded from n.
        per["cube"] = [c for c in cells_all if c["in_cube_reference"] == 1 and np.isfinite(c[label])]
        present = [e for e in ALL if len(per[e]) >= 4]
        # Cube is a reference column, not a pool member, so it is not counted in n
        n = "/".join(str(len(per[e])) for e in present if e in GRADED)
        core = [c for c in cs if c["env"] in GRADED]
        print("=" * 104)
        print(f"{tag}\n  n = {n} on {'/'.join(NAME[e] for e in present)}"
              f"   ({len(cs)} checkpoints, {len({c['run'] for c in cs})} runs)")
        head = f"  {'metric':22s}" + "".join(f"{NAME[e]:>11s}" for e in present)
        head += f"{'Pooled':>9s}{'Calib':>8s}"
        if a.boot:
            head += f"{'pooled 95% CI':>20s}{'calib 95% CI':>18s}"
        print(head)
        print("-" * 104)
        for key, lab in ROWS:
            line = f"  {lab.replace(chr(92) + ';', ' '):22s}"
            for e in present:
                r = rho(per[e], key, label)
                if not np.isfinite(r):
                    line += f"{'--':>11s}"
                elif e == "cube":                   # parenthesised: excluded from pooled/calib
                    line += f"{'(' + format(r, '+.2f') + ')':>11s}"
                else:
                    line += f"{r:>+11.2f}"
            line += f"{rho(core, key, label):>+9.2f}{loto(core, key, label):>8.1f}"
            if a.boot:
                plo, phi = cluster_boot(core, key, label, rho, a.boot)
                clo, chi = cluster_boot(core, key, label, loto, a.boot)
                line += f"   [{plo:+.2f}, {phi:+.2f}]     [{clo:.1f}, {chi:.1f}]"
            print(line)
        print(f"  {'constant reference':22s}" + " " * (11 * len(present))
              + f"{'--':>9s}{const_ref(core, label):>8.1f}")

    # --------------------------------------------------------------- the two transfer pools
    # Ranking is the Spearman over the pool's own cells; the calibration map is the isotonic fit
    # on EVERY development-labelled cell of the three fitted tasks (`in_calibration_fit`),
    # applied here without refitting, so the error is zero-shot rather than leave-one-task-out.
    fit = [c for c in cells_all if c["in_calibration_fit"] == 1
           and np.isfinite(c["sr_development"])]
    for tag, flag, label in FROZEN:
        cs = [c for c in cells_all if c[flag] == 1 and np.isfinite(c[label])
              and c["env"] in GRADED + ("maze2d",)]
        if not cs or not fit:
            continue
        print("=" * 104)
        print(f"{tag}\n  n = {len(cs)} checkpoints from {len({c['run'] for c in cs})} runs   "
              f"(calibration fitted on {len(fit)} cells from {len({c['run'] for c in fit})} runs)")
        print(f"  {'metric':22s}{'Pooled':>11s}{'Calib':>8s}")
        print("-" * 104)
        for key, lab in ROWS:
            g = [c for c in cs if np.isfinite(c[key])]
            r = rho(g, key, label)
            tr = [c for c in fit if np.isfinite(c[key])]
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                [c[key] for c in tr], [c["sr_development"] for c in tr])
            mae = float(np.mean(np.abs(iso.predict([c[key] for c in g])
                                      - np.array([c[label] for c in g]))))
            line = f"  {lab.replace(chr(92) + ';', ' '):22s}"
            line += f"{r:>+11.2f}" if np.isfinite(r) else f"{'const':>11s}"
            print(line + f"{mae:>8.1f}")
        const = float(np.mean(np.abs(np.array([c[label] for c in cs])
                                    - np.median([c["sr_development"] for c in fit]))))
        print(f"  {'constant reference':22s}{'--':>11s}{const:>8.1f}")

    if a.latex:
        print("\n" + "=" * 104 + "\nLATEX\n")
        for tag, flag, label in POOLS:
            cs = [c for c in cells_all if c[flag] == 1 and np.isfinite(c[label])]
            core = [c for c in cs if c["env"] in GRADED]
            per = defaultdict(list)
            for c in cs:
                per[c["env"]].append(c)
            n = "/".join(str(len(per[e])) for e in GRADED)
            print(f"    \\multicolumn{{7}}{{l}}{{\\emph{{{tag}}} \\; ($n={n}$; "
                  f"{len(cs)} checkpoints from {len({c['run'] for c in cs})} runs)}} \\\\")
            for key, lab in ROWS:
                cols = []
                for e in ALL:
                    r = rho(per[e], key, label) if len(per[e]) >= 4 else np.nan
                    if not np.isfinite(r):
                        cols.append("---")
                    else:
                        cols.append(f"$({r:+.2f})$" if e == "cube" else f"${r:+.2f}$")
                hl = r"\rowcolor{green!10}" if key == "vis" else ""
                print(f"    {hl}{lab} & {' & '.join(cols)} & ${rho(core, key, label):+.2f}$ & "
                      f"${loto(core, key, label):.1f}$ \\\\")
            print(f"    \\emph{{Ref.: LOTO constant predictor}} & & & & & & "
                  f"\\emph{{{const_ref(core, label):.1f}}} \\\\")
            print(r"    \midrule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
