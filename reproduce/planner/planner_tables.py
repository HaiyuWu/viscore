#!/usr/bin/env python3
"""Recompute tab:planner-transfer from the frozen planner table. No GPU, no checkpoints.

    python reproduce/planner/planner_tables.py
    python reproduce/planner/planner_tables.py --latex

Everything comes from `planner_cells.csv`: 36 checkpoints, the planner-independent metrics, one
sobriety per probe family, and one success rate per planner.

This experiment is deliberately separate from the pools of `reproduce/tables.py`. Every planner
arm must be evaluated on the SAME checkpoints for the comparison to be paired, so it uses a fixed
36-checkpoint set: all 25 of its training runs also appear in the main pools, 20 of the 36 are
themselves cells of the development or held-out pool, and the other 16 are further epochs of those
same runs. It is therefore a subset of the paper's runs rather than of its reported checkpoints,
and these numbers are not comparable to `reproduce/tables.py`'s.

Sobriety is the one factor containing a search, so each planner needs a probe. The assignment is
fixed by a rule declared before any outcome was seen -- same search family, matched budget
fraction -- not by whichever probe correlates best. Under the declared rule predictive sampling
scores +0.51; under the probe that would have flattered it, +0.80.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

GRADED = ("pusht", "dmc", "tworoom")
NAME = {"pusht": "PushT", "dmc": "Reacher", "tworoom": "Two-Room"}
NUM_EVAL = 50
# (planner key, probe family, printed name)
PLANNERS = [("cem", "mcem", "CEM (300 samples; 30/10/10 iterations)"),
            ("mppi", "mcem", "MPPI (300 samples, 10 iterations, softmax tau=0.5)"),
            ("icem", "mcem", "iCEM (300 samples, 10 iterations, colored noise beta=2)"),
            ("ps", "mcem", "predictive sampling (300 samples, 1 iteration)"),
            ("ms", "madam", "gradient (AdamW, 100 initialisations, 30 steps)"),
            ("ss", "ss", "single-start gradient (AdamW, 1 initialisation, 100 steps)")]
ROWS = [("straightness", "Straightness"), ("probe_r2", "Physical-state probe (R2)"),
        ("m_emp", "Empowerment m_emp"), ("veracity", "  veracity"),
        ("influence", "  influence"), ("sobriety", "  sobriety"), ("vis", "VIScore")]


def load(path: Path) -> list[dict]:
    out = []
    for r in csv.DictReader(open(path)):
        d = {k: (float(v) if v not in ("", "nan") else np.nan)
             for k, v in r.items() if k not in ("run", "env")}
        d.update(run=r["run"], env=r["env"])
        out.append(d)
    return out


def arm(cells, key, probe):
    """The cells as seen by one planner: its labels, and the sobriety of its own probe."""
    out = []
    for c in cells:
        sob = c[f"sobriety_{probe}"]
        out.append(dict(env=c["env"], sr=c[f"sr_{key}"], sobriety=sob,
                        vis=c["veracity"] * c["influence"] * sob,
                        **{m: c[m] for m in ("straightness", "probe_r2", "m_emp",
                                             "veracity", "influence")}))
    return [c for c in out if np.isfinite(c["sr"])]


def rho(cs, k):
    x = np.array([c[k] for c in cs]); y = np.array([c["sr"] for c in cs])
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 4 or len(set(np.round(x[m], 9))) < 2:
        return np.nan
    return float(spearmanr(x[m], y[m]).statistic)


def loto(cs, k):
    errs = []
    for t in GRADED:
        tr = [c for c in cs if c["env"] != t and np.isfinite(c[k])]
        te = [c for c in cs if c["env"] == t and np.isfinite(c[k])]
        if len(te) < 4 or len(tr) < 8:
            continue
        iso = IsotonicRegression(out_of_bounds="clip").fit([c[k] for c in tr],
                                                          [c["sr"] for c in tr])
        errs.append(np.mean(np.abs(iso.predict([c[k] for c in te])
                                   - np.array([c["sr"] for c in te]))))
    return float(np.mean(errs)) if errs else np.nan


def resolvable(cs, env):
    """sd between checkpoints against the binomial SE of one label: below one, unrankable."""
    y = [c["sr"] for c in cs if c["env"] == env]
    if len(y) < 3:
        return np.nan
    p = np.clip(np.mean(y) / 100, 1e-6, 1 - 1e-6)
    return float(np.std(y, ddof=1) / (100 * np.sqrt(p * (1 - p) / NUM_EVAL)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=str(Path(__file__).parent / "planner_cells.csv"))
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    cells = load(Path(a.cells))
    print(f"{len(cells)} checkpoints   "
          + ", ".join(f"{NAME[e]} {sum(1 for c in cells if c['env']==e)}" for e in GRADED))

    for key, probe, title in PLANNERS:
        cs = arm(cells, key, probe)
        print("=" * 96)
        print(f"{title}   [{probe} probe]   n = {len(cs)}")
        res = {e: resolvable(cs, e) for e in GRADED}
        print("  label resolvability sd/SE: "
              + "  ".join(f"{NAME[e]} {res[e]:.1f}{'' if res[e] > 1 else ' (dagger)'}"
                          for e in GRADED))
        print(f"  {'metric':26s}" + "".join(f"{NAME[e]:>11s}" for e in GRADED)
              + f"{'Pooled':>9s}{'Calib':>8s}")
        print("-" * 96)
        for k, lab in ROWS:
            line = f"  {lab:26s}"
            for e in GRADED:
                r = rho([c for c in cs if c["env"] == e], k)
                line += f"{r:>+11.2f}" if np.isfinite(r) else f"{'--':>11s}"
            print(line + f"{rho(cs, k):>+9.2f}{loto(cs, k):>8.1f}")

    if a.latex:
        print("\n" + "=" * 96 + "\nLATEX\n")
        for key, probe, title in PLANNERS:
            cs = arm(cells, key, probe)
            print(f"    \\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\")
            for k, lab in ROWS:
                cols = []
                for e in GRADED:
                    r = rho([c for c in cs if c["env"] == e], k)
                    dag = "^{\\dagger}" if resolvable(cs, e) <= 1 else ""
                    cols.append("---" if not np.isfinite(r) else f"${r:+.2f}{dag}$")
                hl = "\\rowcolor{green!10}" if k == "vis" else ""
                nm = "VIScore" if k == "vis" else lab.replace("  ", "\;\;")
                print(f"    {hl}{nm} & {' & '.join(cols)} & ${rho(cs,k):+.2f}$ & "
                      f"${loto(cs,k):.1f}$ \\\\")
            print("    \\midrule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
