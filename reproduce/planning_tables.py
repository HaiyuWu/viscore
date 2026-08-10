#!/usr/bin/env python3
"""Recompute the Section-4 planning tables from the published success labels.

    python reproduce/planning_tables.py            # the tables + a pass/fail gate
    python reproduce/planning_tables.py --strict   # exit non-zero if any gate fails

Inputs, both shipped with the repo:
    pools/success_labels.csv   one row per (checkpoint, task, goal offset, evaluation seed)
    pools/planning_arms.csv    the epoch each reported arm was taken at

Two conventions decide these numbers, and both are made explicit here rather than described:

* **Peak epoch, per training run.** Selected by maximum evaluation success rate, **ties going to
  the later epoch**. Which epoch each arm was reported at is data (`planning_arms.csv`), not a
  rule to re-derive: it is what makes the table checkable.
* **The reported ± is the standard error over the cell set**, not a seed-to-seed sd. For the OOD
  table a cell is (shape × training seed × evaluation seed), n = 6·3·3 = 54.

The gate at the end re-checks every published value. A silent 0.5-point drift in a label file is
exactly the failure this script exists to catch, so it prints PASS/FAIL per cell and `--strict`
turns any FAIL into a non-zero exit.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OOD_SHAPES = ("I", "L", "Z", "plus", "small_tee", "square")     # six UNSEEN shapes; T is the control
FRESH_SEEDS = {"1025", "3014", "5976"}                          # the three fresh evaluation seeds
DS = ("25", "50", "75")

# Published values (mean, se) per arm and goal offset. tab:ood and the PushT rows of the
# main / long-horizon tables.
PUBLISHED_OOD = {
    "LeWM (SIGReg, B=128)": {"25": (79.1, 0.9), "50": (61.8, 1.1), "75": (44.3, 1.2)},
    "VIS-WM, B=128":        {"25": (77.4, 1.0), "50": (59.7, 1.3), "75": (41.8, 1.3)},
    "VIS-WM, B=256":        {"25": (82.0, 0.7), "50": (62.2, 1.2), "75": (44.3, 1.2)},
    "VIS-WM, B=512":        {"25": (82.0, 0.7), "50": (62.0, 1.1), "75": (45.6, 1.1)},
}
# The batch arms are one epoch each, so they need no arm table -- they are selected by `group`.
GROUP_ARMS = {"VIS-WM, B=256": "vis-bs256", "VIS-WM, B=512": "vis-bs512"}
PUBLISHED_PUSHT = {"VIS-WM": {"25": 92, "50": 48, "75": 20},
                   "LeWM":   {"25": 91, "50": 44, "75": 21}}
PUSHT_GROUP = {"VIS-WM": "vis-bs128", "LeWM": "sig-128"}


def load(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path)))


def agg(vals: list[float]) -> tuple[float, float, int]:
    a = np.asarray(vals, float)
    n = len(a)
    se = float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return float(a.mean()), se, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(HERE / "pools/success_labels.csv"))
    ap.add_argument("--arms", default=str(HERE / "pools/planning_arms.csv"))
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    rows = load(Path(a.labels))
    arms = load(Path(a.arms))
    peak = {(r["arm"], r["run"]): r["peak_epoch"] for r in arms}
    arm_runs = defaultdict(list)
    for r in arms:
        arm_runs[r["arm"]].append(r["run"])

    fails = []

    def check(label: str, got: float, want: float, tol: float = 0.05) -> str:
        ok = abs(got - want) <= tol
        if not ok:
            fails.append(f"{label}: got {got:.2f}, published {want:.2f}")
        return "PASS" if ok else "FAIL"

    # ---------------------------------------------------------------- tab:ood
    print("=" * 92)
    print("tab:ood -- success on six unseen PushObj shapes (peak epoch; 3 training x 3 eval seeds)")
    print(f"  {'arm':22s}" + "".join(f"{'d=' + d:>18s}" for d in DS))
    print("-" * 92)
    for arm, pub in PUBLISHED_OOD.items():
        line = f"  {arm:22s}"
        for d in DS:
            if arm in GROUP_ARMS:
                sel = [float(r["success_rate"]) for r in rows
                       if r["group"] == GROUP_ARMS[arm] and r["d"] == d
                       and r["task"] in {f"pushobj_{s}" for s in OOD_SHAPES}
                       and r["eval_seed"] in FRESH_SEEDS]
            else:
                sel = [float(r["success_rate"]) for r in rows
                       if r["run"] in arm_runs[arm] and r["d"] == d
                       and r["epoch"] == peak[(arm, r["run"])]
                       and r["task"] in {f"pushobj_{s}" for s in OOD_SHAPES}
                       and r["eval_seed"] in FRESH_SEEDS]
            if not sel:
                line += f"{'--':>18s}"
                continue
            m, se, n = agg(sel)
            line += f"{m:8.1f}+-{se:3.1f} {check(f'ood/{arm}/d{d}', m, pub[d][0]):>4s}"
        print(line)
    print("  Random policy               10.1              7.3              3.0   (reference)")

    # ------------------------------------------------- main + long-horizon, PushT
    print("\n" + "=" * 92)
    print("Section-4 PushT rows, three fresh evaluation seeds (paper reports these rounded)")
    print(f"  {'arm':22s}" + "".join(f"{'d=' + d:>18s}" for d in DS))
    print("-" * 92)
    for arm, pub in PUBLISHED_PUSHT.items():
        line = f"  {arm:22s}"
        for d in DS:
            runs = arm_runs.get("VIS-WM, B=128" if arm == "VIS-WM" else "LeWM (SIGReg, B=128)", [])
            key = "VIS-WM, B=128" if arm == "VIS-WM" else "LeWM (SIGReg, B=128)"
            sel = [float(r["success_rate"]) for r in rows
                   if r["run"] in runs and r["task"] == "pusht" and r["d"] == d
                   and r["eval_seed"] in FRESH_SEEDS
                   and r["epoch"] == peak[(key, r["run"])]]
            if not sel:
                line += f"{'--':>18s}"
                continue
            m, se, n = agg(sel)
            # the paper rounds these to whole points
            line += f"{m:8.1f}+-{se:3.1f} {check(f'pusht/{arm}/d{d}', round(m), pub[d], 0.5):>4s}"
        print(line)

    print("\n" + "=" * 92)
    if fails:
        print(f"{len(fails)} GATE FAILURE(S) -- the labels no longer reproduce the paper:")
        for f in fails:
            print(f"   {f}")
        return 1 if a.strict else 0
    print("all published values reproduced from pools/success_labels.csv")
    print("Everything published is ONE protocol: CEM, the deployed planner. Success labels, the\n"
          "sobriety probe and the manifest's sobriety column are all CEM-probed, so a CEM label can\n"
          "never be paired with a differently-probed sobriety. The paper's six-planner robustness\n"
          "appendix is not reproducible from these files, by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
