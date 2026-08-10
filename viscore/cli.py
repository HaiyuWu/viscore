"""Command line: `viscore probe | score | tasks`.

Two stages, because a probe is built once per task and shared by every checkpoint compared, while
scoring is per checkpoint.

Callable by a person or a program: `--json` emits one machine-readable record per checkpoint,
`viscore tasks` prints the task registry, `--exclude` scores any subset of the three factors,
and the exit code is 0 (all scored), 2 (usage error) or 3 (a cell abstained).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .probe import PRESETS, build_preset
from .score import FACTORS, resolve_factors, score_checkpoint, write_csv, write_json
from .tasks import TASKS

EXIT_OK, EXIT_USAGE, EXIT_ABSTAINED = 0, 2, 3


def _epoch_ckpts(run_dir: Path, epochs: str, pattern: str) -> list[Path]:
    """Expand `--epochs 1-10` / `1,3,5` against a run directory's per-epoch dumps.

    `pattern` may be a comma-separated list; each entry is globbed with `{epoch}` substituted, so
    both the published name (`vis-wm_epoch_7.ckpt`) and the training callback's
    (`lewm_epoch_7_object.ckpt`) resolve without configuration.
    """
    eps: list[int] = []
    for part in epochs.split(","):
        if "-" in part:
            a, b = part.split("-")
            eps.extend(range(int(a), int(b) + 1))
        else:
            eps.append(int(part))
    out = []
    for e in eps:
        hits = [h for pat in pattern.split(",")
                for h in sorted(run_dir.glob(pat.strip().format(epoch=e)))]
        if hits:
            out.append(hits[0])
        else:
            print(f"  missing, skipped: epoch {e} in {run_dir}", file=sys.stderr)
    return out


def _resolve_probe(a) -> Path:
    """`--probe` may be a file, or a directory holding probe_<task>.npz.

    A missing probe is built when `--data-home` is given and the task has a preset; otherwise it is a
    hard error, because scoring against the wrong probe is worse than stopping.
    """
    p = Path(a.probe)
    if p.is_dir():
        p = p / f"probe_{a.task}.npz"
    if p.exists():
        return p
    if a.data_home and a.task in PRESETS:
        print(f"probe not found, building it: {p}")
        return Path(build_preset(a.task, a.data_home, p.parent))
    raise SystemExit(f"probe not found: {p}\n"
                     f"build it first:  viscore probe --tasks {a.task} "
                     f"--data-home <datasets> --out-dir {p.parent}")


def cmd_probe(a) -> int:
    # VISCORE_HOME is this project's name for the artifact root; STABLEWM_HOME is the same
    # path under the name the pinned stable-worldmodel reads, and is accepted as a fallback
    data_home = Path(a.data_home or os.environ.get("VISCORE_HOME")
                     or os.environ.get("STABLEWM_HOME", "."))
    for task in a.tasks:
        build_preset(task, data_home, Path(a.out_dir), seed=a.seed)
    return EXIT_OK


def cmd_tasks(a) -> int:
    """The task registry: what a task means and what its probe must contain."""
    rows = []
    for name, t in sorted(TASKS.items()):
        rows.append(dict(task=name, tolerance=t.tolerance, metric=t.metric,
                         horizon=t.horizon, rollout_h_max=t.rollout_h_max,
                         has_nuisance=t.nuisance(_KeyRecorder()) is not None,
                         probe_preset=PRESETS.get(name, {}).get("h5", "-"),
                         probe_keys=sorted(_probe_keys(t))))
    if a.json:
        print(json.dumps(rows, indent=2, default=str))
        return EXIT_OK
    print(f"{'task':10s} {'tol':>8s} {'metric':6s} {'H':>3s} {'nuis':>5s}  probe keys")
    for r in rows:
        print(f"{r['task']:10s} {r['tolerance']:8g} {r['metric']:6s} {r['horizon']:3d} "
              f"{'yes' if r['has_nuisance'] else '  -':>5s}  {', '.join(r['probe_keys'])}")
    print("\nTolerances are read off each environment's success criterion, never tuned on labels.\n"
          "Add your own: viscore.tasks.TASKS['name'] = TaskSpec(...)  (see viscore/tasks.py)")
    return EXIT_OK


class _KeyRecorder(dict):
    """Records which probe keys a TaskSpec's accessors touch, so `tasks` can report them."""

    def __init__(self):
        super().__init__()
        self.seen: set[str] = set()

    def __getitem__(self, k):
        self.seen.add(k)
        import numpy as np
        return np.zeros((4, 8), dtype=np.float32)      # wide enough for any accessor's slicing


def _probe_keys(task) -> set[str]:
    rec = _KeyRecorder()
    for fn in (task.scored, task.nuisance, task.probe_target):
        if fn is None:
            continue
        try:
            fn(rec)
        except Exception:                              # noqa: BLE001 -- slicing a stand-in
            pass
    return rec.seen | {"pixels", "act_blocks", "ep_ptr"}


def cmd_score(a) -> int:
    try:
        factors = resolve_factors(a.exclude)
    except ValueError as e:
        print(f"usage error: {e}", file=sys.stderr)
        return EXIT_USAGE

    # Pass the PATH, not a loaded dict: score_checkpoint then reads the pixel array only when it
    # actually has to encode, which a cached-latents re-score never does.
    probe_path = _resolve_probe(a)
    ckpts = [Path(c) for c in a.ckpt]
    if a.run_dir:
        ckpts += _epoch_ckpts(Path(a.run_dir), a.epochs, a.ckpt_pattern)
    if not ckpts:
        print("no checkpoints given (--ckpt and/or --run-dir + --epochs)", file=sys.stderr)
        return EXIT_USAGE

    if a.exclude:
        print(f"scoring {len(factors)} of 3 factors ({''.join(f[0].upper() for f in FACTORS if f in factors)});"
              f" excluded: {', '.join(sorted(set(FACTORS) - set(factors)))}", file=sys.stderr)

    rows = []
    for ck in ckpts:
        f = score_checkpoint(ck, probe_path, a.task, device=a.device, family=a.family,
                             latents_dir=a.latents_dir, tau=a.tau, horizon=a.horizon,
                             history_size=a.history_size, with_baselines=a.baselines,
                             exclude=a.exclude)
        if a.json:
            print(json.dumps(f.to_dict(), default=float), flush=True)
        else:
            print(f, flush=True)
            if a.baselines:
                print(f"  baselines  probe_r2 {f.extras['probe_r2']:.3f}   "
                      f"straightness {f.extras['straightness']:+.3f}")
        rows.append(f)

    if a.csv:
        print(f"wrote {write_csv(rows, a.csv)}  ({len(rows)} cells)", file=sys.stderr)
    if a.json_out:
        print(f"wrote {write_json(rows, a.json_out)}  ({len(rows)} cells)", file=sys.stderr)
    abstained = [r.cell_id for r in rows if r.status != "ok"]
    if abstained:
        print(f"{len(abstained)} cell(s) ABSTAINED (an included factor could not be measured): "
              f"{', '.join(abstained[:5])}", file=sys.stderr)
        return EXIT_ABSTAINED
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="viscore", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("probe", help="build a frozen probe cache from an HDF5 dataset")
    pp.add_argument("--tasks", nargs="+", required=True, choices=sorted(PRESETS))
    pp.add_argument("--data-home", default=None, help="dataset root (default: $VISCORE_HOME)")
    pp.add_argument("--out-dir", required=True)
    pp.add_argument("--seed", type=int, default=0, help="episode-sampling seed; keep 0")
    pp.set_defaults(fn=cmd_probe)

    pt = sub.add_parser("tasks", help="list the task registry (tolerances, required probe keys)")
    pt.add_argument("--json", action="store_true")
    pt.set_defaults(fn=cmd_tasks)

    ps = sub.add_parser("score", help="score one or more checkpoints")
    ps.add_argument("--ckpt", nargs="*", default=[])
    ps.add_argument("--run-dir", default=None, help="score a run's per-epoch dumps")
    ps.add_argument("--epochs", default="1-10", help="with --run-dir, e.g. 1-10 or 1,5,9")
    ps.add_argument("--ckpt-pattern",
                    default="*_epoch_{epoch}.ckpt,lewm_epoch_{epoch}_object.ckpt",
                    help="comma-separated filename patterns; {epoch} is substituted and each is "
                         "globbed (default matches both the published and the training names)")
    ps.add_argument("--task", required=True, choices=sorted(TASKS))
    ps.add_argument("--probe", required=True,
                    help="probe npz, or a directory holding probe_<task>.npz")
    ps.add_argument("--data-home", default=None,
                    help="if the probe is missing, build it from this dataset root")
    ps.add_argument("--exclude", nargs="+", default=[], metavar="FACTOR",
                    help="factors to skip: veracity|influence|sobriety (or v|i|s). The result is "
                         "the product of the rest, labelled VI/VS/IS/V/I/S -- NOT comparable to "
                         "a full VIS number.")
    ps.add_argument("--family", default="lewm", help="model adapter family")
    ps.add_argument("--device", default="cuda")
    ps.add_argument("--latents-dir", default=None, help="cache encoded latents here")
    ps.add_argument("--tau", type=float, default=None, help="influence cap in nats (default 82)")
    ps.add_argument("--horizon", type=int, default=None,
                    help="diagnostic horizon in action blocks; 5/10/15 = eval d 25/50/75")
    ps.add_argument("--history-size", type=int, default=None,
                    help="override the predictor context length (auto-detected by default)")
    ps.add_argument("--baselines", action="store_true", help="also report probe R2/straightness")
    ps.add_argument("--json", action="store_true", help="one JSON record per line on stdout")
    ps.add_argument("--csv", default=None)
    ps.add_argument("--json-out", default=None, help="write all records to a JSON file")
    ps.set_defaults(fn=cmd_score)

    a = p.parse_args(argv)
    if a.cmd == "score" and a.tau is None:
        from .influence import TAU
        a.tau = TAU
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
