#!/usr/bin/env python3
"""Fetch the third-party checkpoints of the held-out-method pool, ready to score.

    python reproduce/download_external.py --dest $VISCORE_HOME/external
    python reproduce/download_external.py --dest DIR --only qantara --dry-run
    python reproduce/download_external.py --dest DIR --with-code      # also clone each source's repo

These checkpoints were released by other groups and are **not** re-hosted in this project's
Hugging Face repositories, so credit and licensing stay with their authors; cite the papers below
if you use them. This script only automates fetching them from their own releases.

Why a code checkout is also needed. Each release ships its own fork of the LeWM tree, and those
forks define `jepa` / `module` with the same class names as ours. Unpickling one against our
classes binds their weights to different code and fails silently -- the numbers come out wrong
rather than erroring. `--with-code` clones the matching repository next to each download so the
pickle can be loaded against the code it was written by. Scan any third-party pickle
(`python -m pickletools`) before loading it with `weights_only=False`.

The DINO-CLS-WM checkpoints of the same pool are ours; they are published under
`pools/heldout-method/` on the Hub and are not downloaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path

SOURCES = {
    "qantara": dict(
        kind="hf", repo="t-tech/qantara-checkpoints", revision=None,
        code="https://github.com/corl-team/qantara", paper="arXiv:2607.04978",
        contents="24 checkpoints = {qantara, lewm-reproduction} x {pusht, tworoom, cube, reacher} "
                 "x seeds {11, 22, 33}. The lewm-reproduction arms are their baseline, not a new "
                 "method, and are excluded from our pool.",
    ),
    "rcaux": dict(
        kind="hf", repo="biubiu116/RC-aux", revision=None,
        code="https://github.com/Guang000/RC-aux", paper=None,
        contents="10 checkpoints (reacher, cube, tworoom, wall). Their planner adds a reachability "
                 "cost and their non-PushT configs keep the upstream solver.n_steps=30, so their "
                 "published success rates are not comparable to ours -- we re-evaluated under our "
                 "own protocol, and so should you.",
        sha256="SHA256SUMS",
    ),
    "intact": dict(
        kind="hf", repo="INTACT-JEPA/INTACT", revision="paper-e5-goal-v1",
        code="https://github.com/zju3dv/INTACT-JEPA", paper="arXiv:2607.26056",
        contents="18 tarballs: the ablation matrix x 4 tasks x seeds {0, 42, 3072}, all epoch 5. "
                 "Load the pinned evaluator snapshot under paper_runtime/, not the repo root: the "
                 "top-level jepa.py / module.py have drifted from the release hashes.",
        untar=True, sha256="SHA256SUMS",
    ),
    "dinowm": dict(
        kind="osf", repo="bmw48", revision=None,
        code="https://github.com/gaoyuezhou/dino_wm", paper=None,
        contents="PointMaze, PushT, Wall. A different latent (DINOv2 patch features), so the "
                 "influence cap and the d_tol machinery do not transfer unchanged; it is listed "
                 "for completeness and is not part of the reported pool.",
    ),
}


def verify(root: Path, sums: str) -> tuple[int, int]:
    """Check the release's own SHA256SUMS. Returns (ok, failed)."""
    cands = [root / sums] + sorted(root.glob(f"*/{sums}"))
    f = next((c for c in cands if c.exists()), None)
    if f is None:
        return 0, 0
    root = f.parent
    ok = bad = 0
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        want, name = parts[0], parts[1].lstrip("*")
        p = root / name
        if not p.exists():
            continue
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() == want:
            ok += 1
        else:
            bad += 1
            print(f"   CHECKSUM MISMATCH {name}")
    return ok, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="directory to download into")
    ap.add_argument("--only", action="append", default=[], choices=sorted(SOURCES),
                    help="repeatable; default is every source")
    ap.add_argument("--with-code", action="store_true",
                    help="also clone each source's repository, which its pickles need")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    dest = Path(a.dest).expanduser()
    want = a.only or sorted(SOURCES)
    failed = []

    for name in want:
        s = SOURCES[name]
        rev = f" @ {s['revision']}" if s["revision"] else ""
        print(f"\n== {name}: {s['repo']}{rev}  [{s['kind']}]")
        print(f"   code:  {s['code']}")
        if s["paper"]:
            print(f"   paper: {s['paper']}")
        print(f"   {s['contents']}")
        if a.dry_run:
            continue
        out = dest / name
        if s["kind"] == "hf":
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(repo_id=s["repo"], revision=s["revision"],
                                       local_dir=str(out)))
            print(f"   -> {p}")
            if s.get("sha256"):
                ok, bad = verify(p, s["sha256"])
                print(f"   checksums: {ok} ok, {bad} failed")
                if bad:
                    failed.append(name)
            if s.get("untar"):
                tars = sorted(p.glob("*.tar.gz"))
                for t in tars:
                    d = p / t.name.replace(".tar.gz", "")
                    if d.exists():
                        continue
                    with tarfile.open(t) as tf:
                        tf.extractall(d, filter="data")
                print(f"   unpacked {len(tars)} tarball(s)")
        else:
            print("   OSF has no API client here. Download the project archive manually:")
            print(f"      https://osf.io/{s['repo']}/   ->   {out}")
        if a.with_code:
            repo_dir = out / "code"
            if repo_dir.exists():
                print(f"   code already at {repo_dir}")
            else:
                r = subprocess.run(["git", "clone", "--depth", "1", s["code"], str(repo_dir)],
                                   capture_output=True, text=True)
                print(f"   code -> {repo_dir}" if r.returncode == 0
                      else f"   clone FAILED: {r.stderr.strip()[:120]}")

    print("\nThese are other groups' releases: cite their papers and keep their licences with the "
          "files. Load each pickle against the code that wrote it, not against ours.")
    if failed:
        print(f"CHECKSUM FAILURES in: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
