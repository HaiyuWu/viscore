#!/usr/bin/env python3
"""Fetch published artifacts into a local artifact root (`$VISCORE_HOME`).

    python reproduce/download.py --tier bundle    --dest $VISCORE_HOME   # CPU repro, 3.8 GiB
    python reproduce/download.py --tier vis-wm    --dest $VISCORE_HOME   # VIS-WM ckpts, 0.9 GiB
    python reproduce/download.py --tier pools     --dest $VISCORE_HOME   # pool ckpts, 37.8 GiB
    python reproduce/download.py --tier datasets  --dest $VISCORE_HOME   # 268 GB decompressed
    python reproduce/download.py --tier all --dest $VISCORE_HOME --dry-run

Sources:
  models   https://huggingface.co/BooBooWu/viscore
  data     https://huggingface.co/datasets/BooBooWu/viscore          (MAZE, PushObj, bundle)
  base     quentinll/lewm-{pusht,reacher,tworooms,cube}                (LeWorldModel's four,
           collected at https://huggingface.co/collections/quentinll/lewm)

The four base datasets are downloaded from the ORIGINAL LeWorldModel repositories rather than
re-hosted copies. They arrive zstd-compressed and are decompressed in place; the decompressed
footprint is 46 / 99 / 13 / 102 GB, so `--tier datasets` needs ~270 GB free and is the only tier
most people never need. Integrity comes from zstd's own frame checksum plus the Hub's content
hashing, which is why this script carries no sha256 table of its own.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

MODEL_REPO = "BooBooWu/viscore"
DATA_REPO = "BooBooWu/viscore"

# tier -> (repo, repo_type, allow_patterns, path prefix stripped on arrival)
TIERS = {
    "bundle":  (DATA_REPO, "dataset", ["bundle/*"], None),
    "vis-wm":  (MODEL_REPO, "model", ["vis-wm/*", "baselines-lewm/*"], None),
    "pools":   (MODEL_REPO, "model", ["pools/*"], None),
}

# The datasets this work created (in DATA_REPO under data/) plus the four LeWM sources.
# (repo, repo_type, filename in repo, destination relative to --dest)
OURS = [(DATA_REPO, "dataset", "data/maze2d_medium.h5.zst", "maze2d_medium.h5")] + [
    (DATA_REPO, "dataset", f"data/pushobj_{s}.h5.zst", f"pushobj_{s}.h5")
    for s in ("L", "Z", "plus", "I", "small_tee", "square", "T")]
LEWM = [
    ("quentinll/lewm-pusht", "dataset", "pusht_expert_train.h5.zst", "pusht_expert_train.h5"),
    ("quentinll/lewm-reacher", "dataset", "reacher.tar.zst", "dmc/"),
    ("quentinll/lewm-tworooms", "dataset", "tworoom.tar.zst", ""),
    ("quentinll/lewm-cube", "dataset", "cube_single_expert.tar.zst", "ogbench/"),
]


def _hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")
    return hf_hub_download, snapshot_download


def decompress(src: Path, dest_rel: str, dest_root: Path) -> None:
    """zstd file -> plain file, or .tar.zst -> extracted tree. Skips work already done."""
    if src.name.endswith(".tar.zst"):
        target_dir = dest_root / dest_rel
        target_dir.mkdir(parents=True, exist_ok=True)
        tar = src.with_suffix("")                       # strip .zst
        if not tar.exists():
            subprocess.run(["zstd", "-d", "--check", "-f", str(src), "-o", str(tar)], check=True)
        with tarfile.open(tar) as t:
            names = t.getnames()
            if all((target_dir / n).exists() for n in names if n.endswith(".h5")):
                print(f"    already extracted: {dest_rel}")
            else:
                t.extractall(target_dir)                # noqa: S202 -- first-party archive
                print(f"    extracted -> {target_dir}")
        tar.unlink(missing_ok=True)
        return
    out = dest_root / dest_rel
    if out.exists():
        print(f"    already present: {dest_rel}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["zstd", "-d", "--check", "-f", str(src), "-o", str(out)], check=True)
    print(f"    decompressed -> {out} ({out.stat().st_size / 2**30:.1f} GiB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="artifact root ($VISCORE_HOME); any writable path works")
    ap.add_argument("--tier", action="append", default=[],
                    choices=[*TIERS, "datasets", "ours", "all"],
                    help="repeatable; 'ours' = MAZE + PushObj only, 'datasets' adds the four LeWM sets")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tiers = a.tier or ["bundle"]
    if "all" in tiers:
        tiers = [*TIERS, "datasets"]
    dest = Path(a.dest)
    hf_hub_download, snapshot_download = _hub()

    for tier in [t for t in tiers if t in TIERS]:
        repo, rtype, patterns, _ = TIERS[tier]
        print(f"== {tier}: {repo} ({rtype}) {patterns}")
        if a.dry_run:
            continue
        p = snapshot_download(repo_id=repo, repo_type=rtype, allow_patterns=patterns,
                             local_dir=str(dest / "published"))
        print(f"   -> {p}")

    if {"datasets", "ours"} & set(tiers):
        want = OURS + (LEWM if "datasets" in tiers else [])
        print(f"== datasets: {len(want)} sources")
        if not shutil.which("zstd"):
            raise SystemExit("zstd not found; needed to decompress the published datasets")
        for repo, rtype, fname, dest_rel in want:
            print(f"  {repo}:{fname}")
            if a.dry_run:
                continue
            src = Path(hf_hub_download(repo_id=repo, repo_type=rtype, filename=fname))
            decompress(src, dest_rel, dest)

    print("\nExpected layout under --dest: pusht_expert_train.h5, dmc/reacher.h5, tworoom.h5,\n"
          "ogbench/cube_single_expert.h5, maze2d_medium.h5, pushobj_*.h5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
